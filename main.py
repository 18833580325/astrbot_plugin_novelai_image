import asyncio
import base64
import json
import random
import re
import shlex
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import httpx
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.message import ImageURLPart, TextPart, UserMessageSegment


RATIO_TABLE = {
    "1:1": (1024, 1024),
    "2:3": (832, 1216),
    "3:2": (1216, 832),
    "3:5": (768, 1280),
    "5:3": (1280, 768),
    "9:16": (768, 1344),
    "16:9": (1344, 768),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
}

MODEL_OPTIONS = {
    "v45": "nai-diffusion-4-5-full",
    "v45-full": "nai-diffusion-4-5-full",
    "v45-curated": "nai-diffusion-4-5-curated",
    "v4": "nai-diffusion-4-full",
    "v4-full": "nai-diffusion-4-full",
    "v4-curated": "nai-diffusion-4-curated-preview",
    "v3": "nai-diffusion-3",
    "furry": "nai-diffusion-furry-3",
}


@dataclass
class NovelAIRequest:
    prompt: str
    negative_prompt: str
    model: str
    width: int
    height: int
    steps: int
    scale: float
    seed: int
    sampler: str


@register(
    "astrbot_plugin_novelai_image",
    "Codex",
    "Generate images with NovelAI.",
    "0.1.0",
)
class NovelAIImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._astrbot_config_file = Path("/AstrBot/data/cmd_config.json")

    @filter.command("nai", alias={"novelai", "nai画图", "nai生图"})
    async def generate(self, event: AstrMessageEvent):
        api_key = self._clean_api_key(str(self.config.get("api_key", "")).strip())
        if not api_key:
            yield event.plain_result("NovelAI API Key 还没配置，请先在插件配置里填写 api_key。")
            return

        try:
            request = self._parse_request(event.message_str)
        except ValueError as exc:
            yield event.plain_result(str(exc))
            return

        yield event.plain_result(
            f"已收到，正在用 NovelAI 生成图片：{request.width}x{request.height}，"
            f"steps={request.steps}，scale={request.scale}"
        )

        try:
            image_bytes = await self._call_novelai(api_key, request)
        except Exception as exc:
            logger.error(f"NovelAI image generation failed: {exc}")
            yield event.plain_result(f"生成失败：{exc}")
            return

        try:
            allowed, reason = await self._review_image(image_bytes, request)
        except Exception as exc:
            logger.error(f"NovelAI image review failed: {exc}")
            if bool(self.config.get("vision_review_fail_closed", False)):
                yield event.plain_result(f"图片审核失败，已停止发送：{exc}")
                return
            allowed, reason = True, ""

        if not allowed:
            yield event.plain_result(reason or str(self.config.get("vision_block_reply", "图片未通过审核，已停止发送。")))
            return

        yield event.image_result(self._bytes_to_onebot_image(image_bytes))

    @filter.command("nai_help", alias={"novelai_help", "nai帮助"})
    async def help(self, event: AstrMessageEvent):
        yield event.plain_result(
            "\n".join(
                [
                    "NovelAI 画图帮助",
                    "",
                    "基础用法：",
                    "/nai 1girl, white hair, red eyes, masterpiece",
                    "",
                    "常用参数：",
                    "/nai --ratio 2:3 提示词",
                    "/nai --ratio 3:5 提示词",
                    "/nai --size 832x1216 提示词",
                    "/nai --steps 28 --scale 5.5 提示词",
                    "/nai --seed 123456789 提示词",
                    "/nai --sampler k_euler_ancestral 提示词",
                    "/nai --model v45 提示词",
                    "/nai --uc lowres, bad anatomy 提示词",
                    "",
                    "支持比例：1:1、2:3、3:2、3:5、5:3、9:16、16:9、4:3、3:4",
                    "模型简写：v45、v45-curated、v4、v4-curated、v3、furry",
                    "",
                    "例子：",
                    "/nai --ratio 2:3 --steps 28 --scale 5.5 1girl, white hair, red eyes, best quality",
                ]
            )
        )

    def _parse_request(self, message: str) -> NovelAIRequest:
        text = re.sub(r"^[/！!]?(nai画图|nai生图|novelai|nai)\s*", "", message, flags=re.I).strip()
        if not text:
            raise ValueError("请在命令后写提示词，例如：/nai 1girl, white hair, red eyes")

        try:
            parts = shlex.split(text)
        except ValueError as exc:
            raise ValueError(f"参数解析失败：{exc}") from exc

        prompt_parts: list[str] = []
        model = str(self.config.get("model", "nai-diffusion-4-5-full")).strip() or "nai-diffusion-4-5-full"
        width = int(self.config.get("width", 832))
        height = int(self.config.get("height", 1216))
        steps = int(self.config.get("steps", 28))
        scale = float(self.config.get("scale", 5.5))
        seed = int(self.config.get("seed", -1))
        sampler = str(self.config.get("sampler", "k_euler_ancestral")).strip() or "k_euler_ancestral"
        negative_prompt = str(self.config.get("negative_prompt", "")).strip()

        index = 0
        while index < len(parts):
            token = parts[index]
            if token in {"--ratio", "-r"}:
                index += 1
                ratio = self._need_value(parts, index, token)
                if ratio not in RATIO_TABLE:
                    raise ValueError(f"不支持的比例：{ratio}。可用：{', '.join(RATIO_TABLE)}")
                width, height = RATIO_TABLE[ratio]
            elif token == "--size":
                index += 1
                width, height = self._parse_size(self._need_value(parts, index, token))
            elif token == "--model":
                index += 1
                model = self._normalize_model(self._need_value(parts, index, token))
            elif token == "--steps":
                index += 1
                steps = int(self._need_value(parts, index, token))
            elif token == "--scale":
                index += 1
                scale = float(self._need_value(parts, index, token))
            elif token == "--seed":
                index += 1
                seed = int(self._need_value(parts, index, token))
            elif token == "--sampler":
                index += 1
                sampler = self._need_value(parts, index, token)
            elif token in {"--uc", "--negative", "--negative-prompt"}:
                index += 1
                negative_prompt = self._need_value(parts, index, token)
            else:
                prompt_parts.append(token)
            index += 1

        prompt = " ".join(prompt_parts).strip()
        if not prompt:
            raise ValueError("提示词为空，请补充要生成的内容。")

        self._validate_dimensions(width, height)
        steps = max(1, min(50, steps))
        scale = max(0.1, min(20.0, scale))
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        return NovelAIRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=self._normalize_model(model),
            width=width,
            height=height,
            steps=steps,
            scale=scale,
            seed=seed,
            sampler=sampler,
        )

    def _need_value(self, parts: list[str], index: int, option: str) -> str:
        if index >= len(parts):
            raise ValueError(f"{option} 后面缺少参数。")
        return parts[index]

    def _normalize_model(self, model: str) -> str:
        return MODEL_OPTIONS.get(model.strip().lower(), model.strip())

    def _parse_size(self, value: str) -> tuple[int, int]:
        match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", value.lower())
        if not match:
            raise ValueError("尺寸格式应为 832x1216 这种形式。")
        return int(match.group(1)), int(match.group(2))

    def _validate_dimensions(self, width: int, height: int):
        if width < 64 or height < 64:
            raise ValueError("宽高不能小于 64。")
        if width > 2048 or height > 2048:
            raise ValueError("宽高暂时限制在 2048 以内，避免请求过大。")
        if width % 64 != 0 or height % 64 != 0:
            raise ValueError("NovelAI 宽高建议使用 64 的倍数，例如 832x1216。")

    async def _call_novelai(self, api_key: str, request: NovelAIRequest) -> bytes:
        base_url = str(self.config.get("base_url", "https://image.novelai.net/ai/generate-image")).strip()
        endpoint = base_url or "https://image.novelai.net/ai/generate-image"
        timeout = float(self.config.get("timeout_seconds", 180))
        proxy_url = str(self.config.get("proxy_url", "")).strip() or None

        body = self._build_payload(request, endpoint)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Origin": "https://novelai.net",
            "Referer": "https://novelai.net/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        transport = httpx.AsyncHTTPTransport(proxy=proxy_url) if proxy_url else None
        async with httpx.AsyncClient(timeout=timeout, transport=transport, trust_env=False) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            if response.status_code >= 400:
                raise RuntimeError(f"API Error {response.status_code}: {response.text[:1000]}")
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                return await self._parse_sse_response(response.text, client, headers)
            if "application/json" in content_type or endpoint.endswith("/chat/completions"):
                return await self._parse_json_response(response, client, headers)
            return self._parse_binary_response(response.content)

    def _build_payload(self, request: NovelAIRequest, endpoint: str) -> dict:
        if "/chat/completions" in endpoint:
            return {
                "model": request.model,
                "messages": [{"role": "user", "content": request.prompt}],
                "stream": True,
                "width": request.width,
                "height": request.height,
                "scale": request.scale,
                "steps": request.steps,
                "seed": request.seed,
                "sampler": request.sampler,
                "negative_prompt": request.negative_prompt,
            }

        parameters = {
            "width": int(request.width),
            "height": int(request.height),
            "scale": float(request.scale),
            "steps": int(request.steps),
            "seed": int(request.seed),
            "sampler": request.sampler,
            "negative_prompt": request.negative_prompt,
            "sm": False,
            "sm_dyn": False,
            "n_samples": 1,
        }
        body = {
            "input": request.prompt,
            "model": request.model,
            "action": "generate",
            "parameters": parameters,
        }

        if "nai-diffusion-4" in request.model:
            parameters.pop("sm", None)
            parameters.pop("sm_dyn", None)
            parameters.update(
                {
                    "params_version": 3,
                    "noise_schedule": "karras",
                    "cfg_rescale": 0,
                    "qualityToggle": True,
                    "ucPreset": 4,
                    "uncond_scale": 0,
                    "skip_cfg_below_sigma": 0,
                    "prefer_brownian": True,
                    "use_coords": False,
                    "augment_image": False,
                    "uc": request.negative_prompt,
                    "v4_prompt": {
                        "caption": {"base_caption": request.prompt, "char_captions": []},
                        "use_coords": False,
                        "use_order": True,
                    },
                    "v4_negative_prompt": {
                        "caption": {"base_caption": request.negative_prompt, "char_captions": []}
                    },
                }
            )
        return body

    async def _parse_sse_response(self, text: str, client: httpx.AsyncClient, headers: dict) -> bytes:
        full_content = ""
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                full_content += delta.get("content") or chunk.get("data") or ""
            except Exception:
                full_content += data
        return await self._image_string_to_result(full_content.strip(), client, headers)

    async def _parse_json_response(self, response: httpx.Response, client: httpx.AsyncClient, headers: dict) -> bytes:
        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"JSON 响应解析失败：{exc}") from exc
        image = data.get("data", [{}])[0].get("b64_json") or data.get("data", [{}])[0].get("url")
        if not image:
            raise RuntimeError(f"JSON 响应里没有图片字段：{json.dumps(data, ensure_ascii=False)[:1000]}")
        return await self._image_string_to_result(image, client, headers)

    async def _image_string_to_result(self, image: str, client: httpx.AsyncClient, headers: dict) -> bytes:
        markdown_match = re.search(r"!\[.*?\]\((.*?)\)", image)
        if markdown_match:
            image = markdown_match.group(1)
        if image.startswith("http://") or image.startswith("https://"):
            image_response = await client.get(image, headers=headers)
            image_response.raise_for_status()
            return image_response.content
        if image.startswith("data:image"):
            _, _, payload = image.partition(",")
            return base64.b64decode(payload)
        return base64.b64decode(image)

    def _parse_binary_response(self, content: bytes) -> bytes:
        if content.startswith(b"\x89PNG"):
            return content
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                names = archive.namelist()
                image_names = [name for name in names if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
                target = image_names[0] if image_names else names[0]
                return archive.read(target)
        except Exception as exc:
            raise RuntimeError(f"响应不是 PNG，也无法按 ZIP 解压：{exc}") from exc

    async def _review_image(self, image_bytes: bytes, request: NovelAIRequest) -> tuple[bool, str]:
        if not bool(self.config.get("vision_review_enabled", False)):
            return True, ""

        mode = str(self.config.get("vision_review_mode", "astrbot_caption")).strip() or "astrbot_caption"
        if mode == "off":
            return True, ""

        if mode in {"astrbot_caption", "astrbot_current"}:
            return await self._review_image_with_astrbot_provider(image_bytes, request, mode)

        if mode != "custom_openai":
            raise RuntimeError(f"未知视觉审核模式：{mode}")

        return await self._review_image_with_custom_openai(image_bytes, request)

    async def _review_image_with_astrbot_provider(
        self, image_bytes: bytes, request: NovelAIRequest, mode: str
    ) -> tuple[bool, str]:
        image_data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        provider_id = await self._get_astrbot_review_provider_id(mode)
        if not provider_id:
            raise RuntimeError(
                "没有找到可用的 AstrBot 视觉审核 Provider。"
                "请先在 AstrBot 里配置图片描述模型，或改用 custom_openai。"
            )

        user_msg = UserMessageSegment(
            content=[
                TextPart(text=self._build_review_prompt(request)),
                ImageURLPart(image_url=ImageURLPart.ImageURL(url=image_data_url)),
            ]
        )
        llm_resp = await self.context.llm_generate(
            chat_provider_id=provider_id,
            contexts=[user_msg],
        )
        content = getattr(llm_resp, "completion_text", "") or ""
        review = self._parse_review_json(content)
        return self._review_result_to_tuple(review)

    async def _get_astrbot_review_provider_id(self, mode: str) -> str:
        if mode == "astrbot_current":
            provider_id = await self.context.get_current_chat_provider_id()
            return str(provider_id or "").strip()

        config = self._load_astrbot_config()
        provider_settings = config.get("provider_settings", {}) if isinstance(config, dict) else {}
        provider_id = provider_settings.get("default_image_caption_provider_id", "")
        if provider_id:
            return str(provider_id).strip()

        provider_id = await self.context.get_current_chat_provider_id()
        return str(provider_id or "").strip()

    def _load_astrbot_config(self) -> dict:
        try:
            with open(self._astrbot_config_file, "r", encoding="utf-8-sig") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning(f"Failed to load AstrBot config: {exc}")
            return {}

    def _build_review_prompt(self, request: NovelAIRequest) -> str:
        review_prompt = str(self.config.get("vision_review_prompt", "")).strip()
        if not review_prompt:
            review_prompt = (
                "你是图片发送前的审核器。请判断图片是否适合发送到聊天群。"
                "重点拦截：未成年人或疑似未成年色情、真实人物色情、血腥暴力、违法内容、明显仇恨或极端内容。"
                "如果只是成人二次元性感内容，且没有未成年特征，可以允许。"
                "只返回 JSON，不要返回 Markdown。格式：{\"allow\": true, \"reason\": \"\"}。"
            )
        return (
            f"{review_prompt}\n\n"
            "原始提示词如下，仅作为辅助判断，不要根据提示词臆测图片不存在的内容：\n"
            f"{request.prompt}"
        )

    async def _review_image_with_custom_openai(
        self, image_bytes: bytes, request: NovelAIRequest
    ) -> tuple[bool, str]:
        api_key = self._clean_api_key(str(self.config.get("vision_api_key", "")).strip())
        if not api_key:
            raise RuntimeError("已启用视觉审核，但 vision_api_key 为空。")

        base_url = str(self.config.get("vision_base_url", "")).strip()
        if not base_url:
            raise RuntimeError("已启用视觉审核，但 vision_base_url 为空。")

        model = str(self.config.get("vision_model", "")).strip()
        if not model:
            raise RuntimeError("已启用视觉审核，但 vision_model 为空。")

        image_data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_review_prompt(request)},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "stream": False,
        }

        timeout = float(self.config.get("vision_timeout_seconds", 120))
        proxy_url = str(self.config.get("vision_proxy_url", "")).strip() or str(
            self.config.get("proxy_url", "")
        ).strip() or None
        transport = httpx.AsyncHTTPTransport(proxy=proxy_url) if proxy_url else None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=timeout, transport=transport, trust_env=False) as client:
            response = await client.post(base_url, headers=headers, json=payload)
            if response.status_code >= 400:
                raise RuntimeError(f"视觉审核 API Error {response.status_code}: {response.text[:1000]}")
            data = response.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        review = self._parse_review_json(content)
        return self._review_result_to_tuple(review)

    def _review_result_to_tuple(self, review: dict) -> tuple[bool, str]:
        allow = bool(review.get("allow", False))
        reason = str(review.get("reason", "")).strip()
        if not allow and not reason:
            reason = str(self.config.get("vision_block_reply", "图片未通过审核，已停止发送。"))
        return allow, reason

    def _parse_review_json(self, content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        match = re.search(r"\{.*\}", content, flags=re.S)
        if match:
            content = match.group(0)
        try:
            data = json.loads(content)
        except Exception as exc:
            raise RuntimeError(f"视觉审核返回不是 JSON：{content[:500]}") from exc
        if not isinstance(data, dict) or "allow" not in data:
            raise RuntimeError(f"视觉审核 JSON 缺少 allow 字段：{data}")
        return data

    def _bytes_to_onebot_image(self, image_bytes: bytes) -> str:
        return "base64://" + base64.b64encode(image_bytes).decode("ascii")

    def _clean_api_key(self, api_key: str) -> str:
        return api_key.removeprefix("Bearer ").strip()
