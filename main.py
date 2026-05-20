import asyncio
import base64
import json
import random
import re
import shlex
import struct
import time
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    original_prompt: str
    negative_prompt: str
    style_name: str
    use_quality_prompt: bool
    use_style_prompt: bool
    model: str
    width: int
    height: int
    steps: int
    scale: float
    seed: int
    sampler: str
    llm_optimize: bool


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
        self._output_dir = Path("/AstrBot/data/plugin_data/novelai_image/outputs")
        self._quota_file = Path("/AstrBot/data/plugin_data/novelai_image/quota_usage.json")
        self._quota_usage = self._load_quota_usage()
        self._generation_lock = asyncio.Lock()
        self._queue_waiting = 0

    @filter.command("nai", alias={"novelai", "nai画图", "nai生图"})
    async def generate(self, event: AstrMessageEvent):
        api_key = self._clean_api_key(str(self.config.get("api_key", "")).strip())
        if not api_key:
            yield event.plain_result("NovelAI API Key 还没配置，请先在插件配置里填写 api_key。")
            return

        sender_id = str(event.get_sender_id())
        policy_error = self._check_usage_policy(sender_id)
        if policy_error:
            yield event.plain_result(policy_error)
            return

        try:
            request = self._parse_request(event.message_str)
        except ValueError as exc:
            yield event.plain_result(str(exc))
            return

        if request.llm_optimize:
            try:
                optimized_prompt = await self._optimize_prompt_with_llm(request.prompt, event)
                if optimized_prompt:
                    logger.info(
                        "NovelAI prompt optimized by LLM. original=%s optimized=%s",
                        request.prompt,
                        optimized_prompt,
                    )
                    request.prompt = optimized_prompt
            except Exception as exc:
                logger.error(f"NovelAI prompt optimization failed: {exc}")
                yield event.plain_result(f"提示词优化失败：{exc}")
                return

        try:
            request.prompt = self._apply_prompt_presets(request)
        except ValueError as exc:
            yield event.plain_result(str(exc))
            return
        logger.info(
            "NovelAI final prompt prepared. style=%s quality=%s prompt=%s",
            request.style_name,
            request.use_quality_prompt,
            request.prompt,
        )

        queue_position = self._queue_waiting + (1 if self._generation_lock.locked() else 0)
        self._queue_waiting += 1
        if queue_position > 0:
            yield event.plain_result(f"已加入 NovelAI 画图队列，前面还有 {queue_position} 个任务。")

        async with self._generation_lock:
            self._queue_waiting = max(0, self._queue_waiting - 1)
            if queue_position > 0:
                yield event.plain_result("轮到你了，开始生成 NovelAI 图片。")
            else:
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
                allowed, reason = await self._review_image(image_bytes, request, event)
            except Exception as exc:
                logger.error(f"NovelAI image review failed: {exc}")
                if bool(self.config.get("vision_review_fail_closed", False)):
                    yield event.plain_result(f"图片审核失败，已停止发送：{exc}")
                    return
                allowed, reason = True, ""

            if not allowed:
                if reason:
                    logger.warning(f"NovelAI image blocked by vision review: {reason}")
                block_reply = str(self.config.get("vision_block_reply", "您生成的内容被拦截。"))
                yield self._mention_sender_result(event, block_reply)
                return

            image_path = self._save_image(image_bytes, request)
            self._record_quota_usage(sender_id)
            yield event.image_result(str(image_path))

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
                    "/nai --style cinematic 提示词",
                    "/nai --uc lowres, bad anatomy 提示词",
                    "/nai -llm 中文提示词",
                    "/nai --no-style 提示词",
                    "/nai --no-quality 提示词",
                    "",
                    "支持比例：1:1、2:3、3:2、3:5、5:3、9:16、16:9、4:3、3:4",
                    "模型简写：v45、v45-curated、v4、v4-curated、v3、furry",
                    f"当前默认画风：{self._default_style_name()}",
                    f"可用画风：{', '.join(self._style_presets().keys()) or '无'}",
                    "",
                    "例子：",
                    "/nai --ratio 2:3 --steps 28 --scale 5.5 1girl, white hair, red eyes, best quality",
                    "/nai --style watercolor 白发红瞳少女，海边",
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
        style_name = self._default_style_name()
        use_quality_prompt = True
        use_style_prompt = True
        llm_optimize = False

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
            elif token in {"--style", "-s", "--画风"}:
                index += 1
                style_name = self._need_value(parts, index, token)
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
            elif token in {"-llm", "--llm", "--optimize", "--优化"}:
                llm_optimize = True
            elif token in {"--no-quality", "--noquality", "--不要质量词"}:
                use_quality_prompt = False
            elif token in {"--no-style", "--nostyle", "--不要画风"}:
                use_style_prompt = False
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
            original_prompt=prompt,
            negative_prompt=negative_prompt,
            style_name=style_name,
            use_quality_prompt=use_quality_prompt,
            use_style_prompt=use_style_prompt,
            model=self._normalize_model(model),
            width=width,
            height=height,
            steps=steps,
            scale=scale,
            seed=seed,
            sampler=sampler,
            llm_optimize=llm_optimize,
        )

    def _check_usage_policy(self, sender_id: str) -> str | None:
        if sender_id in self._string_list("blacklist_user_ids"):
            return str(self.config.get("blacklist_reply", "你已被加入 NovelAI 画图黑名单，无法使用该功能。"))

        if self._is_quota_exempt(sender_id):
            return None

        disabled_reason = self._disabled_time_reason()
        if disabled_reason:
            return disabled_reason

        daily_limit = int(self.config.get("daily_quota_limit", 10))
        if daily_limit > 0 and self._quota_used_today(sender_id) >= daily_limit:
            return str(
                self.config.get(
                    "quota_exceeded_reply",
                    f"你今天的 NovelAI 画图额度已用完（{daily_limit} 张/天），明天再来吧。",
                )
            )
        return None

    def _is_quota_exempt(self, sender_id: str) -> bool:
        return sender_id in self._string_list("allowed_user_ids")

    def _disabled_time_reason(self) -> str | None:
        if not bool(self.config.get("time_limit_enabled", True)):
            return None
        start = str(self.config.get("disabled_start_time", "23:00")).strip()
        end = str(self.config.get("disabled_end_time", "08:00")).strip()
        start_time = self._parse_hhmm(start)
        end_time = self._parse_hhmm(end)
        if not start_time or not end_time:
            return None

        now = datetime.now(self._local_timezone()).time()
        if start_time <= end_time:
            disabled = start_time <= now < end_time
        else:
            disabled = now >= start_time or now < end_time
        if not disabled:
            return None
        return str(self.config.get("time_limit_reply", f"NovelAI 画图功能在 {start}-{end} 暂停使用，请稍后再试。"))

    def _parse_hhmm(self, value: str):
        try:
            return datetime.strptime(value.strip(), "%H:%M").time()
        except ValueError:
            logger.warning(f"Invalid NovelAI time config: {value}")
            return None

    def _local_timezone(self) -> ZoneInfo:
        timezone_name = str(self.config.get("timezone", "Asia/Shanghai")).strip() or "Asia/Shanghai"
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            logger.warning(f"Invalid NovelAI timezone config: {timezone_name}, fallback to Asia/Shanghai")
            return ZoneInfo("Asia/Shanghai")

    def _quota_key(self) -> str:
        return datetime.now(self._local_timezone()).strftime("%Y-%m-%d")

    def _quota_used_today(self, sender_id: str) -> int:
        return int(self._quota_usage.get(self._quota_key(), {}).get(sender_id, 0))

    def _record_quota_usage(self, sender_id: str):
        if self._is_quota_exempt(sender_id):
            return
        day = self._quota_key()
        self._quota_usage.setdefault(day, {})
        self._quota_usage[day][sender_id] = int(self._quota_usage[day].get(sender_id, 0)) + 1
        for key in list(self._quota_usage.keys()):
            if key != day:
                self._quota_usage.pop(key, None)
        self._save_quota_usage()

    def _load_quota_usage(self) -> dict:
        try:
            if self._quota_file.exists():
                with open(self._quota_file, "r", encoding="utf-8") as file:
                    data = json.load(file)
                return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning(f"Failed to load NovelAI quota usage: {exc}")
        return {}

    def _save_quota_usage(self):
        try:
            self._quota_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._quota_file, "w", encoding="utf-8") as file:
                json.dump(self._quota_usage, file, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to save NovelAI quota usage: {exc}")

    def _string_list(self, key: str) -> list[str]:
        value = self.config.get(key, [])
        if isinstance(value, str):
            value = value.replace("\n", ",").split(",")
        return [str(item).strip() for item in value if str(item).strip()]

    def _mention_sender_result(self, event: AstrMessageEvent, text: str):
        if not self._is_group_event(event):
            return event.plain_result(text)
        try:
            import astrbot.api.message_components as Comp

            return event.chain_result([Comp.At(qq=event.get_sender_id()), Comp.Plain(f" {text}")])
        except Exception as exc:
            logger.warning(f"Failed to build at-message result, fallback to plain text: {exc}")
            return event.plain_result(f"@{event.get_sender_id()} {text}")

    def _is_group_event(self, event: AstrMessageEvent) -> bool:
        message_obj = getattr(event, "message_obj", None)
        return bool(getattr(message_obj, "group_id", ""))

    def _apply_prompt_presets(self, request: NovelAIRequest) -> str:
        parts = [request.prompt.strip()]
        if request.use_quality_prompt:
            quality_prompt = str(self.config.get("quality_prompt", "")).strip()
            if quality_prompt:
                parts.append(quality_prompt)

        style_name = request.style_name.strip()
        if request.use_style_prompt and style_name and style_name.lower() not in {"none", "off", "无"}:
            styles = self._style_presets()
            style_prompt = styles.get(style_name)
            if style_prompt is None:
                available = ", ".join(styles.keys()) or "无"
                raise ValueError(f"未知画风：{style_name}。可用画风：{available}")
            if style_prompt.strip():
                parts.append(style_prompt.strip())
        return self._join_prompt_parts(parts)

    def _join_prompt_parts(self, parts: list[str]) -> str:
        cleaned = [part.strip().strip(",") for part in parts if str(part).strip().strip(",")]
        return ", ".join(cleaned)

    def _default_style_name(self) -> str:
        return str(self.config.get("default_style", "none")).strip() or "none"

    def _style_presets(self) -> dict[str, str]:
        raw = self.config.get("style_presets", "")
        if isinstance(raw, dict):
            return {str(key).strip(): str(value).strip() for key, value in raw.items() if str(key).strip()}
        if isinstance(raw, list):
            presets = {}
            for item in raw:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                    prompt = str(item.get("prompt", "")).strip()
                    if name:
                        presets[name] = prompt
            return presets
        text = str(raw).strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except Exception as exc:
            logger.warning(f"Failed to parse style_presets JSON: {exc}")
            return {}
        if not isinstance(data, dict):
            logger.warning("style_presets must be a JSON object.")
            return {}
        return {str(key).strip(): str(value).strip() for key, value in data.items() if str(key).strip()}

    async def _optimize_prompt_with_llm(self, prompt: str, event: AstrMessageEvent) -> str:
        provider_id = await self.context.get_current_chat_provider_id(self._event_umo(event))
        if not provider_id:
            raise RuntimeError("当前会话没有可用的大模型 Provider。")

        system_prompt = str(self.config.get("llm_prompt_optimizer_prompt", "")).strip()
        if not system_prompt:
            system_prompt = (
                "你是 NovelAI 提示词优化器。把用户输入改写为适合 NovelAI/anime diffusion 的英文提示词。"
                "要求：只输出最终英文 prompt，不要 Markdown，不要解释。"
                "保留用户的主体、动作、服装、场景、构图和风格要求。"
                "如果用户输入中文，请准确翻译并补充常用英文 tag。"
                "避免加入用户没有要求的露骨色情、未成年、血腥、政治或违法内容。"
                "输出以逗号分隔的 tag/prompt。"
            )

        user_msg = UserMessageSegment(
            content=[
                TextPart(
                    text=(
                        f"{system_prompt}\n\n"
                        "用户原始提示词：\n"
                        f"{prompt}"
                    )
                )
            ]
        )
        llm_resp = await self.context.llm_generate(
            chat_provider_id=str(provider_id),
            contexts=[user_msg],
        )
        text = (getattr(llm_resp, "completion_text", "") or "").strip()
        return self._clean_llm_prompt(text)

    def _clean_llm_prompt(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:text|txt|prompt)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"^(prompt|optimized prompt|final prompt)\s*[:：]\s*", "", text, flags=re.I).strip()
        return text.strip().strip("\"'")

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

    async def _review_image(
        self, image_bytes: bytes, request: NovelAIRequest, event: AstrMessageEvent
    ) -> tuple[bool, str]:
        if not bool(self.config.get("vision_review_enabled", False)):
            return True, ""

        mode = str(self.config.get("vision_review_mode", "astrbot_caption")).strip() or "astrbot_caption"
        if mode == "off":
            return True, ""

        if mode in {"astrbot_caption", "astrbot_current"}:
            return await self._review_image_with_astrbot_provider(image_bytes, request, event, mode)

        if mode != "custom_openai":
            raise RuntimeError(f"未知视觉审核模式：{mode}")

        return await self._review_image_with_custom_openai(image_bytes, request)

    async def _review_image_with_astrbot_provider(
        self, image_bytes: bytes, request: NovelAIRequest, event: AstrMessageEvent, mode: str
    ) -> tuple[bool, str]:
        image_data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        provider_id = await self._get_astrbot_review_provider_id(mode, event)
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

    async def _get_astrbot_review_provider_id(self, mode: str, event: AstrMessageEvent) -> str:
        if mode == "astrbot_current":
            provider_id = await self.context.get_current_chat_provider_id(self._event_umo(event))
            return str(provider_id or "").strip()

        config = self._load_astrbot_config()
        provider_settings = config.get("provider_settings", {}) if isinstance(config, dict) else {}
        provider_id = provider_settings.get("default_image_caption_provider_id", "")
        if provider_id:
            return str(provider_id).strip()

        provider_id = await self.context.get_current_chat_provider_id(self._event_umo(event))
        return str(provider_id or "").strip()

    def _event_umo(self, event: AstrMessageEvent) -> str:
        return str(getattr(event.message_obj, "unified_msg_origin", "") or "")

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
            reason = "视觉审核未提供具体原因。"
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

    def _save_image(self, image_bytes: bytes, request: NovelAIRequest) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        image_bytes = self._replace_png_metadata(image_bytes)
        filename = f"nai_{int(time.time())}_{request.seed}.png"
        path = self._output_dir / filename
        path.write_bytes(image_bytes)
        return path

    def _replace_png_metadata(self, image_bytes: bytes) -> bytes:
        if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return image_bytes

        signature = image_bytes[:8]
        offset = 8
        chunks: list[tuple[bytes, bytes]] = []
        inserted = False
        text_chunk = self._png_chunk(
            b"iTXt",
            b"Description\x00\x00\x00\x00\x00" + "generated by 白毛红瞳魔法师".encode("utf-8"),
        )

        while offset + 8 <= len(image_bytes):
            length = struct.unpack(">I", image_bytes[offset : offset + 4])[0]
            chunk_type = image_bytes[offset + 4 : offset + 8]
            data_start = offset + 8
            data_end = data_start + length
            crc_end = data_end + 4
            if crc_end > len(image_bytes):
                return image_bytes

            data = image_bytes[data_start:data_end]
            offset = crc_end

            if chunk_type in {b"tEXt", b"zTXt", b"iTXt"}:
                continue

            if chunk_type == b"IEND" and not inserted:
                chunks.append((b"__RAW__", text_chunk))
                inserted = True

            chunks.append((chunk_type, data))
            if chunk_type == b"IEND":
                break

        output = bytearray(signature)
        for chunk_type, data in chunks:
            if chunk_type == b"__RAW__":
                output.extend(data)
            else:
                output.extend(self._png_chunk(chunk_type, data))
        return bytes(output)

    def _png_chunk(self, chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type)
        crc = zlib.crc32(data, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    def _clean_api_key(self, api_key: str) -> str:
        return api_key.removeprefix("Bearer ").strip()
