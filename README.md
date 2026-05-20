# astrbot_plugin_novelai_image

AstrBot NovelAI 画图插件，独立于 `astrbot_plugin_grsai_image`。

## 功能

- 使用 NovelAI 官方 `/ai/generate-image` 接口生成图片。
- 支持 NovelAI 常见 ZIP/PNG 响应解析。
- 兼容 JSON 和 SSE wrapper 响应。
- 支持模型、比例、尺寸、步数、scale、seed、sampler、负面提示词。
- 可选：生成后先交给有视觉能力的大模型审核，通过后才发送。
- 支持 HTTP 代理。

## 命令

```text
/nai 1girl, white hair, red eyes, masterpiece
/novelai --ratio 2:3 1girl, white hair, red eyes
/nai --ratio 3:5 1girl, white hair, red eyes
/nai --size 768x1280 --steps 28 --scale 6 提示词
/nai --seed 123456789 提示词
/nai --model v45 提示词
/nai --uc "lowres, bad anatomy" 提示词
/nai_help
```

## 模型简写

- `v45`: `nai-diffusion-4-5-full`
- `v45-curated`: `nai-diffusion-4-5-curated`
- `v4`: `nai-diffusion-4-full`
- `v4-curated`: `nai-diffusion-4-curated-preview`
- `v3`: `nai-diffusion-3`
- `furry`: `nai-diffusion-furry-3`

## 比例

- `1:1`: 1024x1024
- `2:3`: 832x1216
- `3:2`: 1216x832
- `3:5`: 768x1280
- `5:3`: 1280x768
- `9:16`: 768x1344
- `16:9`: 1344x768
- `4:3`: 1152x896
- `3:4`: 896x1152

## 配置建议

如果 AstrBot 运行在 Docker 中，且宿主机 mihomo 监听 `7890`，插件代理可填：

```text
http://172.17.0.1:7890
```

NovelAI 生成接口建议填：

```text
https://image.novelai.net/ai/generate-image
```

## 视觉审核

开启 `vision_review_enabled` 后，插件会先把生成后的图片交给视觉模型审核。模型必须只返回 JSON：

```json
{"allow": true, "reason": ""}
```

常用配置项：

- `vision_review_mode`: 审核模型来源
  - `astrbot_caption`: 使用 AstrBot 的图片描述模型，推荐
  - `astrbot_current`: 使用当前会话大模型，前提是它支持图片输入
  - `custom_openai`: 使用插件中单独填写的 OpenAI 兼容视觉接口
  - `off`: 不审核
- `vision_base_url`: 仅 `custom_openai` 需要，OpenAI 兼容 `chat/completions` 地址
- `vision_api_key`: 仅 `custom_openai` 需要，视觉模型 API Key
- `vision_model`: 仅 `custom_openai` 需要，支持图片输入的模型名
- `vision_review_fail_closed`: 审核接口失败时是否禁止发送

如果使用 `astrbot_caption`，请先在 AstrBot 中配置 `provider_settings.default_image_caption_provider_id` 对应的视觉模型。
