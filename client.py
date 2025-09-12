from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import requests
from PIL import Image
import base64 as _b64
import time
import json


API_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"


def _load_api_key() -> str:
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ARK_API_KEY is not set. Create env.local and set ARK_API_KEY."
        )
    return api_key


def _image_to_data_url_bytes(image: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    b64 = _b64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


@dataclass
class GenerationResult:
    images: List[Image.Image]
    urls: List[str]
    sizes: List[str]
    raw_response: Dict


def _normalize_images_payload(
    image_inputs: Optional[List[str]] = None,
    pil_images: Optional[List[Image.Image]] = None,
) -> Optional[List[str]]:
    if not image_inputs and not pil_images:
        return None
    urls_or_b64: List[str] = []
    if image_inputs:
        urls_or_b64.extend(image_inputs)
    if pil_images:
        for im in pil_images:
            urls_or_b64.append(_image_to_data_url_bytes(im))
    return urls_or_b64


def _headers(accept_sse: bool = False) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_load_api_key()}",
    }
    if accept_sse:
        headers["Accept"] = "text/event-stream"
    return headers


def _download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def generate_images(
    prompt: str,
    *,
    model: str = "seedream-4-0-250828",
    size: Optional[str] = "1296×1728",
    resolution_tag: Optional[str] = None,
    images_url_or_b64: Optional[List[str]] = None,
    pil_images: Optional[List[Image.Image]] = None,
    sequential_mode: str = "disabled",
    max_images: Optional[int] = None,
    stream: bool = False,
    seed: Optional[int] = None,
    response_format: str = "url",
    logger: Optional[Callable[[str], None]] = None,
    on_partial: Optional[Callable[[int, Image.Image, str, Optional[str]], None]] = None,
) -> GenerationResult:
    def _log(msg: str) -> None:
        if logger is not None:
            try:
                logger(msg)
            except Exception:
                pass
    payload: Dict = {
        "model": model,
        "prompt": prompt,
        "sequential_image_generation": sequential_mode,
        "response_format": response_format,
        "stream": stream,
        # Force watermark off per user preference
        "watermark": False,
    }

    if size and resolution_tag:
        raise ValueError("Use either size or resolution_tag (1K/2K/4K), not both.")

    if resolution_tag:
        payload["size"] = resolution_tag
    elif size:
        payload["size"] = size

    images_payload = _normalize_images_payload(images_url_or_b64, pil_images)
    if images_payload:
        # API supports string or array; we use array for multi-ref
        payload["image"] = images_payload if len(images_payload) > 1 else images_payload[0]
        # brief sizes log
        count = len(images_payload) if isinstance(images_payload, list) else 1
        _log(f"payload images prepared | count={count}")

    if sequential_mode == "auto" and max_images is not None:
        payload["sequential_image_generation_options"] = {"max_images": int(max_images)}

    # Only some models support seed (seedream-3.0-t2i and seededit-3.0-i2i)
    model_l = model.lower()
    if seed is not None and ("seedream-3.0" in model_l or "seededit-3.0" in model_l):
        payload["seed"] = int(seed)

    _log(f"POST {API_URL} | model={model} size={payload.get('size')} seq={sequential_mode} stream={stream}")
    # simple manual retries for transient errors
    last_exc: Optional[Exception] = None
    urls: List[str] = []
    images: List[Image.Image] = []
    sizes: List[str] = []

    if stream:
        # SSE streaming mode (Seedream 4.0 only)
        for attempt in range(3):
            try:
                t0 = time.time()
                resp = requests.post(
                    API_URL,
                    headers=_headers(accept_sse=True),
                    json=payload,
                    timeout=300,
                    stream=True,
                )
                dt = (time.time() - t0) * 1000
                _log(f"attempt {attempt+1}/3 (SSE) -> status={resp.status_code} t={dt:.0f}ms")
                if resp.status_code >= 500:
                    raise RuntimeError(f"Server error {resp.status_code}: {resp.text}")
                if resp.status_code >= 400:
                    raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

                # Parse SSE lines
                event_buf: List[str] = []
                raw_events: List[Dict] = []
                for line in resp.iter_lines(decode_unicode=True):
                    if line is None:
                        continue
                    line_str = line.strip()
                    if not line_str:
                        # dispatch accumulated event
                        if event_buf:
                            try:
                                event_json_str = "\n".join(event_buf)
                                # Typical SSE uses 'data: {...}' per line
                                # Strip optional leading 'data: '
                                cleaned_lines = []
                                for l in event_buf:
                                    cleaned_lines.append(l[5:].strip() if l.startswith("data:") else l)
                                event_json_str = "\n".join(cleaned_lines)
                                evt = json.loads(event_json_str)
                            except Exception:
                                evt = {}
                            raw_events.append(evt)  # keep for raw_response
                            etype = evt.get("type")
                            if etype == "image_generation.partial_succeeded":
                                size_val = evt.get("size", "")
                                if response_format == "url":
                                    url = evt.get("url", "")
                                    if url:
                                        urls.append(url)
                                        try:
                                            img = _download_image(url)
                                        except Exception:
                                            img = Image.new("RGB", (1024, 1024), color=(255, 255, 255))
                                        images.append(img)
                                        sizes.append(size_val)
                                        if on_partial is not None:
                                            try:
                                                on_partial(len(images) - 1, img, size_val, url)
                                            except Exception:
                                                pass
                                else:
                                    b64s = evt.get("b64_json", "")
                                    if b64s:
                                        try:
                                            raw = base64.b64decode(b64s)
                                            img = Image.open(io.BytesIO(raw)).convert("RGB")
                                        except Exception:
                                            img = Image.new("RGB", (1024, 1024), color=(255, 255, 255))
                                        images.append(img)
                                        urls.append("")
                                        sizes.append(size_val)
                                        if on_partial is not None:
                                            try:
                                                on_partial(len(images) - 1, img, size_val, None)
                                            except Exception:
                                                pass
                            elif etype == "image_generation.partial_failed":
                                err = evt.get("error", {})
                                _log(f"partial_failed | idx={evt.get('image_index')} code={err.get('code')} msg={err.get('message')}")
                                # Continue to next event
                            elif etype == "image_generation.completed":
                                _log("completed event received")
                                # End of stream; response body may continue but we can break after this event
                                # Do not break immediately; consume remainder quickly to close connection
                        event_buf = []
                    else:
                        # Accumulate only lines pertinent to data; capture JSON-only lines too
                        if line_str.startswith("data:") or line_str.startswith("{"):
                            event_buf.append(line_str)
                # Build a raw_response compatible object
                raw_response = {"type": "stream", "events": raw_events}
                _log(f"response parsed (SSE) | images={len(images)} sizes={sizes}")
                return GenerationResult(images=images, urls=urls, sizes=sizes, raw_response=raw_response)
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    _log(f"request failed (SSE) | {e}")
                    raise e
        # Streaming failed after retries. Fallback to normal blocking request.
        # This preserves UX when SSE is not supported by endpoint.
        _log(f"streaming failed, falling back to non-streaming | err={last_exc}")
        stream = False
        # fall through to non-streaming path below
    else:
        for attempt in range(3):
            try:
                t0 = time.time()
                resp = requests.post(API_URL, headers=_headers(), json=payload, timeout=120)
                dt = (time.time() - t0) * 1000
                _log(f"attempt {attempt+1}/3 -> status={resp.status_code} t={dt:.0f}ms")
                if resp.status_code >= 500:
                    raise RuntimeError(f"Server error {resp.status_code}: {resp.text}")
                if resp.status_code >= 400:
                    # client errors are not retried
                    raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
                break
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    _log(f"request failed | {e}")
                    raise e
        data = resp.json()

        for item in data.get("data", []):
            if "url" in item:
                url = item["url"]
                urls.append(url)
                try:
                    images.append(_download_image(url))
                except Exception:
                    # If download fails, leave a placeholder blank image
                    placeholder = Image.new("RGB", (1024, 1024), color=(255, 255, 255))
                    images.append(placeholder)
            elif "b64_json" in item:
                raw = base64.b64decode(item["b64_json"])  # type: ignore[arg-type]
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                urls.append("")
            sizes.append(item.get("size", ""))

        _log(f"response parsed | images={len(images)} sizes={sizes}")
        return GenerationResult(images=images, urls=urls, sizes=sizes, raw_response=data)


