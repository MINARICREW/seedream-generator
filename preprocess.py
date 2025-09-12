from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import importlib.util
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass


def _pil_to_cv(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(mat: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))


def fix_orientation(pil_image: Image.Image) -> Image.Image:
    """Correct EXIF-based orientation; return image unchanged on failure."""
    try:
        return ImageOps.exif_transpose(pil_image)
    except Exception:
        return pil_image


def _import_user_face_cropper():
    """Dynamically import user's FaceCropper from 'cropper_v2.py' (preferred) or legacy name."""
    root = Path(__file__).resolve().parent
    candidates = [
        root / "cropper_v2.py",
        root / "crop_v2 (1).py",
    ]
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("face_cropper_user", str(candidate))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
                return getattr(module, "FaceCropper", None)
    return None


_FACE_CROPPER_INSTANCE = None


def _get_face_cropper():
    global _FACE_CROPPER_INSTANCE
    if _FACE_CROPPER_INSTANCE is not None:
        return _FACE_CROPPER_INSTANCE
    FC = _import_user_face_cropper()
    if FC is None:
        return None
    try:
        _FACE_CROPPER_INSTANCE = FC(margin=0.5, target_size=1024, debug=False, confidence_threshold=0.7, crop_method="balanced_v2")
        return _FACE_CROPPER_INSTANCE
    except Exception:
        return None


def balanced_crop_v2(pil_image: Image.Image, margin_ratio: float = 0.5, target_side: Optional[int] = None) -> Image.Image:
    img = _pil_to_cv(pil_image)
    h, w = img.shape[:2]

    fc = _get_face_cropper()
    if fc is None:
        # Fallback to center crop if user's DNN not available
        side = min(w, h)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        crop = img[y1:y1+side, x1:x1+side]
        if target_side:
            crop = cv2.resize(crop, (target_side, target_side), interpolation=cv2.INTER_AREA)
        return _cv_to_pil(crop)

    # Align margin with parameter
    try:
        fc.margin = margin_ratio  # type: ignore[attr-defined]
    except Exception:
        pass

    detection = None
    try:
        detection = fc.detect_face_dnn(img)
    except Exception:
        detection = None

    if detection and "box" in detection:
        x, y, fw, fh = detection["box"]
        cx = x + fw // 2
        cy = y + fh // 2
        face_size = max(fw, fh)
        x1, y1, x2, y2 = fc.calculate_balanced_crop_v2(cx, cy, face_size, w, h, [x, y, fw, fh])
        side = min(x2 - x1, y2 - y1)
        crop = img[y1:y1+side, x1:x1+side]
    else:
        # Fallback: center crop
        side = min(w, h)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        crop = img[y1:y1+side, x1:x1+side]

    if target_side:
        crop = cv2.resize(crop, (target_side, target_side), interpolation=cv2.INTER_AREA)
    return _cv_to_pil(crop)


def compress_to_target(pil_image: Image.Image, *, max_bytes: int = 9_800_000, prefer_webp: bool = True) -> Tuple[bytes, str]:
    # Try WEBP first (lossy) with quality sweep, then JPEG if not small enough
    if prefer_webp:
        for q in [90, 85, 80, 75, 70, 65, 60]:
            buf = _encode_image(pil_image, format="WEBP", quality=q)
            if len(buf) <= max_bytes:
                return buf, "webp"
    # JPEG fallback
    for q in [90, 85, 80, 75, 70, 65, 60]:
        buf = _encode_image(pil_image, format="JPEG", quality=q)
        if len(buf) <= max_bytes:
            return buf, "jpg"
    # Last resort: downscale by half and try again once
    downsized = pil_image.resize(
        (max(1, pil_image.width // 2), max(1, pil_image.height // 2)), Image.LANCZOS
    )
    return _encode_image(downsized, format="JPEG", quality=75), "jpg"


def _encode_image(pil_image: Image.Image, *, format: str, quality: int = 85) -> bytes:
    import io

    buf = io.BytesIO()
    save_params = {}
    if format.upper() in {"JPEG", "JPG"}:
        save_params = {"quality": quality, "optimize": True}
    elif format.upper() == "WEBP":
        save_params = {"quality": quality, "method": 6}
    pil_image.save(buf, format=format, **save_params)
    return buf.getvalue()


