from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


SANITIZE_RE = re.compile(r"[^A-Za-z0-9._\- ]+")


def sanitize_for_fs(name: str, max_len: int = 80) -> str:
    clean = SANITIZE_RE.sub("_", name).strip().replace(" ", "_")
    return clean[:max_len] if clean else "untitled"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class SaveResult:
    folder: Path
    image_paths: List[Path]
    meta_path: Path


def save_generation(
    *,
    images: List[Image.Image],
    prompt: str,
    output_base: Path,
    model: str,
    sizes: List[str],
    urls: List[str],
    references: Optional[List[Path]] = None,
    reference_uploads: Optional[List[Tuple[Image.Image, str]]] = None,
    suffix: Optional[str] = None,
) -> SaveResult:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prompt_snippet = sanitize_for_fs(prompt)[:60]
    suffix_clean = sanitize_for_fs(suffix) if suffix else None
    folder_name = "_".join(filter(None, [timestamp, prompt_snippet, suffix_clean]))
    folder = output_base / folder_name
    ensure_dir(folder)

    # Copy reference images if any (from filesystem paths)
    if references:
        refs_dir = folder / "references"
        ensure_dir(refs_dir)
        for p in references:
            try:
                shutil.copy2(p, refs_dir / p.name)
            except Exception:
                pass
    # Save uploaded reference images if provided (from memory)
    if reference_uploads:
        refs_dir = folder / "references"
        ensure_dir(refs_dir)
        for img, name in reference_uploads:
            safe_name = sanitize_for_fs(name) or "ref"
            # ensure extension
            if not safe_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                safe_name += '.jpg'
            (refs_dir / safe_name).parent.mkdir(parents=True, exist_ok=True)
            img.save(refs_dir / safe_name, format="JPEG", quality=95)

    image_paths: List[Path] = []
    for idx, img in enumerate(images):
        filename = f"{idx+1:02d}.jpg"
        path = folder / filename
        img.save(path, format="JPEG", quality=95)
        image_paths.append(path)

    meta = {
        "prompt": prompt,
        "model": model,
        "sizes": sizes,
        "urls": urls,
        "num_images": len(images),
        "created": timestamp,
    }
    meta_path = folder / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return SaveResult(folder=folder, image_paths=image_paths, meta_path=meta_path)


def save_images_with_names(
    *,
    images: List[Image.Image],
    names: List[str],
    prompt: str,
    output_base: Path,
    model: str,
    sizes: List[str],
    urls: List[str],
    suffix: Optional[str] = None,
) -> SaveResult:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prompt_snippet = sanitize_for_fs(prompt)[:60]
    suffix_clean = sanitize_for_fs(suffix) if suffix else "bulk"
    folder_name = "_".join(filter(None, [timestamp, prompt_snippet, suffix_clean]))
    folder = output_base / folder_name
    ensure_dir(folder)

    image_paths: List[Path] = []
    for img, name in zip(images, names):
        base = sanitize_for_fs(Path(name).stem) or "image"
        path = folder / f"{base}.jpg"
        img.save(path, format="JPEG", quality=95)
        image_paths.append(path)

    meta = {
        "prompt": prompt,
        "model": model,
        "num_images": len(images),
        "created": timestamp,
        "items": [
            {"name": Path(n).name, "size": s if i < len(sizes) else "", "url": urls[i] if i < len(urls) else ""}
            for i, (n, s) in enumerate(zip(names, sizes))
        ],
    }
    meta_path = folder / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return SaveResult(folder=folder, image_paths=image_paths, meta_path=meta_path)


