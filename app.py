from __future__ import annotations

import os
from pathlib import Path
import io
import time
import re
import math
import html
import base64
import json
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from client import GenerationResult, generate_images
from preprocess import balanced_crop_v2, compress_to_target, fix_orientation
from utils import SaveResult, ensure_dir, save_generation, sanitize_for_fs


load_dotenv("env.local", override=False)
load_dotenv(override=False)  # allow standard .env too


st.set_page_config(page_title="Seedream 4.0 Studio", layout="wide")


ASPECT_RATIO_OPTIONS = [
    "1:1", "1:2", "2:1", "2:3", "3:2", "4:3", "3:4",
    "2:5", "5:2", "3:5", "5:3", "4:5", "5:4",
    "5:6", "6:5", "6:7", "7:6", "7:9", "9:7",
]


def _nearest_multiple(value: int, base: int = 32) -> int:
    return int(round(value / base) * base)


def _format_ratio_option(ratio: str) -> str:
    labels = {
        "7:9": "7:9 – 주민등록증/여권사진",
        "5:6": "5:6 – 증명사진",
        "5:7": "5:7 – 명함판",
        "3:4": "3:4 – 반명함",
    }
    return labels.get(ratio, ratio)


# --- Preset helpers (keep UI state consistent and avoid NameError) ---
def _update_t2i_from_preset() -> None:
    try:
        ratio = st.session_state.get("t2i_ratio", "7:9")
        k_tag = st.session_state.get("t2i_k", "2K")
        w, h = _compute_preset_size(ratio, k_tag)
        st.session_state["t2i_w"] = w
        st.session_state["t2i_h"] = h
        st.session_state["t2i_size_mode"] = "Preset"
    except Exception:
        pass


def _set_t2i_explicit() -> None:
    try:
        st.session_state["t2i_size_mode"] = "Explicit WxH"
    except Exception:
        pass


def _update_i2i_from_preset() -> None:
    try:
        ratio = st.session_state.get("i2i_ratio", "7:9")
        k_tag = st.session_state.get("i2i_k", "2K")
        w, h = _compute_preset_size(ratio, k_tag)
        st.session_state["i2i_w"] = w
        st.session_state["i2i_h"] = h
        st.session_state["i2i_mode"] = "Preset"
    except Exception:
        pass


def _set_i2i_explicit() -> None:
    try:
        st.session_state["i2i_mode"] = "Explicit WxH"
    except Exception:
        pass


def _compute_preset_size(ratio: str, k_tag: str) -> tuple[int, int]:
    try:
        w_r_s, h_r_s = ratio.split(":")
        w_r = int(w_r_s)
        h_r = int(h_r_s)
    except Exception:
        w_r, h_r = 7, 9
    long_side_map = {"1K": 1024, "2K": 2048, "4K": 4096}
    target_long = long_side_map.get(k_tag, 2048)
    if w_r >= h_r:
        width = target_long
        height = int(round(target_long * h_r / w_r))
    else:
        height = target_long
        width = int(round(target_long * w_r / h_r))
    width = max(32, _nearest_multiple(width, 32))
    height = max(32, _nearest_multiple(height, 32))
    # clamp to API/UI bounds
    width = min(width, 4096)
    height = min(height, 4096)
    return width, height


RATIO_LABELS = {
    "3:4": "반명함",
    "5:7": "명함판",
    "7:9": "주민등록증/여권사진",
    "5:6": "증명사진",
}


def _format_ratio_option(r: str) -> str:
    label = RATIO_LABELS.get(r)
    return f"{r} – {label}" if label else r


def _update_t2i_from_preset() -> None:
    ratio = st.session_state.get("t2i_ratio", "7:9")
    kval = st.session_state.get("t2i_k", "2K")
    w, h = _compute_preset_size(ratio, kval)
    st.session_state["t2i_w"] = w
    st.session_state["t2i_h"] = h
    st.session_state["t2i_w_in"] = w
    st.session_state["t2i_h_in"] = h
    st.session_state["t2i_size_mode"] = "Preset"


def _set_t2i_explicit() -> None:
    st.session_state["t2i_size_mode"] = "Explicit WxH"


def _update_i2i_from_preset() -> None:
    ratio = st.session_state.get("i2i_ratio", "7:9")
    kval = st.session_state.get("i2i_k", "2K")
    w, h = _compute_preset_size(ratio, kval)
    st.session_state["i2i_w"] = w
    st.session_state["i2i_h"] = h
    st.session_state["i2i_size_mode"] = "Preset"


def _set_i2i_explicit() -> None:
    st.session_state["i2i_size_mode"] = "Explicit WxH"


def _update_grid_from_preset() -> None:
    ratio = st.session_state.get("grid_ratio", "7:9")
    kval = st.session_state.get("grid_k", "2K")
    w, h = _compute_preset_size(ratio, kval)
    st.session_state["grid_w"] = w
    st.session_state["grid_h"] = h
    st.session_state["grid_mode"] = "Preset"


def _set_grid_explicit() -> None:
    st.session_state["grid_mode"] = "Explicit WxH"


def _update_bulk_from_preset() -> None:
    ratio = st.session_state.get("bulk_ratio", "7:9")
    kval = st.session_state.get("bulk_k", "2K")
    w, h = _compute_preset_size(ratio, kval)
    st.session_state["bulk_w"] = w
    st.session_state["bulk_h"] = h
    st.session_state["bulk_mode"] = "Preset"


def _set_bulk_explicit() -> None:
    st.session_state["bulk_mode"] = "Explicit WxH"


def get_default_output_dir() -> Path:
    base = os.getenv("OUTPUT_BASE_DIR", "outputs")
    return Path(base)


def sidebar() -> Path:
    # Top banner-like Settings header with icon aligned to top toolbar
    st.sidebar.markdown(
        """
        <style>
          .sd-settings-banner{display:flex;align-items:center;margin-top:-22px;padding:0 0 2px 0;}
          .sd-settings-banner .icon{font-size:22px;}
          .sd-settings-banner .label{font-weight:700;font-size:22px;margin-left:8px;}
        </style>
        <div class="sd-settings-banner">
          <span class="icon">⚙️</span>
          <span class="label">Settings</span>
        </div>
        <hr style="margin:4px 0 10px 0;opacity:0.3;">
        """,
        unsafe_allow_html=True,
    )
    base_dir = st.sidebar.text_input(
        "Output base folder",
        value=str(get_default_output_dir()),
        help="Where generated images and metadata will be saved.",
    )
    output_dir = Path(base_dir)
    ensure_dir(output_dir)

    model = st.sidebar.text_input(
        "Model ID or Endpoint ID",
        value=os.getenv("SEEDREAM_MODEL_ID", "seedream-4-0-250828"),
        help="Use 'seedream-4-0-250828' (default) or paste an endpoint ID if you created one.",
    )
    st.session_state["model"] = model

    # Performance (moved up)
    st.sidebar.subheader("Performance")
    default_parallel = int(os.getenv("MAX_PARALLEL", "8"))
    max_parallel = st.sidebar.slider(
        "Max parallel requests (Grid/Bulk)", 1, 16, value=default_parallel
    )
    st.session_state["max_parallel"] = max_parallel
    # Streaming is always enabled
    st.session_state["stream"] = True
    # Progress panel (now after Performance)
    st.sidebar.subheader("Progress")
    # Always recreate progress widgets each run to keep UI stable across reruns
    try:
        current_ratio = 0.0
        if "p_total" in st.session_state:
            total = max(1, int(st.session_state.get("p_total", 1)))
            current_ratio = float(st.session_state.get("p_current", 0)) / float(total)
        st.session_state["progress_bar"] = st.sidebar.progress(current_ratio)
    except Exception:
        st.session_state["progress_bar"] = None
    st.session_state["progress_text"] = st.sidebar.empty()
    if "p_total" in st.session_state:
        try:
            st.session_state["progress_text"].write(
                f"{st.session_state.get('p_name','Progress')}: {st.session_state.get('p_current',0)}/{st.session_state.get('p_total',0)}"
            )
        except Exception:
            pass
    # Live Logs (now last)
    st.sidebar.subheader("Live Logs")
    # Recreate placeholder every run to avoid stale widget references
    st.session_state["log_placeholder"] = st.sidebar.empty()
    if "log_messages" not in st.session_state:
        st.session_state["log_messages"] = []
    _render_logs()
    return output_dir


def log(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    st.session_state["log_messages"].append(f"[{ts}] {message}")
    _render_logs()


def _render_logs() -> None:
    if "log_placeholder" in st.session_state:
        content = "\n".join(st.session_state.get("log_messages", [])[-200:])
        st.session_state["log_placeholder"].code(content or "(no logs yet)")


def progress_start(name: str, total: int) -> None:
    st.session_state["p_name"] = name
    st.session_state["p_total"] = max(1, int(total))
    st.session_state["p_current"] = 0
    try:
        st.session_state["progress_bar"] = st.sidebar.progress(0)
    except Exception:
        st.session_state["progress_bar"] = None
    try:
        st.session_state["progress_text"].write(f"{name}: 0/{st.session_state['p_total']}")
    except Exception:
        pass


def progress_step(increment: int = 1) -> None:
    if "p_total" not in st.session_state:
        return
    st.session_state["p_current"] = min(
        st.session_state["p_total"], st.session_state.get("p_current", 0) + increment
    )
    ratio = st.session_state["p_current"] / st.session_state["p_total"]
    try:
        if st.session_state.get("progress_bar") is not None:
            st.session_state["progress_bar"].progress(ratio)
    except Exception:
        pass
    try:
        st.session_state["progress_text"].write(
            f"{st.session_state.get('p_name','Progress')}: {st.session_state['p_current']}/{st.session_state['p_total']}"
        )
    except Exception:
        pass


def progress_done() -> None:
    if "p_total" in st.session_state:
        st.session_state["p_current"] = st.session_state["p_total"]
        try:
            if st.session_state.get("progress_bar") is not None:
                st.session_state["progress_bar"].progress(1.0)
        except Exception:
            pass
        try:
            st.session_state["progress_text"].write(
                f"{st.session_state.get('p_name','Progress')}: done ({st.session_state['p_total']}/{st.session_state['p_total']})"
            )
        except Exception:
            pass


def render_form(output_dir: Path) -> None:
    st.title("Seedream 4.0 – Text-to-Image / Image-to-Image")

    # Global compact styles
    st.markdown(
        """
        <style>
        .prompt-label { font-size: 0.9rem; line-height: 1.15; margin: 0 0 4px 0; }
        div[data-testid="column"] { padding-left: 0.15rem; padding-right: 0.15rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Text to Image", "Image to Image (Refs)", "Grid Generator", "Bulk Runner"])

    # Initialize interactive defaults for Preset-driven sizing
    if "t2i_ratio" not in st.session_state:
        st.session_state["t2i_ratio"] = "7:9"
    if "t2i_k" not in st.session_state:
        st.session_state["t2i_k"] = "2K"
    if "t2i_w" not in st.session_state or "t2i_h" not in st.session_state:
        _w0, _h0 = _compute_preset_size(st.session_state["t2i_ratio"], st.session_state["t2i_k"])
        st.session_state["t2i_w"] = _w0
        st.session_state["t2i_h"] = _h0
    if "t2i_size_mode" not in st.session_state:
        st.session_state["t2i_size_mode"] = "Preset"

    if "i2i_ratio" not in st.session_state:
        st.session_state["i2i_ratio"] = "7:9"
    if "i2i_k" not in st.session_state:
        st.session_state["i2i_k"] = "2K"
    if "i2i_w" not in st.session_state or "i2i_h" not in st.session_state:
        _w1, _h1 = _compute_preset_size(st.session_state["i2i_ratio"], st.session_state["i2i_k"])
        st.session_state["i2i_w"] = _w1
        st.session_state["i2i_h"] = _h1
    if "i2i_mode" not in st.session_state:
        st.session_state["i2i_mode"] = "Preset"

    if "grid_ratio" not in st.session_state:
        st.session_state["grid_ratio"] = "7:9"
    if "grid_k" not in st.session_state:
        st.session_state["grid_k"] = "2K"
    if "grid_w" not in st.session_state or "grid_h" not in st.session_state:
        _w2, _h2 = _compute_preset_size(st.session_state["grid_ratio"], st.session_state["grid_k"])
        st.session_state["grid_w"] = _w2
        st.session_state["grid_h"] = _h2
    if "grid_mode" not in st.session_state:
        st.session_state["grid_mode"] = "Explicit WxH"

    if "bulk_ratio" not in st.session_state:
        st.session_state["bulk_ratio"] = "7:9"
    if "bulk_k" not in st.session_state:
        st.session_state["bulk_k"] = "2K"
    if "bulk_w" not in st.session_state or "bulk_h" not in st.session_state:
        _w3, _h3 = _compute_preset_size(st.session_state["bulk_ratio"], st.session_state["bulk_k"])
        st.session_state["bulk_w"] = _w3
        st.session_state["bulk_h"] = _h3
    if "bulk_mode" not in st.session_state:
        st.session_state["bulk_mode"] = "Preset"

    with tabs[0]:
        prompt = st.text_area("Prompt", height=130, key="t2i_prompt")
        # Size controls (interactive)
        col1, col2, col3 = st.columns(3)
        with col1:
            size_mode = st.radio("Size mode", ["Explicit WxH", "Preset"], key="t2i_size_mode")
        # Always render Aspect ratio/Resolution for stability; disable when not Preset
        colp1, colp2 = st.columns(2)
        with colp1:
            st.selectbox(
                "Aspect ratio",
                ASPECT_RATIO_OPTIONS,
                index=ASPECT_RATIO_OPTIONS.index(st.session_state.get("t2i_ratio", "7:9")),
                format_func=_format_ratio_option,
                key="t2i_ratio",
                disabled=(size_mode != "Preset"),
            )
        with colp2:
            st.selectbox(
                "Resolution",
                ["1K", "2K", "4K"],
                index=["1K", "2K", "4K"].index(st.session_state.get("t2i_k", "2K")),
                key="t2i_k",
                disabled=(size_mode != "Preset"),
            )
        # Sync T2I numeric fields from preset when in Preset mode
        if size_mode == "Preset":
            _tw, _th = _compute_preset_size(st.session_state.get("t2i_ratio", "7:9"), st.session_state.get("t2i_k", "2K"))
            st.session_state["t2i_w"] = _tw
            st.session_state["t2i_h"] = _th
        # Explicit fields: reflect current values and switch to explicit on manual change
        with col2:
            width = st.number_input(
                "Width", min_value=256, max_value=4096,
                value=int(st.session_state.get("t2i_w", 1600)), step=32, key="t2i_w", on_change=_set_t2i_explicit
            )
        with col3:
            height = st.number_input(
                "Height", min_value=256, max_value=4096,
                value=int(st.session_state.get("t2i_h", 2048)), step=32, key="t2i_h", on_change=_set_t2i_explicit
            )

        seq_mode = st.selectbox("Batch mode", ["disabled", "auto"], index=0,
                                 help="'auto' lets the model return multiple related images.")
        max_images = None
        if seq_mode == "auto":
            max_images = st.slider("Max images", 1, 15, 4)

        save_suffix = st.text_input("Save folder suffix (optional)")
        submitted = st.button("Generate")

        if submitted:
            ph_t2i = None
            try:
                with st.spinner("Generating..."):
                    log(f"T2I start | model={st.session_state.get('model')}")
                    # Enforce multiples of 32
                    w_use = max(256, min(4096, _nearest_multiple(int(width), 32)))
                    h_use = max(256, min(4096, _nearest_multiple(int(height), 32)))
                    # Use explicit WxH (Preset populates these fields)
                    size_arg = f"{w_use}x{h_use}"
                    ph_t2i = st.empty()
                    def _on_partial(_idx: int, img: Image.Image, _size: str, _url: Optional[str]):
                        try:
                            ph_t2i.image(img, caption="")
                        except Exception:
                            pass
                    result = generate_images(
                        prompt=prompt,
                        model=st.session_state.get("model", "seedream-4-0-250828"),
                        size=size_arg,
                        resolution_tag=None,
                        sequential_mode=seq_mode,
                        max_images=max_images,
                        stream=True,
                        # watermark disabled in client
                        logger=log,
                        on_partial=_on_partial,
                    )
                log(f"T2I done | images={len(result.images)}")
                show_and_save(result, output_dir, prompt, save_suffix, references=None)
            except Exception as e:
                msg = str(e)
                st.error(msg)
                log(f"T2I error | {msg}")
                if "ModelNotOpen" in msg:
                    st.info(
                        "The model is not activated for your account. Go to Ark Console → Model List → Seedream 4.0 and click Activate. Then retry."
                    )
            finally:
                try:
                    if ph_t2i is not None:
                        ph_t2i.empty()
                except Exception:
                    pass

    with tabs[1]:
        prompt2 = st.text_area("Prompt", height=120, key="i2i_prompt")
        uploaded_files = st.file_uploader(
            "Reference images (drag & drop, 1-10)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        # Always preprocess: size reduction/format conversion (no face crop here)
        col1, col2, col3 = st.columns(3)
        with col1:
            size_mode2 = st.radio(
                "Size mode",
                ["Explicit WxH", "Preset"],
                key="i2i_mode",
            )
        # If Preset, compute target WxH BEFORE rendering number_inputs to avoid session_state mutation after instantiation
        if size_mode2 == "Preset":
            _iw, _ih = _compute_preset_size(
                st.session_state.get("i2i_ratio", "7:9"),
                st.session_state.get("i2i_k", "2K"),
            )
            st.session_state["i2i_w"] = _iw
            st.session_state["i2i_h"] = _ih
        # Width/Height on the same row like T2I
        with col2:
            width2 = st.number_input(
                "Width", min_value=256, max_value=4096,
                value=int(st.session_state.get("i2i_w", 1600)), step=32, key="i2i_w", on_change=_set_i2i_explicit
            )
        with col3:
            height2 = st.number_input(
                "Height", min_value=256, max_value=4096,
                value=int(st.session_state.get("i2i_h", 2048)), step=32, key="i2i_h", on_change=_set_i2i_explicit
            )

        # Aspect ratio / Resolution row below
        coli1, coli2 = st.columns(2)
        with coli1:
            st.selectbox(
                "Aspect ratio",
                ASPECT_RATIO_OPTIONS,
                format_func=_format_ratio_option,
                index=ASPECT_RATIO_OPTIONS.index(st.session_state.get("i2i_ratio", "7:9")),
                key="i2i_ratio",
                disabled=(size_mode2 != "Preset"),
            )
        with coli2:
            st.selectbox(
                "Resolution",
                ["1K", "2K", "4K"],
                index=["1K", "2K", "4K"].index(st.session_state.get("i2i_k", "2K")),
                key="i2i_k",
                disabled=(size_mode2 != "Preset"),
            )
        # Values already synced before number_inputs

        # width/height already rendered above
        save_suffix2 = st.text_input("Save folder suffix (optional)", key="i2i_suffix")
        submitted2 = st.button("Generate with Refs", key="i2i_generate")

        if submitted2:
            refs_pil: List[Image.Image] = []  # only for saving copies
            ref_paths: List[Path] = []
            uploaded_pairs = []
            images_data_urls: List[str] = []
            if uploaded_files:
                for f in uploaded_files[:10]:
                    original_name = getattr(f, 'name', 'ref')
                    raw = f.getvalue() if hasattr(f, 'getvalue') else f.read()
                    # build data URL with original mime; default to jpeg
                    ext = Path(original_name).suffix.lower()
                    mime = 'image/jpeg'
                    if ext in ['.png']:
                        mime = 'image/png'
                    elif ext in ['.webp']:
                        mime = 'image/webp'
                    data_url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
                    images_data_urls.append(data_url)
                    # for saving references alongside outputs
                    try:
                        pil = Image.open(io.BytesIO(raw)).convert("RGB")
                        refs_pil.append(pil)
                        uploaded_pairs.append((pil, original_name))
                    except Exception:
                        pass
                log(f"i2i refs prepared (original) | count={len(images_data_urls)}")
            w2_use = max(256, min(4096, _nearest_multiple(int(width2), 32)))
            h2_use = max(256, min(4096, _nearest_multiple(int(height2), 32)))
            size_arg2 = f"{w2_use}x{h2_use}"
            res_tag_send = None

            try:
                with st.spinner("Generating..."):
                    log(f"i2i start | refs={len(refs_pil)} model={st.session_state.get('model')}")
                    result2 = generate_images(
                        prompt=prompt2,
                        model=st.session_state.get("model", "seedream-4-0-250828"),
                        size=size_arg2,
                        resolution_tag=res_tag_send,
                        images_url_or_b64=images_data_urls if images_data_urls else None,
                        sequential_mode="disabled",
                        stream=True,
                        logger=log,
                        on_partial=lambda _i, img, _s, _u: st.image(img, caption="", width=500),
                    )
                log(f"i2i done | images={len(result2.images)}")
                show_and_save(result2, output_dir, prompt2, save_suffix2, references=ref_paths, uploaded_pairs=uploaded_pairs)
            except Exception as e:
                msg = str(e)
                st.error(msg)
                log(f"i2i error | {msg}")
                if "ModelNotOpen" in msg:
                    st.info(
                        "The model is not activated for your account. Go to Ark Console → Model List → Seedream 4.0 and click Activate. Then retry."
                    )
    with tabs[2]:
        st.subheader("Grid Generator: up to 10 reference groups × 10 prompts")
        st.caption("Uses current model. Watermark is disabled. Seed is applied only on models that support it.")

        # Optional: load groups from a root folder (top-level dirs only)
        st.markdown("**Load groups from folder (optional)**")
        groups_root = st.text_input("Groups root folder", value="inputs", key="grid_root")
        load_groups_btn = st.button("Load groups from folder", key="grid_load")
        if load_groups_btn:
            root_path = Path(groups_root)
            names: List[str] = []
            paths_by_group: List[List[Path]] = []
            if root_path.exists() and root_path.is_dir():
                subdirs = sorted([p for p in root_path.iterdir() if p.is_dir()], key=lambda p: p.name)[:10]
                for d in subdirs:
                    names.append(d.name)
                    files = [
                        p for p in d.iterdir()
                        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"]
                    ]
                    paths_by_group.append(files)
                st.success(f"Loaded {len(names)} groups from {root_path}")
            else:
                st.error("Folder not found or not a directory")
                names, paths_by_group = [], []
            st.session_state["grid_loaded_group_names"] = names
            st.session_state["grid_loaded_group_paths"] = paths_by_group
            # Keep the number input in sync; safe here before widget instantiation
            st.session_state["grid_num_groups"] = max(1, min(10, len(names)))

        # Controls
        loaded_names_boot = st.session_state.get("grid_loaded_group_names")
        if loaded_names_boot:
            # Keep the number input in sync with loaded groups
            st.session_state["grid_num_groups"] = max(1, min(10, len(loaded_names_boot)))
        num_groups = st.number_input(
            "Number of reference groups",
            min_value=1,
            max_value=10,
            value=st.session_state.get("grid_num_groups", 3),
            key="grid_num_groups",
            disabled=bool(loaded_names_boot),
        )
        if loaded_names_boot:
            st.info(f"Using {len(loaded_names_boot)} groups loaded from folder")
        num_prompts = st.number_input("Number of prompts", min_value=1, max_value=10, value=3)
        _model_lower = st.session_state.get("model", "seedream-4-0-250828").lower()
        seed_disabled = "seedream-4" in _model_lower
        seed_val = st.text_input("Seed (integer; fixed across grid)", value="12345", disabled=seed_disabled)
        # Size controls for Grid (match T2I layout)
        col1, col2, col3 = st.columns(3)
        with col1:
            grid_mode = st.radio("Size mode", ["Explicit WxH", "Preset"], index=(1 if st.session_state.get("grid_mode") == "Preset" else 0), key="grid_mode")
        if grid_mode == "Preset":
            _gw, _gh = _compute_preset_size(
                st.session_state.get("grid_ratio", "7:9"),
                st.session_state.get("grid_k", "2K"),
            )
            st.session_state["grid_w"] = _gw
            st.session_state["grid_h"] = _gh
        with col2:
            width_g = st.number_input("Width", min_value=256, max_value=4096, value=int(st.session_state.get("grid_w", 1966)), step=32, key="grid_w", on_change=_set_grid_explicit)
        with col3:
            height_g = st.number_input("Height", min_value=256, max_value=4096, value=int(st.session_state.get("grid_h", 2592)), step=32, key="grid_h", on_change=_set_grid_explicit)

        # Aspect ratio / Resolution row
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            ratio_grid = st.selectbox(
                "Aspect ratio",
                ASPECT_RATIO_OPTIONS,
                format_func=_format_ratio_option,
                index=ASPECT_RATIO_OPTIONS.index("7:9"),
                key="grid_ratio",
                disabled=(grid_mode != "Preset"),
            )
        with gcol2:
            k_grid = st.selectbox("Resolution", ["1K", "2K", "4K"], index=1, key="grid_k", disabled=(grid_mode != "Preset"))
        # Do not mutate grid_w/h after number_inputs are instantiated in this run.
        # The values are computed above when grid_mode == "Preset" before the number_inputs.
        apply_crop = st.checkbox("Apply Balanced Crop v2 (DNN) to all references (Grid only)", value=True,
                                 help="Uses your provided FaceCropper DNN. If unavailable, falls back to center square.")

        # Prompts (common for all groups)
        prompts: List[str] = []
        st.markdown("**Prompts**")
        num_prompts_int = int(num_prompts)
        for pi in range(num_prompts_int):
            prompts.append(st.text_input(f"Prompt {pi+1}", key=f"grid_prompt_{pi}"))

        # Detect {variables} in prompts
        var_pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
        vars_in_prompts = sorted({v for p in prompts for v in var_pattern.findall(p or "")})
        if vars_in_prompts:
            st.info("Detected variables: " + ", ".join(vars_in_prompts))

        group_uploads: List[List[Image.Image]] = []
        group_names: List[str] = []

        # Common reference images (up to 5), inserted at indices 1..N for all groups
        st.markdown("**Common references (0-5, inserted at indices starting from 1)**")
        common_ref_files = st.file_uploader(
            "Upload up to 5 common reference images",
            type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
            key="grid_common_ref",
            accept_multiple_files=True,
        )
        common_ref_data_urls: List[str] = []
        if common_ref_files:
            import base64 as _b64tmp
            cols_common = st.columns(5)
            for i, f in enumerate(common_ref_files[:5]):
                try:
                    raw = f.getvalue() if hasattr(f, 'getvalue') else f.read()
                    ext = Path(getattr(f, 'name', 'ref')).suffix.lower()
                    mime = None
                    if ext in [".jpg", ".jpeg"]:
                        mime = "image/jpeg"
                    elif ext in [".png"]:
                        mime = "image/png"
                    else:
                        try:
                            img = Image.open(io.BytesIO(raw)).convert("RGB")
                        except Exception:
                            try:
                                from pillow_heif import read_heif
                                heif = read_heif(io.BytesIO(raw))
                                img = Image.frombytes(heif.mode, heif.size, heif.data, "raw").convert("RGB")
                            except Exception:
                                img = None
                        if img is not None:
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=95)
                            raw = buf.getvalue()
                            mime = "image/jpeg"
                    if mime is None:
                        continue
                    common_ref_data_urls.append(f"data:{mime};base64,{_b64tmp.b64encode(raw).decode()}")
                    with cols_common[i % 5]:
                        st.image(Image.open(io.BytesIO(raw)), caption=f"image {i+1}", width=200)
                except Exception:
                    continue
        

        loaded_names: Optional[List[str]] = st.session_state.get("grid_loaded_group_names")
        loaded_paths: Optional[List[List[Path]]] = st.session_state.get("grid_loaded_group_paths")

        num_groups_int = len(loaded_names) if loaded_names else int(num_groups)
        tabs_groups = st.tabs([
            (loaded_names[i] if loaded_names and i < len(loaded_names) else f"Group {i+1}")
            for i in range(num_groups_int)
        ])
        group_vars_list: List[dict] = []
        for gi, tab in enumerate(tabs_groups):
            with tab:
                default_name = loaded_names[gi] if loaded_names and gi < len(loaded_names) else f"group_{gi+1}"
                gname = st.text_input(f"Group {gi+1} name", key=f"gname_{gi}", value=default_name)
                files = None
                preloaded_file_paths: Optional[List[Path]] = None
                if loaded_paths and gi < len(loaded_paths):
                    preloaded_file_paths = loaded_paths[gi]
                else:
                    files = st.file_uploader(
                        f"Images for group {gi+1} (0-10)",
                        type=["png", "jpg", "jpeg"],
                        accept_multiple_files=True,
                        key=f"gfiles_{gi}",
                    )
                # Per-group variables (if any)
                group_vars: dict = {}
                if vars_in_prompts:
                    st.caption("Group variables")
                    for var in vars_in_prompts:
                        group_vars[var] = st.text_input(var, key=f"g{gi}_var_{var}")
                group_vars_list.append(group_vars)
                group_names.append(gname)
                imgs: List[Image.Image] = []
                if preloaded_file_paths is not None:
                    previews_dir = output_dir / "previews" / "grid" / sanitize_for_fs(gname)
                    ensure_dir(previews_dir)
                    for idx, pth in enumerate(preloaded_file_paths[:50]):
                        try:
                            im_raw = fix_orientation(Image.open(pth)).convert("RGB")
                        except Exception:
                            # Attempt HEIC via pillow-heif if available
                            try:
                                from pillow_heif import read_heif
                                heif = read_heif(str(pth))
                                im_raw = fix_orientation(Image.frombytes(
                                    heif.mode, heif.size, heif.data, "raw"
                                )).convert("RGB")
                            except Exception:
                                continue
                        im_proc = balanced_crop_v2(im_raw, margin_ratio=0.5, target_side=1024) if apply_crop else im_raw
                        # Save preview strictly as JPEG
                        preview_path = previews_dir / f"{idx+1:02d}.jpg"
                        im_proc.save(preview_path, format="JPEG", quality=95)
                        # Build API image (jpeg)
                        api_bytes, _ = compress_to_target(im_proc, max_bytes=9_800_000, prefer_webp=False)
                        im = Image.open(io.BytesIO(api_bytes)).convert("RGB")
                        imgs.append(im)
                    group_uploads.append(imgs)
                elif files:
                    previews_dir = output_dir / "previews" / "grid" / sanitize_for_fs(gname)
                    ensure_dir(previews_dir)
                    for idx, f in enumerate(files[:10]):
                        try:
                            im_raw = fix_orientation(Image.open(f)).convert("RGB")
                        except Exception:
                            # Try HEIC from upload
                            try:
                                from pillow_heif import read_heif
                                heif = read_heif(io.BytesIO(f.read()))
                                im_raw = fix_orientation(Image.frombytes(
                                    heif.mode, heif.size, heif.data, "raw"
                                )).convert("RGB")
                            except Exception:
                                continue
                        im_proc = balanced_crop_v2(im_raw, margin_ratio=0.5, target_side=1024) if apply_crop else im_raw
                        # Save preview strictly as JPEG (no WEBP conversion)
                        preview_path = previews_dir / f"{idx+1:02d}.jpg"
                        im_proc.save(preview_path, format="JPEG", quality=95)
                        try:
                            sz_mb = preview_path.stat().st_size / 1_000_000
                        except Exception:
                            sz_mb = 0
                        log(f"grid preview | group={gname} idx={idx+1} saved JPEG {sz_mb:.2f}MB crop={'on' if apply_crop else 'off'}")
                        # Build API image (jpeg)
                        api_bytes, _ = compress_to_target(im_proc, max_bytes=9_800_000, prefer_webp=False)
                        im = Image.open(io.BytesIO(api_bytes)).convert("RGB")
                        imgs.append(im)
                    group_uploads.append(imgs)

        # prompts already collected above

        run_grid = st.button("Run Grid")
        grid_placeholder = st.empty()

        if run_grid:
            num_cols = len(group_names)
            total_cells = len(prompts) * max(1, num_cols)
            progress_start("Grid", total_cells)
            log(f"grid start | groups={len(group_uploads)} prompts={len(prompts)} parallel={st.session_state.get('max_parallel')}")

            # Session folder for reloadable Grid View
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            first_snippet = sanitize_for_fs((prompts[0] if prompts else "grid") or "grid")[:40]
            session_folder = output_dir / f"{timestamp}_{first_snippet}_grid"
            ensure_dir(session_folder)
            cells_dir = session_folder / "cells"
            ensure_dir(cells_dir)
            manifest = {
                "type": "grid",
                "created": timestamp,
                "model": st.session_state.get("model", "seedream-4-0-250828"),
                "width": int(width_g),
                "height": int(height_g),
                "seed": seed_val if not seed_disabled else None,
                "apply_crop": bool(apply_crop),
                "group_names": group_names,
                "prompts": prompts,
                "variables": group_vars_list,
                "num_rows": len(prompts),
                "num_cols": len(group_uploads),
                "cells": {},
                "refs": {},
            }
            with open(session_folder / "grid.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            # Save reference thumbnails used for API into session for later reload
            refs_dir = session_folder / "refs"
            ensure_dir(refs_dir)
            for gi, gname in enumerate(group_names):
                sub = refs_dir / sanitize_for_fs(gname)
                ensure_dir(sub)
                imgs = group_uploads[gi] if gi < len(group_uploads) else []
                ref_files = []
                for idx, im in enumerate(imgs[:10]):
                    pth = sub / f"{idx+1:02d}.jpg"
                    try:
                        im.save(pth, format="JPEG", quality=95)
                        ref_files.append(f"refs/{sanitize_for_fs(gname)}/{idx+1:02d}.jpg")
                    except Exception:
                        pass
                manifest["refs"][str(gi+1)] = {"group": gname, "files": ref_files}
            try:
                with open(session_folder / "grid.json", "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # Header with thumbnails
            cols_header = st.columns(num_cols + 1)
            with cols_header[0]:
                st.write("")
            for gi, gname in enumerate(group_names):
                with cols_header[gi+1]:
                    st.markdown(f"**{gname}**")
                    thumbs = (group_uploads[gi] if gi < len(group_uploads) else [])[:10]
                    if thumbs:
                        col_count = 2
                        total = len(thumbs)
                        rows = math.ceil(total / col_count)
                        for r in range(rows):
                            cols_t = st.columns(col_count)
                            for c in range(col_count):
                                idx = r * col_count + c
                                if idx < total:
                                    with cols_t[c]:
                                        st.image(thumbs[idx], use_column_width=True)

            # Pre-create placeholders per cell
            cell_placeholders = {}
            for pi, prompt_val in enumerate(prompts):
                row_cols = st.columns(num_cols + 1)
                with row_cols[0]:
                    label = prompt_val or f"Prompt {pi+1}"
                    st.markdown(f"<div class='prompt-label'>{html.escape(label)}</div>", unsafe_allow_html=True)
                for gi in range(num_cols):
                    with row_cols[gi+1]:
                        cell_placeholders[(pi, gi)] = st.empty()

            # Task function
            def _run_cell(pi_idx: int, gi_idx: int, prompt_val: str, imgs: List[Image.Image]):
                mapping = group_vars_list[gi_idx] if gi_idx < len(group_vars_list) else {}
                resolved_prompt = var_pattern.sub(lambda m: str(mapping.get(m.group(1), "")), prompt_val or "")
                ph = cell_placeholders.get((pi_idx, gi_idx))

                def _on_partial(_idx: int, img: Image.Image, _size: str, _url: Optional[str]):
                    try:
                        if ph is not None:
                            ph.image(img, caption="")
                    except Exception:
                        pass
                # Respect API limit: total refs (common + group) <= 10
                limit_for_group = max(0, 10 - len(common_ref_data_urls)) if common_ref_data_urls else 10
                effective_imgs = imgs[:limit_for_group] if imgs else []
                # Enforce multiples of 32
                gw_use = max(256, min(4096, _nearest_multiple(int(width_g), 32)))
                gh_use = max(256, min(4096, _nearest_multiple(int(height_g), 32)))
                _grid_seed = int(seed_val) if (not seed_disabled and seed_val.strip().isdigit()) else None
                res = generate_images(
                    prompt=resolved_prompt,
                    model=st.session_state.get("model", "seedream-4-0-250828"),
                    size=f"{gw_use}x{gh_use}",
                    pil_images=effective_imgs if effective_imgs else None,
                    images_url_or_b64=(common_ref_data_urls if common_ref_data_urls else None),
                    sequential_mode="disabled",
                    seed=_grid_seed,
                    stream=True,
                    logger=log,
                    on_partial=_on_partial,
                )
                return pi_idx, gi_idx, resolved_prompt, res

            futures = []
            max_workers = int(st.session_state.get("max_parallel", 4))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for pi, prompt_val in enumerate(prompts):
                    for gi in range(num_cols):
                        imgs = group_uploads[gi] if gi < len(group_uploads) else None
                        futures.append(ex.submit(_run_cell, pi, gi, prompt_val, imgs or []))

                for fut in as_completed(futures):
                    try:
                        pi_idx, gi_idx, resolved_prompt, res = fut.result()
                        ph = cell_placeholders.get((pi_idx, gi_idx))
                        if res.images:
                            if ph is not None:
                                ph.image(res.images[0], caption="")
                            # Save into session cells and update manifest
                            cell_name = f"r{pi_idx+1}_c{gi_idx+1}.jpg"
                            save_path = cells_dir / cell_name
                            try:
                                res.images[0].save(save_path, format="JPEG", quality=95)
                            except Exception:
                                pass
                            manifest["cells"][f"{pi_idx+1}_{gi_idx+1}"] = {
                                "file": f"cells/{cell_name}",
                                "prompt": resolved_prompt,
                                "size": res.sizes[0] if res.sizes else "",
                                "url": res.urls[0] if res.urls else "",
                            }
                            try:
                                with open(session_folder / "grid.json", "w", encoding="utf-8") as f:
                                    json.dump(manifest, f, ensure_ascii=False, indent=2)
                            except Exception:
                                pass
                            # Also save through standard saver (optional extra copy)
                            save_generation(
                                images=res.images,
                                prompt=resolved_prompt,
                                output_base=output_dir,
                                model=st.session_state.get("model", "seedream-4-0-250828"),
                                sizes=res.sizes,
                                urls=res.urls,
                                references=None,
                                suffix=f"grid_{(group_names[gi_idx] if gi_idx < len(group_names) else 'group')}_p{pi_idx+1}",
                            )
                        progress_step(1)
                    except Exception as e:
                        err = str(e)
                        log(f"grid cell error | {err}")
                        progress_step(1)
            progress_done()

            st.success(f"Grid saved to {session_folder}")

        # Reload previous grid when idle
        st.markdown("---")
        st.subheader("Grid View (load previous session)")
        load_folder = st.text_input("Grid session folder (contains grid.json)", value="", key="grid_reload")
        if st.button("Load Grid Session") and load_folder:
            sf = Path(load_folder)
            meta_path = sf / "grid.json"
            if not meta_path.exists():
                st.error("grid.json not found in the specified folder")
            else:
                try:
                    manifest = json.load(open(meta_path, "r", encoding="utf-8"))
                    gnames = manifest.get("group_names", [])
                    prompts_loaded = manifest.get("prompts", [])
                    cols_header = st.columns(len(gnames) + 1)
                    with cols_header[0]:
                        st.write("")
                    refs_map = manifest.get("refs", {})
                    for gi, gname in enumerate(gnames):
                        with cols_header[gi+1]:
                            st.markdown(f"**{gname}**")
                            # Show stored reference thumbnails (2 columns)
                            entry = refs_map.get(str(gi+1), {})
                            files = entry.get("files", [])
                            if files:
                                col_count = 2
                                total = len(files)
                                rows = math.ceil(total / col_count)
                                for r in range(rows):
                                    cols_t = st.columns(col_count)
                                    for c in range(col_count):
                                        idx = r * col_count + c
                                        if idx < total:
                                            img_path = sf / files[idx]
                                            if img_path.exists():
                                                with cols_t[c]:
                                                    st.image(Image.open(img_path), use_column_width=True)
                    for pi, ptxt in enumerate(prompts_loaded):
                        row_cols = st.columns(len(gnames) + 1)
                        with row_cols[0]:
                            st.markdown(f"**{ptxt}**")
                        for gi in range(len(gnames)):
                            with row_cols[gi+1]:
                                cell = manifest.get("cells", {}).get(f"{pi+1}_{gi+1}")
                                if cell:
                                    img_path = sf / cell.get("file", "")
                                    if img_path.exists():
                                        st.image(Image.open(img_path), use_column_width=True)
                except Exception as e:
                    st.error(str(e))

    with tabs[3]:
        st.subheader("Bulk Runner: run one prompt over all images in a folder")
        prompt_b = st.text_area("Prompt", height=120, key="bulk_prompt")
        # Size controls for Bulk
        # Bulk controls layout to match T2I
        col1, col2, col3 = st.columns(3)
        with col1:
            bulk_mode = st.radio("Size mode", ["Explicit WxH", "Preset"], index=(1 if st.session_state.get("bulk_mode") == "Preset" else 0), key="bulk_mode")
        if bulk_mode == "Preset":
            _bw, _bh = _compute_preset_size(
                st.session_state.get("bulk_ratio", "7:9"),
                st.session_state.get("bulk_k", "2K"),
            )
            st.session_state["bulk_w"] = _bw
            st.session_state["bulk_h"] = _bh
        with col2:
            width_b = st.number_input("Width", min_value=256, max_value=4096, value=int(st.session_state.get("bulk_w", 1600)), step=32, key="bulk_w", on_change=_set_bulk_explicit)
        with col3:
            height_b = st.number_input("Height", min_value=256, max_value=4096, value=int(st.session_state.get("bulk_h", 2048)), step=32, key="bulk_h", on_change=_set_bulk_explicit)

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            ratio_bulk = st.selectbox(
                "Aspect ratio",
                ASPECT_RATIO_OPTIONS,
                format_func=_format_ratio_option,
                index=ASPECT_RATIO_OPTIONS.index("7:9"),
                key="bulk_ratio",
                disabled=(bulk_mode != "Preset"),
            )
        with bcol2:
            k_bulk = st.selectbox("Resolution", ["1K", "2K", "4K"], index=1, key="bulk_k", disabled=(bulk_mode != "Preset"))
        # Avoid mutating bulk_w/h after number_inputs exist in this run; these were set above
        # when bulk_mode == "Preset" so reruns will bring them up-to-date.
        base_folder = st.text_input("Input folder (absolute or relative)", value="inputs", key="bulk_in")
        run_bulk = st.button("Run Bulk", key="bulk_run")

        if run_bulk:
            folder = Path(base_folder)
            if not folder.exists() or not folder.is_dir():
                st.error("Input folder not found.")
            else:
                # gather images
                paths = [p for p in folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
                if not paths:
                    st.error("No images in folder.")
                else:
                    progress_start("Bulk", len(paths))
                    log(f"bulk start | files={len(paths)} model={st.session_state.get('model')} parallel={st.session_state.get('max_parallel')}")
                    # create session folder now for incremental saves
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    prompt_snippet = sanitize_for_fs(prompt_b)[:60]
                    session_folder = output_dir / f"{timestamp}_{prompt_snippet}_bulk"
                    ensure_dir(session_folder)
                    meta = {
                        "prompt": prompt_b,
                        "model": st.session_state.get("model", "seedream-4-0-250828"),
                        "created": timestamp,
                        "items": [],
                    }

                    # Pre-create placeholders for immediate previews
                    st.subheader("Results (live)")
                    cols = st.columns(3)
                    bulk_placeholders = {}
                    for i, p in enumerate(paths):
                        with cols[i % 3]:
                            c = st.container()
                            c.markdown(f"**{p.name}**")
                            bulk_placeholders[p] = c.empty()

                    def _run_bulk_item(pth: Path):
                        raw = pth.read_bytes()
                        mime = 'image/jpeg'
                        if pth.suffix.lower() == '.png':
                            mime = 'image/png'
                        elif pth.suffix.lower() == '.webp':
                            mime = 'image/webp'
                        data_url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"

                        ph = bulk_placeholders.get(pth)

                        def _on_partial(_idx: int, img: Image.Image, _size: str, _url: Optional[str]):
                            try:
                                if ph is not None:
                                    ph.image(img, caption=pth.name, width=500)
                            except Exception:
                                pass
                        # Enforce multiples of 32
                        bw_use = max(256, min(4096, _nearest_multiple(int(width_b), 32)))
                        bh_use = max(256, min(4096, _nearest_multiple(int(height_b), 32)))
                        res = generate_images(
                            prompt=prompt_b,
                            model=st.session_state.get("model", "seedream-4-0-250828"),
                            size=f"{bw_use}x{bh_use}",
                            images_url_or_b64=[data_url],
                            sequential_mode="disabled",
                            stream=True,
                            on_partial=_on_partial,
                            logger=log,
                        )
                        return pth, res

                    futures = []
                    max_workers = int(st.session_state.get("max_parallel", 4))
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        for p in paths:
                            futures.append(ex.submit(_run_bulk_item, p))

                        for fut in as_completed(futures):
                            try:
                                pth, res = fut.result()
                                if res.images:
                                    img = res.images[0]
                                    save_path = session_folder / f"{pth.stem}.jpg"
                                    img.save(save_path, format="JPEG", quality=95)
                                    item = {
                                        "name": pth.name,
                                        "saved": str(save_path),
                                        "size": res.sizes[0] if res.sizes else "",
                                        "url": res.urls[0] if res.urls else "",
                                    }
                                    meta["items"].append(item)
                                    with open(session_folder / "metadata.json", "w", encoding="utf-8") as f:
                                        json.dump(meta, f, ensure_ascii=False, indent=2)
                                    st.image(img, caption=pth.name, width=500)
                                    st.write(str(save_path))
                                    log(f"bulk item saved | {pth.name} -> {save_path}")
                                progress_step(1)
                            except Exception as e:
                                st.error(str(e))
                                log(f"bulk item error | {e}")
                                progress_step(1)
                    st.success(f"Bulk completed. Saved to {session_folder}")
                    progress_done()


def show_and_save(
    result: GenerationResult,
    output_dir: Path,
    prompt: str,
    save_suffix: Optional[str],
    references: Optional[List[Path]],
    uploaded_pairs: Optional[list] = None,
) -> None:
    cols = st.columns(len(result.images)) if result.images else st.columns(1)
    for idx, img in enumerate(result.images):
        with cols[idx % len(cols)]:
            st.image(img, caption=f"Result {idx+1} – {result.sizes[idx] if idx < len(result.sizes) else ''}", width=500)

    # Auto-save immediately to the selected folder
    auto_saved: SaveResult = save_generation(
        images=result.images,
        prompt=prompt,
        output_base=output_dir,
        model=os.environ.get("MODEL", "seedream-4-0-250828"),
        sizes=result.sizes,
        urls=result.urls,
        references=references,
        reference_uploads=uploaded_pairs,
        suffix=save_suffix,
    )
    st.success(f"Saved {len(auto_saved.image_paths)} image(s) to {auto_saved.folder}")
    for p in auto_saved.image_paths:
        st.write(str(p))

    # Optional manual save button (kept for explicit control if desired)
    save = st.button("Save again", type="secondary")
    if save:
        saved: SaveResult = save_generation(
            images=result.images,
            prompt=prompt,
            output_base=output_dir,
            model=os.environ.get("MODEL", "seedream-4-0-250828"),
            sizes=result.sizes,
            urls=result.urls,
            references=references,
            reference_uploads=uploaded_pairs,
            suffix=save_suffix,
        )
        st.success(f"Saved {len(saved.image_paths)} image(s) to {saved.folder}")
        for p in saved.image_paths:
            st.write(str(p))


def main():
    output_dir = sidebar()
    render_form(output_dir)


if __name__ == "__main__":
    main()


