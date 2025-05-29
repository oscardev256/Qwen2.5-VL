# qwen_vl_utils/vision_process.py
# Copyright 2024 Alibaba Group & Hugging Face – MIT/Apache‐2.0
# Extended to handle audio input                              ▲

from __future__ import annotations

import base64
import copy
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional, Tuple, List

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# ----------------------------- Global constants ---------------------------- #
# --------------------------------------------------------------------------- #
IMAGE_FACTOR     = 28
MIN_PIXELS       = 4 * 28 * 28
MAX_PIXELS       = 16384 * 28 * 28
MAX_RATIO        = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR     = 2
FPS              = 2.0
FPS_MIN_FRAMES   = 4
FPS_MAX_FRAMES   = 768

VIDEO_TOTAL_PIXELS = int(float(os.getenv("VIDEO_MAX_PIXELS",
                                         128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")

# --------------------------------------------------------------------------- #
# ----------------------------- Helper functions ---------------------------- #
# --------------------------------------------------------------------------- #
def round_by_factor(n: int, f: int)  -> int: return round(n / f) * f
def ceil_by_factor(n: int, f: int)   -> int: return math.ceil (n / f) * f
def floor_by_factor(n: int, f: int)  -> int: return math.floor(n / f) * f

def smart_resize(
    h: int, w: int,
    *, factor:   int = IMAGE_FACTOR,
       min_pixels: int = MIN_PIXELS,
       max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """Return (h', w') divisible by `factor` and within pixel bounds."""
    if max(h, w) / min(h, w) > MAX_RATIO:
        raise ValueError(f"aspect ratio > {MAX_RATIO}")
    h_ = max(factor, round_by_factor(h, factor))
    w_ = max(factor, round_by_factor(w, factor))
    if h_ * w_ > max_pixels:               # too large → down-scale
        β  = math.sqrt((h * w) / max_pixels)
        h_ = floor_by_factor(h / β, factor)
        w_ = floor_by_factor(w / β, factor)
    elif h_ * w_ < min_pixels:             # too small → up-scale
        β  = math.sqrt(min_pixels / (h * w))
        h_ = ceil_by_factor(h * β, factor)
        w_ = ceil_by_factor(w * β, factor)
    return h_, w_

def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")

# --------------------------------------------------------------------------- #
# ------------------------------ Image loading ----------------------------- #
# --------------------------------------------------------------------------- #
def fetch_image(ele: dict, *, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    src = ele["image"] if "image" in ele else ele["image_url"]
    # 1) Load ----------------------------------------------------------------------------
    if isinstance(src, Image.Image):               img = src
    elif src.startswith(("http://", "https://")):
        with requests.get(src, stream=True) as r:
            r.raise_for_status()
            with BytesIO(r.content) as bio:
                img = copy.deepcopy(Image.open(bio))
    elif src.startswith("file://"):                img = Image.open(src[7:])
    elif src.startswith("data:image") and "base64," in src:
        b64 = src.split("base64,", 1)[1]
        with BytesIO(base64.b64decode(b64)) as bio:
            img = copy.deepcopy(Image.open(bio))
    else:                                          img = Image.open(src)
    if img is None:
        raise ValueError(f"Unrecognised image source: {src}")
    img = to_rgb(img)

    # 2) Resize --------------------------------------------------------------------------
    if "resized_height" in ele and "resized_width" in ele:
        rh, rw = smart_resize(ele["resized_height"], ele["resized_width"],
                              factor=size_factor)
    else:
        w, h   = img.size
        rh, rw = smart_resize(h, w,
                              factor=size_factor,
                              min_pixels=ele.get("min_pixels", MIN_PIXELS),
                              max_pixels=ele.get("max_pixels", MAX_PIXELS))
    return img.resize((rw, rh))

# --------------------------------------------------------------------------- #
# ------------------------------ Audio loading ----------------------------- #
# --------------------------------------------------------------------------- #
def fetch_audio(ele: dict) -> Tuple[np.ndarray, int]:
    """Return (float32 numpy array, sample_rate)."""
    src = ele.get("audio")
    if src is None:
        raise ValueError("fetch_audio: element has no 'audio' field")

    raw: bytes | None = None
    path: str  | None = None

    if isinstance(src, dict):
        raw  = src.get("bytes")
        path = src.get("path")
    elif isinstance(src, (bytes, bytearray)):      raw  = src
    elif isinstance(src, str):                     path = src
    else:
        raise ValueError(f"Unsupported audio type {type(src)}")

    if raw is not None:
        with BytesIO(raw) as bio:
            arr, sr = sf.read(bio)
    else:  # path / URL / data URI
        if path.startswith(("http://", "https://")):
            resp = requests.get(path); resp.raise_for_status()
            with BytesIO(resp.content) as bio:
                arr, sr = sf.read(bio)
        elif path.startswith("data:audio") and "base64," in path:
            data = base64.b64decode(path.split("base64,", 1)[1])
            with BytesIO(data) as bio:
                arr, sr = sf.read(bio)
        else:                                     arr, sr = sf.read(path)

    return np.asarray(arr, dtype=np.float32), sr

# --------------------------------------------------------------------------- #
# ---------------------------- Video  helpers ------------------------------ #
# --------------------------------------------------------------------------- #
def smart_nframes(ele: dict, total: int, fps: float) -> int:
    """Return frame count divisible by FRAME_FACTOR."""
    assert not ("fps" in ele and "nframes" in ele)
    if "nframes" in ele:
        n = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        tgt_fps   = ele.get("fps", FPS)
        min_f     = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_f     = floor_by_factor(ele.get("max_frames",
                                           min(FPS_MAX_FRAMES, total)), FRAME_FACTOR)
        n         = total / fps * tgt_fps
        n         = floor_by_factor(min(max(n, min_f), max_f), FRAME_FACTOR)
    if not (FRAME_FACTOR <= n <= total):
        raise ValueError(f"nframes ∉ [{FRAME_FACTOR}, {total}] – got {n}")
    return int(n)

# --------------------- TorchVision backend --------------------------------- #
def _read_video_torchvision(ele: dict) -> Tuple[torch.Tensor, float]:
    path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if path.startswith(("http://", "https://")):
            warnings.warn("torchvision<0.19 can't read HTTP videos.")
        if path.startswith("file://"): path = path[7:]

    t0 = time.time()
    vid, _aud, info = io.read_video(
        path,
        start_pts = ele.get("video_start", 0.0),
        end_pts   = ele.get("video_end"),
        pts_unit  = "sec",
        output_format="TCHW",
    )
    tot, fps = vid.size(0), info["video_fps"]
    n        = smart_nframes(ele, tot, fps)
    idx      = torch.linspace(0, tot-1, n).round().long()
    logger.info(f"torchvision read {path=}  {tot=}  fps={fps:.2f}  "
                f"{time.time()-t0:.3f}s")
    return vid[idx], n / max(tot,1e-6) * fps

# --------------------- Decord backend -------------------------------------- #
def is_decord_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("decord") is not None

def _read_video_decord(ele: dict) -> Tuple[torch.Tensor, float]:
    import decord
    path = ele["video"]; t0 = time.time()
    vr   = decord.VideoReader(path)
    tot, fps = len(vr), vr.get_avg_fps()

    n   = smart_nframes(ele, tot, fps)
    idx = torch.linspace(0, tot-1, n).round().long().tolist()
    vid = vr.get_batch(idx).permute(0,3,1,2)  # TCHW
    logger.info(f"decord read {path=} {tot=} fps={fps:.2f} "
                f"{time.time()-t0:.3f}s")
    return vid.float(), n / max(tot,1e-6) * fps

# --------------------- TorchCodec backend ---------------------------------- #
def is_torchcodec_available() -> bool:
    try:
        import importlib.util, torchcodec.decoders  # noqa: F401
        return importlib.util.find_spec("torchcodec") is not None
    except Exception:
        return False

def _read_video_torchcodec(ele: dict) -> Tuple[torch.Tensor, float]:
    from torchcodec.decoders import VideoDecoder
    num_threads = int(os.getenv("TORCHCODEC_NUM_THREADS", 8))
    path = ele["video"]; t0 = time.time()
    dec  = VideoDecoder(path, num_ffmpeg_threads=num_threads)
    fps, tot = dec.metadata.average_fps, dec.metadata.num_frames
    n   = smart_nframes(ele, tot, fps)
    idx = torch.linspace(0, tot-1, n).round().long().tolist()
    vid = dec.get_frames_at(indices=idx).data  # TCHW
    logger.info(f"torchcodec read {path=} {tot=} fps={fps:.2f} "
                f"{time.time()-t0:.3f}s")
    return vid.float(), n / max(tot,1e-6) * fps

# --------------------- Backend registry ------------------------------------ #
VIDEO_READER_BACKENDS = {
    "torchvision": _read_video_torchvision,
    "decord":      _read_video_decord,
    "torchcodec":  _read_video_torchcodec,
}

FORCE_BACKEND = os.getenv("FORCE_QWENVL_VIDEO_READER")

@lru_cache(1)
def _backend() -> str:
    if FORCE_BACKEND:                  return FORCE_BACKEND
    if is_torchcodec_available():      return "torchcodec"
    if is_decord_available():          return "decord"
    return "torchvision"

# --------------------- Public video entry  --------------------------------- #
def fetch_video(
    ele: dict,
    *, image_factor: int = IMAGE_FACTOR,
       return_video_sample_fps: bool = False,
) -> Tuple[torch.Tensor | List[Image.Image], float | None]:
    """
    When ele['video'] is a path → return a TCHW tensor;
    when it's a list/tuple of frames → return list[Image.Image].
    """
    src = ele["video"]
    # ---------- Path/URL/video file --------------------------------------------------- #
    if isinstance(src, str):
        try:
            vid, sample_fps = VIDEO_READER_BACKENDS[_backend()](ele)
        except Exception as e:
            logger.warning(f"backend {_backend()} failed: {e}; falling back to torchvision")
            vid, sample_fps = _read_video_torchvision(ele)

        n, _, h, w = vid.shape
        min_p   = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_p = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_p   = max(min(VIDEO_MAX_PIXELS, total_p/n*FRAME_FACTOR),
                      int(min_p*1.05))
        max_p   = min(ele.get("max_pixels", max_p), max_p)

        if "resized_height" in ele and "resized_width" in ele:
            rh, rw = smart_resize(ele["resized_height"], ele["resized_width"],
                                  factor=image_factor)
        else:
            rh, rw = smart_resize(h, w,
                                  factor=image_factor,
                                  min_pixels=min_p,
                                  max_pixels=max_p)
        vid = transforms.functional.resize(
            vid, [rh, rw],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        ).float()

        return (vid, sample_fps) if return_video_sample_fps else (vid, None)

    # ---------- Pre-extracted frames -------------------------------------------------- #
    assert isinstance(src, (list, tuple)), "`video` must be str or list"
    meta = ele.copy(); meta.pop("video", None); meta.pop("type", None)
    imgs = [fetch_image({"image": f, **meta}, size_factor=image_factor) for f in src]
    n    = ceil_by_factor(len(imgs), FRAME_FACTOR)
    imgs.extend([imgs[-1]] * (n - len(imgs)))  # pad to multiple of FRAME_FACTOR
    return (imgs, meta.get("fps", FPS)) if return_video_sample_fps else (imgs, None)

# --------------------------------------------------------------------------- #
# ------------------------ Conversation utilities --------------------------- #
# --------------------------------------------------------------------------- #
def extract_vision_info(convs: list[dict] | list[list[dict]]) -> List[dict]:
    if isinstance(convs[0], dict): convs = [convs]
    infos: list[dict] = []
    for conv in convs:
        for msg in conv:
            if isinstance(msg["content"], list):
                for ele in msg["content"]:
                    if (
                        "image" in ele or "image_url" in ele or
                        "video" in ele or "audio" in ele or
                        ele.get("type","") in ("image","image_url","video","audio")
                    ):
                        infos.append(ele)
    return infos

def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    *, return_video_kwargs: bool = False
) -> Tuple[
    List[Image.Image] | None,
    List[torch.Tensor | List[Image.Image]] | None,
    List[np.ndarray] | None | dict
]:
    imgs, vids, auds, fps_list = [], [], [], []
    for info in extract_vision_info(conversations):
        if "image" in info or "image_url" in info:
            imgs.append(fetch_image(info))
        elif "video" in info:
            v, fps = fetch_video(info, return_video_sample_fps=True)
            vids.append(v); fps_list.append(fps)
        elif "audio" in info:
            #arr, _sr = fetch_audio(info); auds.append(arr)
            arr, sr = fetch_audio(info)
            print("Process info.....")
            auds.append((arr, sr))            # <-- keep SR
        else:
            raise ValueError("Unknown vision info type")

    if not imgs: imgs = None
    if not vids: vids = None
    if not auds: auds = None

    if return_video_kwargs:
        return imgs, vids, {"fps": fps_list}
    return imgs, vids, auds
