# -*- coding: utf-8 -*-
"""Processed jsonl + memmap dataset.

This dataset matches the *VAR stage-2* preprocessing format you provided:
  <proc_dir>/samples.jsonl
  <proc_dir>/samples_offsets.pt
  <proc_dir>/shards/shard_XXXXXX.meta.pt
  <proc_dir>/shards/<kv_bin>.bin

Each jsonl line contains at least: {"path": <img_path>, "shard": int, "idx": int}

We provide two modes:
  - return_kv=False: only load images (recommended for tokenizer stage-1 / fine-tuning).
  - return_kv=True : also load per-sample kv embeddings via memmap and pack them into
    an Infinity-style text_cond_tuple in the collate function.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None





def _resolve_samples_and_offsets(proc_dir: Path, train: bool) -> Tuple[Path, Path]:
    """Resolve jsonl + offsets paths for train/val splits.

    We follow the validation loading style typically used in the provided preprocessing:
      - train: samples.jsonl + samples_offsets.pt
      - val  : tries common variants (samples_val.jsonl, val_samples.jsonl, etc.)
    If no validation files are found, we fall back to train files.
    """
    if train:
        return proc_dir / "samples.jsonl", proc_dir / "samples_offsets.pt"

    candidates = [
        ("samples_val.jsonl", "samples_val_offsets.pt"),
        ("val_samples.jsonl", "val_samples_offsets.pt"),
        ("samples_valid.jsonl", "samples_valid_offsets.pt"),
        ("samples_validation.jsonl", "samples_validation_offsets.pt"),
        ("validation_samples.jsonl", "validation_samples_offsets.pt"),
        ("samples.jsonl", "samples_offsets.pt"),  # fallback
    ]
    for a, b in candidates:
        pa, pb = proc_dir / a, proc_dir / b
        if pa.exists() and pb.exists():
            return pa, pb
    # last resort (should not happen if proc_dir exists)
    return proc_dir / "samples.jsonl", proc_dir / "samples_offsets.pt"
def _pil_resize_shorter_side(im: Image.Image, target: int) -> Image.Image:
    w, h = im.size
    scale = target / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return im.resize((nw, nh), resample=Image.BICUBIC)


def _pil_center_crop(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    return im.crop((left, top, left + size, top + size))


def load_image_to_tensor(img_path: str, size: int, random_flip: bool = False, flip_prob: float = 0.5) -> torch.Tensor:
    """Load RGB image, resize shorter side, center crop to size, map to [-1, 1]."""
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        if random_flip and (random.random() < float(flip_prob)):
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = _pil_resize_shorter_side(im, size)
        im = _pil_center_crop(im, size)
        arr = np.asarray(im).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]
        ten = ten * 2.0 - 1.0
        return ten


class MemmapShardEmbeddingReader:
    """Load shard meta via torch.load (small), kv via numpy.memmap (no full RAM load)."""

    def __init__(self, proc_dir: str, cache_size: int = 1):
        self.proc_dir = Path(proc_dir)
        self.shards_dir = self.proc_dir / "shards"
        if not self.shards_dir.exists():
            raise FileNotFoundError(f"Missing shards dir: {self.shards_dir}")

        self._meta_cache: Dict[int, Dict[str, Any]] = {}
        self._mm_cache: "OrderedDict[int, np.memmap]" = OrderedDict()
        self.cache_size = int(max(1, cache_size))

    def load_meta(self, shard_id: int) -> Dict[str, Any]:
        if shard_id in self._meta_cache:
            return self._meta_cache[shard_id]
        p = self.shards_dir / f"shard_{shard_id:06d}.meta.pt"
        pack = torch.load(p, map_location="cpu")
        self._meta_cache[shard_id] = pack
        return pack

    def _open_memmap(self, shard_id: int) -> np.memmap:
        meta = self.load_meta(shard_id)
        rows = int(meta["rows"])
        hidden = int(meta["hidden_size"])
        dtype = str(meta.get("dtype", "fp16"))
        kv_bin = str(meta["kv_bin"])
        p = self.shards_dir / kv_bin

        np_dtype = np.float16 if dtype == "fp16" else np.float32
        mm = np.memmap(str(p), dtype=np_dtype, mode="r", shape=(rows, hidden))
        return mm

    def _get_mm(self, shard_id: int) -> np.memmap:
        if shard_id in self._mm_cache:
            self._mm_cache.move_to_end(shard_id)
            return self._mm_cache[shard_id]

        mm = self._open_memmap(shard_id)
        self._mm_cache[shard_id] = mm
        self._mm_cache.move_to_end(shard_id)

        # LRU eviction
        while len(self._mm_cache) > self.cache_size:
            _, old_mm = self._mm_cache.popitem(last=False)
            del old_mm
        return mm

    def get(self, shard_id: int, idx: int) -> Tuple[torch.Tensor, int]:
        meta = self.load_meta(shard_id)
        offsets: torch.Tensor = meta["offsets"]
        lens: torch.Tensor = meta["lens"]

        st = int(offsets[idx].item())
        ed = int(offsets[idx + 1].item())
        le = int(lens[idx].item())

        mm = self._get_mm(shard_id)
        sl = mm[st:ed]  # [L,C] view
        kv = torch.from_numpy(np.asarray(sl))  # float16/float32 CPU tensor
        return kv, le


def collate_infinity_text_cond(batch: List[Dict[str, Any]]):
    """Pack variable-length kv into Infinity-style tuple.

    Returns a dict compatible with BitVAE training loops:
      image: [B,3,H,W]
      text_cond_tuple: (kv_compact, lens, cu_seqlens_k, max_seqlen_k)
      type: ["image", ...]
      path: list[str]
    """
    imgs = torch.stack([b["image"] for b in batch], dim=0)

    kv_list = [b["kv"] for b in batch]
    lens = torch.tensor([int(kv.shape[0]) for kv in kv_list], dtype=torch.int32)
    max_seqlen = int(lens.max().item()) if lens.numel() else 0
    cu = torch.zeros((lens.numel() + 1,), dtype=torch.int32)
    cu[1:] = torch.cumsum(lens, dim=0)
    kv_compact = torch.cat([kv.contiguous() for kv in kv_list], dim=0).contiguous()

    out = {
        "image": imgs,
        "text_cond_tuple": (kv_compact, lens, cu, max_seqlen),
        "type": ["image"] * len(batch),
        "path": [b.get("path", "") for b in batch],
        "height": [b.get("height", None) for b in batch],
        "width": [b.get("width", None) for b in batch],
    }
    return out


class ProcessedJsonlMemmapDataset(Dataset):
    """Processed jsonl dataset backed by memmap shards.

    External contract is aligned to BitVAE training code:
      - returns dict with key "image" (tensor [-1,1]) and "type".
      - optionally returns kv + len when return_kv=True.

    Note on randomness:
      We intentionally use Python's global random module so that DataLoader's
      worker_init_fn seeding controls per-worker randomness (resolution choice, flips).
    """

    def __init__(
        self,
        proc_dir: str,
        res_list: List[int],
        seed: int = 0,
        cache_size: int = 1,
        return_kv: bool = False,
        random_flip: bool = False,
        flip_prob: float = 0.5,
        train: bool = True,
        val_proc_dir: Optional[str] = None,
    ):
        super().__init__()
        self.train = bool(train)

        base_dir = Path(proc_dir)
        if (not self.train) and val_proc_dir:
            base_dir = Path(val_proc_dir)
        self.proc_dir = base_dir

        self.samples_path, self.offsets_path = _resolve_samples_and_offsets(self.proc_dir, train=self.train)
        if not self.samples_path.exists():
            raise FileNotFoundError(f"Missing {self.samples_path}")
        if not self.offsets_path.exists():
            raise FileNotFoundError(f"Missing {self.offsets_path}")
        self.res_list = list(res_list)
        self.return_kv = bool(return_kv)
        self.random_flip = bool(random_flip) and self.train
        self.flip_prob = float(flip_prob)

        # Seed python random at dataset construction for deterministic single-worker runs.
        # In multi-worker, worker_init_fn will reseed global random per worker.
        random.seed(int(seed))

        self.reader: Optional[MemmapShardEmbeddingReader] = None
        if self.return_kv:
            self.reader = MemmapShardEmbeddingReader(str(proc_dir), cache_size=cache_size)

        self.offsets = torch.load(self.offsets_path, map_location="cpu").long()
        self._fh = None  # lazy open per worker

    def __len__(self):
        return int(self.offsets.numel())

    def _ensure_fh(self):
        if self._fh is None:
            self._fh = open(self.samples_path, "rb")

    def _read_rec(self, i: int) -> Dict[str, Any]:
        self._ensure_fh()
        off = int(self.offsets[i].item())
        self._fh.seek(off)
        line = self._fh.readline()
        return json.loads(line.decode("utf-8"))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self._read_rec(idx)
        img_path = rec["path"]

        res = self.res_list[0] if ((not self.train) or (len(self.res_list) == 1)) else random.choice(self.res_list)
        img = load_image_to_tensor(img_path, res, random_flip=self.random_flip, flip_prob=self.flip_prob)

        out: Dict[str, Any] = {
            "image": img,
            "type": "image",
            "path": img_path,
            "height": int(res),
            "width": int(res),
        }

        if self.return_kv:
            assert self.reader is not None
            shard_id = int(rec["shard"])
            local_idx = int(rec["idx"])
            kv, le = self.reader.get(shard_id, local_idx)
            out["kv"] = kv
            out["len"] = le

        return out
