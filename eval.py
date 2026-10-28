import os
import sys
from pathlib import Path

os.environ.setdefault("CC", "gcc")
os.environ.setdefault("CXX", "g++")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch, torchvision
import zlib  
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List

from PIL import Image

import lpips
from DISTS_pytorch import DISTS

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tools.run_infinity import *
from compression.util import *
from compression.sender import encoding
from compression.reciever import decompress_cfg, decoding

PROMPT_ALPHABET = "0123456789,"
PROMPT_EOS_SYMBOL = len(PROMPT_ALPHABET)
PROMPT_NUM_SYMBOLS = len(PROMPT_ALPHABET) + 1
PROMPT_PAYLOAD_LEN_BITS = 32

_ARITH_STATE_BITS = 32
_ARITH_FULL = 1 << _ARITH_STATE_BITS
_ARITH_HALF = _ARITH_FULL >> 1
_ARITH_QUARTER = _ARITH_HALF >> 1
_ARITH_THREE_QUARTER = _ARITH_QUARTER * 3
_ARITH_MAX_TOTAL = 1 << 15


def _encode_prompt_ids_t5(tokenizer, prompt: str, max_len: int = 512):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    ids = enc["input_ids"][0].tolist()
    return list(map(int, ids))


def _adaptive_cumulative(freqs):
    cum = [0]
    s = 0
    for f in freqs:
        s += int(f)
        cum.append(s)
    return cum


def _adaptive_update(freqs, sym):
    freqs[sym] += 1
    if sum(freqs) >= _ARITH_MAX_TOTAL:
        for i, f in enumerate(freqs):
            freqs[i] = max(1, (f + 1) // 2)


def _pack_bits_to_bytes(bits):
    if not bits:
        return b""
    out = bytearray((len(bits) + 7) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(out)


class _BitReader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.pos = 0

    def read(self):
        if self.pos >= len(self.payload) * 8:
            self.pos += 1
            return 0
        byte = self.payload[self.pos >> 3]
        bit = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit


def _adaptive_arith_encode_symbols(symbols):
    freqs = [1] * PROMPT_NUM_SYMBOLS
    low = 0
    high = _ARITH_FULL - 1
    pending = 0
    out_bits = []

    def emit(bit):
        nonlocal pending
        out_bits.append(bit)
        if pending:
            out_bits.extend([1 - bit] * pending)
            pending = 0

    for sym in symbols:
        cum = _adaptive_cumulative(freqs)
        total = cum[-1]
        width = high - low + 1
        high = low + (width * cum[sym + 1] // total) - 1
        low = low + (width * cum[sym] // total)

        while True:
            if high < _ARITH_HALF:
                emit(0)
            elif low >= _ARITH_HALF:
                emit(1)
                low -= _ARITH_HALF
                high -= _ARITH_HALF
            elif low >= _ARITH_QUARTER and high < _ARITH_THREE_QUARTER:
                pending += 1
                low -= _ARITH_QUARTER
                high -= _ARITH_QUARTER
            else:
                break
            low = low * 2
            high = high * 2 + 1

        if sym != PROMPT_EOS_SYMBOL:
            _adaptive_update(freqs, sym)

    pending += 1
    if low < _ARITH_QUARTER:
        emit(0)
    else:
        emit(1)

    return _pack_bits_to_bytes(out_bits), len(out_bits)


def _adaptive_arith_decode_symbols(payload: bytes, max_symbols: int = 100000):
    freqs = [1] * PROMPT_NUM_SYMBOLS
    low = 0
    high = _ARITH_FULL - 1
    reader = _BitReader(payload)
    code = 0
    for _ in range(_ARITH_STATE_BITS):
        code = (code << 1) | reader.read()

    decoded = []
    for _ in range(max_symbols):
        cum = _adaptive_cumulative(freqs)
        total = cum[-1]
        width = high - low + 1
        value = ((code - low + 1) * total - 1) // width

        sym = None
        for s in range(PROMPT_NUM_SYMBOLS):
            if cum[s] <= value < cum[s + 1]:
                sym = s
                break
        if sym is None:
            raise RuntimeError("Adaptive arithmetic decode failed to locate symbol")

        high = low + (width * cum[sym + 1] // total) - 1
        low = low + (width * cum[sym] // total)

        while True:
            if high < _ARITH_HALF:
                pass
            elif low >= _ARITH_HALF:
                low -= _ARITH_HALF
                high -= _ARITH_HALF
                code -= _ARITH_HALF
            elif low >= _ARITH_QUARTER and high < _ARITH_THREE_QUARTER:
                low -= _ARITH_QUARTER
                high -= _ARITH_QUARTER
                code -= _ARITH_QUARTER
            else:
                break
            low = low * 2
            high = high * 2 + 1
            code = code * 2 + reader.read()

        if sym == PROMPT_EOS_SYMBOL:
            return decoded

        decoded.append(sym)
        _adaptive_update(freqs, sym)

    raise RuntimeError("Adaptive arithmetic decode exceeded max_symbols without EOS")


def _arith_encode_ids_packet(ids):
    if ids is None:
        ids = []

    text = ",".join(map(str, ids))
    char_to_sym = {ch: i for i, ch in enumerate(PROMPT_ALPHABET)}
    symbols = []
    for ch in text:
        if ch not in char_to_sym:
            raise ValueError(f"Unexpected prompt symbol: {ch!r}")
        symbols.append(char_to_sym[ch])
    symbols.append(PROMPT_EOS_SYMBOL)

    payload, arithmetic_bit_length = _adaptive_arith_encode_symbols(symbols)
    payload_bits = len(payload) * 8  
    side_bits = PROMPT_PAYLOAD_LEN_BITS
    total_bits = payload_bits + side_bits

    return {
        "payload": payload,
        "char_len": len(text),
        "ids_len": len(ids),
        "arithmetic_bit_length": arithmetic_bit_length,
        "payload_bits": payload_bits,
        "cdf_bits": 0,
        "side_bits": side_bits,
        "bits": total_bits,
    }


def _arith_decode_ids_packet(packet):
    decoded_syms = _adaptive_arith_decode_symbols(packet["payload"])
    text = "".join(PROMPT_ALPHABET[int(s)] for s in decoded_syms)
    ids = [int(x) for x in text.split(",") if x != ""]
    return ids


def _decode_prompt_text_from_ids_t5(tokenizer, ids):
    if ids is None or len(ids) == 0:
        return ""
    try:
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
    except TypeError:
        return tokenizer.decode(ids)

def _prompt_zlib_bytes(prompt: str) -> int:
    b = prompt.encode("utf-8", errors="ignore")
    return len(zlib.compress(b))

def compress(args, vae, scale_schedule, scale_q, infinity, text_tokenizer, text_encoder, img_path, text):
    inp_B3HW = load_img(img_path, args)
    raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW.to(device), scale_schedule=scale_schedule)
    x_BLC_wo_prefix, gt_ms_idx_Bl = mask_quant(vae, scale_q, raw_features, device)
    trans_list, help_list, bpp = encoding(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl)

    return trans_list, help_list, bpp, gt_ms_idx_Bl

def _load_tensor_01(path: str) -> torch.Tensor:
    """
    Return [1,3,H,W], float32, [0,1].
    Metrics are evaluated on the actually saved reconstruction image,
    matching the previous offline evaluation pipeline.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()



def _append_metric(
    aggregates: Dict[str, Dict[str, List[float]]],
    scale_idx: str,
    metrics: Dict[str, float],
):
    if scale_idx not in aggregates:
        aggregates[scale_idx] = {
            "bpp": [],
            "img_bpp": [],
            "text_bpp": [],
            "text_payload_bpp": [],
            "text_side_bpp": [],
            "lpips": [],
            "dists": [],
        }

    for key in aggregates[scale_idx]:
        aggregates[scale_idx][key].append(float(metrics[key]))


def _write_partial_summary(
    path: str,
    model_name: str,
    scale_aggregates: Dict[str, Dict[str, List[float]]],
    completed_images: int,
):
    """Save incremental averages so partial runs retain useful results."""
    partial = {
        "status": "running",
        "model_name": model_name,
        "completed_images": completed_images,
        "summary": {},
    }

    for scale_idx, vals in scale_aggregates.items():
        partial["summary"][scale_idx] = {
            "avg_bpp": float(np.mean(vals["bpp"])),
            "avg_img_bpp": float(np.mean(vals["img_bpp"])),
            "avg_text_bpp": float(np.mean(vals["text_bpp"])),
            "avg_lpips": float(np.mean(vals["lpips"])),
            "avg_dists": float(np.mean(vals["dists"])),
            "image_count": len(vals["bpp"]),
        }

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(partial, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def decompress_modified(
    args,
    infinity,
    vae,
    scale_schedule,
    text,
    text_tokenizer,
    text_encoder,
    gt_ms_idx_Bl,
    rec_base_path,
    img_path,
    img_name,
    trans_list,
    help_list,
    device,
    h,
    w,
    text_bpp,
    text_payload_bpp,
    text_side_bpp,
    lpips_fn,
    dists_fn,
):
    dec_idx = decoding(
        args,
        infinity,
        vae,
        scale_schedule,
        text,
        text_tokenizer,
        text_encoder,
        gt_ms_idx_Bl,
        trans_list,
        help_list,
    )

    scale_results = {}

    std_tensor = _load_tensor_01(img_path)

    for gt_leak in range(0, len(gt_ms_idx_Bl)):
        sum_bit = 0
        for i in range(gt_leak + 1):
            sum_bit += len(help_list[i])
            for j in range(len(trans_list[i])):
                sum_bit += len(trans_list[i][j])

        current_img_bpp = sum_bit / (h * w)
        current_total_bpp = current_img_bpp + text_bpp

        scale_folder = os.path.join(
            rec_base_path,
            f"scale_{gt_leak}",
        )
        os.makedirs(scale_folder, exist_ok=True)

        img = decompress_cfg(
            infinity,
            vae,
            scale_schedule,
            text,
            text_tokenizer,
            text_encoder,
            gt_leak,
            dec_idx,
        )

        rec_img_path = os.path.join(scale_folder, img_name)
        torchvision.utils.save_image(img.cpu(), rec_img_path)

        rec_tensor = _load_tensor_01(rec_img_path)

        if std_tensor.shape != rec_tensor.shape:
            raise ValueError(
                "GT / reconstruction shape mismatch:\n"
                f"GT   : {img_path} {tuple(std_tensor.shape)}\n"
                f"Recon: {rec_img_path} {tuple(rec_tensor.shape)}"
            )

        gt_gpu = std_tensor.to(device)
        rec_gpu = rec_tensor.to(device)

        with torch.no_grad():
            lpips_val = float(
                lpips_fn(
                    gt_gpu * 2.0 - 1.0,
                    rec_gpu * 2.0 - 1.0,
                ).item()
            )
            dists_val = float(
                dists_fn(
                    gt_gpu,
                    rec_gpu,
                ).item()
            )

        del gt_gpu, rec_gpu

        scale_results[str(gt_leak)] = {
            "bpp": float(current_total_bpp),
            "total_bpp": float(current_total_bpp),
            "img_bpp": float(current_img_bpp),
            "text_bpp": float(text_bpp),
            "text_payload_bpp": float(text_payload_bpp),
            "text_side_bpp": float(text_side_bpp),
            "lpips": lpips_val,
            "dists": dists_val,
            "rec_path": str(Path(rec_img_path).relative_to(rec_base_path).as_posix()),
        }

    return scale_results


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Evaluate progressive image compression with BPP, LPIPS and DISTS."
    )
    parser.add_argument(
        "--vae-ckpt", type=Path,
        default=PROJECT_ROOT / "checkpoints" / "stage1_bsqvae.ckpt",
        help="Stage-1 BSQ-VAE checkpoint",
    )
    parser.add_argument(
        "--infinity-ckpt", type=Path,
        default=PROJECT_ROOT / "checkpoints" / "stage2_infinity.pth",
        help="Stage-2 Infinity checkpoint (slim model state_dict)",
    )
    parser.add_argument(
        "--t5-path", default=os.environ.get("T5_MODEL_PATH", "google/flan-t5-xl"),
        help="Local FLAN-T5-XL directory or a Hugging Face model identifier",
    )
    parser.add_argument(
        "--dataset-jsonl", type=Path,
        default=PROJECT_ROOT / "dataset" / "DIV2K.jsonl",
        help="JSONL with img_path and txt fields",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results" / "DIV2K",
    )
    parser.add_argument("--pn", default="1M", help="Infinity scale-schedule preset")
    parser.add_argument("--max-images", type=int, default=0, help="0 means evaluate all images")
    parser.add_argument(
        "--image-root", type=Path, default=None,
        help="Optional new image folder; matches images by basename (for migrated absolute paths)",
    )
    parser.add_argument("--prompt-bits-mode", choices=["adaptive", "zlib"], default="adaptive")
    parser.add_argument("--no-prompt-bits", action="store_true")
    cli = parser.parse_args()
    for name, path in (("VAE checkpoint", cli.vae_ckpt),
                       ("Infinity checkpoint", cli.infinity_ckpt),
                       ("Dataset JSONL", cli.dataset_jsonl)):
        if not path.is_file():
            parser.error(f"{name} not found: {path}")
    if cli.max_images < 0:
        parser.error("--max-images must be >= 0")
    if not torch.cuda.is_available():
        parser.error("This model loader requires CUDA; select a GPU with CUDA_VISIBLE_DEVICES")
    return cli


if __name__ == "__main__":
    cli = parse_cli()
    model_name = cli.infinity_ckpt.stem
    current_model_path = str(cli.infinity_ckpt.resolve())
    dataset_jsonl = str(cli.dataset_jsonl.resolve())

    args = argparse.Namespace(
        pn=cli.pn,
        model_path=current_model_path,
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=str(cli.vae_ckpt.resolve()),
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type="infinity_2b",
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=cli.t5_path,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir=str(cli.output_dir),
        enable_model_cache=0,
        checkpoint_type="torch",
        seed=0,
        bf16=0,
        rec_path=str(cli.output_dir.resolve()),
        add_prompt_bits=int(not cli.no_prompt_bits),
        prompt_bits_mode=cli.prompt_bits_mode,
        tlen=512,
    )

    device = "cuda"
    metric_device = torch.device(device)
    os.makedirs(args.rec_path, exist_ok=True)
    print(f"[*] Stage-1 VAE: {args.vae_path}")
    print(f"[*] Stage-2 Infinity: {args.model_path}")
    print(f"[*] Dataset: {dataset_jsonl}")
    print(f"[*] Output: {args.rec_path}")

    print("[1/3] Loading T5 tokenizer/text encoder...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    print("[2/3] Loading visual tokenizer...")
    vae = load_visual_tokenizer(args)

    print("[3/3] Loading LPIPS + DISTS...")
    lpips_evaluator = lpips.LPIPS(net="alex").to(metric_device).eval()
    dists_evaluator = DISTS().to(metric_device).eval()

    json_data = []
    with open(dataset_jsonl, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                json_data.append(json.loads(line))
    if cli.max_images:
        json_data = json_data[:cli.max_images]
    if not json_data:
        raise ValueError(f"No evaluation images in {dataset_jsonl}")
    print(f"[*] Loaded {len(json_data)} evaluation images")

    details_jsonl_path = os.path.join(args.rec_path, "metrics_details.jsonl")
    summary_json_path = os.path.join(args.rec_path, "metrics_summary.json")
    partial_json_path = os.path.join(args.rec_path, "metrics_partial.json")
    summary_csv_path = os.path.join(args.rec_path, "avg_metrics.csv")

    print(f"[*] Loading selected Infinity checkpoint: {current_model_path}")
    infinity = load_transformer(vae, args)

    scale_aggregates: Dict[str, Dict[str, List[float]]] = {}

    with open(
        details_jsonl_path,
        "w",
        encoding="utf-8",
    ) as details_f:

        for image_idx, data in enumerate(json_data):
            raw_image_path = Path(data["img_path"])
            if cli.image_root is not None:
                img_path = str(cli.image_root / raw_image_path.name)
            elif raw_image_path.is_absolute():
                img_path = str(raw_image_path)
            else:
                img_path = str(cli.dataset_jsonl.parent / raw_image_path)
            text = data["txt"]
            img_name = os.path.basename(img_path)

            print(
                f"\n[{image_idx + 1}/{len(json_data)}] "
                f"Processing {img_name}"
            )

            inp_B3HW = load_img(img_path, args)
            _, _, h, w = inp_B3HW.shape

            h_div_w = h / w
            h_div_w_template_ = h_div_w_templates[
                np.argmin(
                    np.abs(
                        h_div_w_templates - h_div_w
                    )
                )
            ]

            scale_schedule = dynamic_resolution_h_w[
                h_div_w_template_
            ][args.pn]["scales"]
            scale_schedule = [
                (1, scale_h, scale_w)
                for (_, scale_h, scale_w)
                in scale_schedule
            ]

            scale_q = [
                (
                    scale_schedule[i][0],
                    scale_schedule[i][1],
                    scale_schedule[i][2],
                    int(
                        (i + 1)
                        // ((len(scale_schedule) // 3) + 1)
                        + 2
                    ),
                )
                for i in range(len(scale_schedule))
            ]
            prompt_bits = 0
            prompt_payload_bits = 0
            prompt_side_bits = 0
            text_for_decode = text

            if args.add_prompt_bits:
                if args.prompt_bits_mode == "adaptive":
                    prompt_ids = _encode_prompt_ids_t5(
                        text_tokenizer,
                        text,
                        max_len=args.tlen,
                    )
                    prompt_packet = _arith_encode_ids_packet(
                        prompt_ids
                    )

                    prompt_bits = prompt_packet["bits"]
                    prompt_payload_bits = prompt_packet[
                        "payload_bits"
                    ]
                    prompt_side_bits = prompt_packet[
                        "side_bits"
                    ]

                    recovered_ids = _arith_decode_ids_packet(
                        prompt_packet
                    )
                    assert (
                        recovered_ids == prompt_ids
                    ), "Prompt arithmetic decode mismatch!"

                    text_for_decode = (
                        _decode_prompt_text_from_ids_t5(
                            text_tokenizer,
                            recovered_ids,
                        )
                    )

                elif args.prompt_bits_mode == "zlib":
                    prompt_payload_bits = (
                        _prompt_zlib_bytes(text) * 8
                    )
                    prompt_bits = prompt_payload_bits
                    prompt_side_bits = 0
                    text_for_decode = text

                else:
                    raise ValueError(
                        f"Unknown prompt_bits_mode: "
                        f"{args.prompt_bits_mode}"
                    )

            prompt_bpp = (
                prompt_bits / (h * w)
                if args.add_prompt_bits
                else 0.0
            )
            prompt_payload_bpp = (
                prompt_payload_bits / (h * w)
                if args.add_prompt_bits
                else 0.0
            )
            prompt_side_bpp = (
                prompt_side_bits / (h * w)
                if args.add_prompt_bits
                else 0.0
            )

            print(
                f"    Text Bits={prompt_bits} | "
                f"Payload={prompt_payload_bits} | "
                f"Side={prompt_side_bits} | "
                f"Text BPP={prompt_bpp:.8f}"
            )

            with (
                torch.no_grad(),
                torch.cuda.amp.autocast(
                    dtype=torch.bfloat16
                ),
            ):
                trans_list, help_list, _, gt_ms_idx_Bl = compress(
                    args,
                    vae,
                    scale_schedule,
                    scale_q,
                    infinity,
                    text_tokenizer,
                    text_encoder,
                    img_path,
                    text,
                )

                scale_metrics = decompress_modified(
                    args,
                    infinity,
                    vae,
                    scale_schedule,
                    text_for_decode,
                    text_tokenizer,
                    text_encoder,
                    gt_ms_idx_Bl,
                    args.rec_path,
                    img_path,
                    img_name,
                    trans_list,
                    help_list,
                    device,
                    h,
                    w,
                    prompt_bpp,
                    prompt_payload_bpp,
                    prompt_side_bpp,
                    lpips_evaluator,
                    dists_evaluator,
                )

            image_record = {
                "image_index": image_idx,
                "image_name": img_name,
                "image_path": raw_image_path.name,
                "text": text,
                "text_bits": int(prompt_bits),
                "text_payload_bits": int(
                    prompt_payload_bits
                ),
                "text_side_bits": int(prompt_side_bits),
                "text_bpp": float(prompt_bpp),
                "text_payload_bpp": float(
                    prompt_payload_bpp
                ),
                "text_side_bpp": float(
                    prompt_side_bpp
                ),
                "scales_data": scale_metrics,
            }

            details_f.write(
                json.dumps(
                    image_record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            details_f.flush()

            for scale_idx, metrics in scale_metrics.items():
                _append_metric(
                    scale_aggregates,
                    scale_idx,
                    metrics,
                )

            bpp_list = [
                f"{metrics['bpp']:.8f}"
                for metrics in scale_metrics.values()
            ]
            print(
                f"    Finished {img_name}, "
                f"Scales total BPP: {bpp_list}"
            )

            _write_partial_summary(
                partial_json_path,
                model_name,
                scale_aggregates,
                image_idx + 1,
            )

    average_metrics_summary = {}

    print("\n" + "=" * 100)
    print(
        f"[*] model {model_name} "
        "BPP / LPIPS / DISTS average results"
    )
    print("=" * 100)

    for scale_idx in sorted(
        scale_aggregates.keys(),
        key=lambda x: int(x),
    ):
        vals = scale_aggregates[scale_idx]

        avg_bpp = float(np.mean(vals["bpp"]))
        avg_img_bpp = float(
            np.mean(vals["img_bpp"])
        )
        avg_text_bpp = float(
            np.mean(vals["text_bpp"])
        )
        avg_text_payload_bpp = float(
            np.mean(vals["text_payload_bpp"])
        )
        avg_text_side_bpp = float(
            np.mean(vals["text_side_bpp"])
        )

        avg_lpips = float(
            np.mean(vals["lpips"])
        )
        avg_dists = float(
            np.mean(vals["dists"])
        )
        average_metrics_summary[scale_idx] = {
            "avg_bpp": avg_bpp,
            "avg_total_bpp": avg_bpp,
            "avg_img_bpp": avg_img_bpp,
            "avg_text_bpp": avg_text_bpp,
            "avg_text_payload_bpp": (
                avg_text_payload_bpp
            ),
            "avg_text_side_bpp": avg_text_side_bpp,

            "avg_lpips": avg_lpips,
            "avg_dists": avg_dists,

            "image_count": len(vals["bpp"]),
        }

        print(
            f"Scale {int(scale_idx):02d} | "
            f"BPP {avg_bpp:.8f} | "
            f"LPIPS {avg_lpips:.6f} | "
            f"DISTS {avg_dists:.6f}"
        )

    final_summary = {
        "status": "complete",
        "model_name": model_name,
        "model_path": cli.infinity_ckpt.name,
        "vae_path": cli.vae_ckpt.name,
        "dataset_jsonl": cli.dataset_jsonl.name,
        "image_count": len(json_data),

        "metrics": ["LPIPS", "DISTS"],
        "metric_protocol": {
            "lpips": (
                "lpips.LPIPS(net='alex'); "
                "GT/recon scaled from [0,1] to [-1,1]"
            ),
            "dists": (
                "DISTS_pytorch.DISTS; "
                "GT/recon in [0,1]"
            ),
        },

        "text_codec": "adaptive_arithmetic",
        "bitrate_accounting": {
            "total_bpp": "img_bpp + text_bpp",
            "img_bpp": (
                "existing visual payload bits + per-token "
                "hbits mode flags from sender.py"
            ),
            "text_bpp": (
                "adaptive-arithmetic byte-padded payload + "
                "32-bit text payload-length field; "
                "no per-caption CDF"
            ),
            "note": (
                "visual Python-list framing/boundary overhead "
                "is NOT added in this file"
            ),
        },

        "details_file": Path(details_jsonl_path).name,
        "summary": average_metrics_summary,
    }

    with open(
        summary_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    df_summary = pd.DataFrame(
        average_metrics_summary
    ).T
    df_summary.index.name = "scale"
    df_summary.to_csv(summary_csv_path)

    if os.path.exists(partial_json_path):
        os.remove(partial_json_path)

    print("\n[*] Detailed per-image results:")
    print(f"    {details_jsonl_path}")
    print("[*] Final summary JSON:")
    print(f"    {summary_json_path}")
    print("[*] Final summary CSV:")
    print(f"    {summary_csv_path}")


    print("\n[*] Evaluation complete.")
