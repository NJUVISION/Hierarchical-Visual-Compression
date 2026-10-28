
<h1 align="center">Hierarchical Visual Description Learning for Ultra-low Rate Progressive Image Compression</h1>
<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11749545">📄 Paper (IEEE Xplore)</a>
</p>
<p align="center">
  Hanfei&nbsp;Li &ensp; <b>&middot;</b> &ensp;
  Anle&nbsp;Ke &ensp; <b>&middot;</b> &ensp;
  Junqi&nbsp;Shi &ensp; <b>&middot;</b> &ensp;
  Tong&nbsp;Chen &ensp; <b>&middot;</b> &ensp;
  Zhan&nbsp;Ma
</p>

<p align="center">
  Nanjing University
</p>


<p align="left">
  <strong>We propose a VAR-based framework for ultra-low-rate progressive image compression with enhanced semantic preservation and reconstruction quality.</strong>
</p>


<p align="center">
  <img src="./figs/framework.png" width="100%" alt="Overview of the proposed framework">
</p>

## 📢 News
- **2026.10.25** — Our paper is now available on [IEEE Xplore](https://ieeexplore.ieee.org/document/11749545) (Early Access)!
- **2026.10.14** — Our paper has been accepted by **IEEE JETCAS**!


## 📊 Quantitative Results

We evaluate our method on DIV2K and CLIC in terms of LPIPS, DISTS, FID, CLIP-Sem, and DINO-Sem across different bitrates.

<p align="center">
  <img src="./figs/rd.png" width="100%" alt="Quantitative results on DIV2K and CLIC">
</p>

## 🖼️ Visual Comparisons

Visual comparisons at extremely low bitrates. Our method preserves plausible visual structures and semantic content at compression ratios up to 76,000×.

<p align="center">
  <img src="./figs/visual.png" width="100%" alt="Visual comparisons at ultra-low bitrates">
</p>

## 🚀 Inference and Evaluation

### Pretrained Models

Our pretrained checkpoints are available on [ModelScope](https://modelscope.cn/models/LiHanfei/Hierarchical-Visual-Compression/files).

The following models are required for inference:

| Model | Checkpoint | Source |
|---|---|---|
| Stage-1 BSQ-VAE | `stage1_bsqvae.ckpt` | ModelScope |
| Stage-2 Infinity-2B | `stage2_infinity.pth` | ModelScope |
| Flan-T5-XL | `google/flan-t5-xl` | Hugging Face |

Download our checkpoints using ModelScope:

```bash
pip install modelscope

modelscope download \
    --model LiHanfei/Hierarchical-Visual-Compression \
    stage1_bsqvae.ckpt stage2_infinity.pth \
    --local_dir ./checkpoints
```

Download Flan-T5-XL separately:

```bash
hf download google/flan-t5-xl \
    --local-dir ./checkpoints/flan-t5-xl
```

### Environment Setup

Our inference environment is based on Python 3.10.19, PyTorch 2.8.0, and CUDA 12.8. Experiments were conducted on NVIDIA A100 GPUs.

```bash
conda create -n hvc python=3.10 -y
conda activate hvc

pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install flash-attn==2.8.2 --no-build-isolation
```

### Dataset Preparation

We evaluate our method on DIV2K and CLIC. All images are preprocessed to **1024 × 1024** by resizing the shorter side to 1024 while preserving the aspect ratio, followed by a center crop.

Image captions are generated using **BLIP-2**. We provide `dataset/DIV2K.jsonl` and `dataset/CLIC.jsonl`, containing image filenames and their corresponding captions.

Please specify the directory of preprocessed images using `--image-root`.

### Inference

Run inference and evaluate BPP, LPIPS, and DISTS on one image:

```bash
python eval.py \
    --vae-ckpt ./checkpoints/stage1_bsqvae.ckpt \
    --infinity-ckpt ./checkpoints/stage2_infinity.pth \
    --t5-path ./checkpoints/flan-t5-xl \
    --dataset-jsonl ./dataset/DIV2K.jsonl \
    --image-root /path/to/DIV2K/images \
    --output-dir ./results/DIV2K \
    --pn 1M \
    --max-images 1
```

Set `--max-images 0` to evaluate the entire dataset.

The script performs progressive compression and reconstruction across 13 scales and reports the corresponding bitrate and perceptual quality.

The reconstructed images and evaluation results are saved under the specified output directory.


## 🌊 TODO
- [x] Release quantitative results
- [x] Release inference code
- [x] Release pretrained models
- [ ] Release training code

## ❤ Acknowledgement

This work builds upon [Infinity](https://github.com/FoundationVision/Infinity). We thank the authors for their valuable contributions.

## 🙇 Citation

If you find our work useful, please consider citing:

```bibtex
@article{Li2026Hierarchical,
    author  = {Li, Hanfei and Ke, Anle and Shi, Junqi and Chen, Tong and Ma, Zhan},
    title   = {Hierarchical Visual Description Learning for Ultra-low Rate Progressive Image Compression},
    journal = {IEEE Journal on Emerging and Selected Topics in Circuits and Systems},
    year    = {2026},
    doi     = {10.1109/JETCAS.2026.3746235},
    url     = {https://doi.org/10.1109/JETCAS.2026.3746235}
}
```
