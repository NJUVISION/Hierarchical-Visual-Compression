<!-- ## Hierarchical Visual Description Learning for Ultra-low Rate Progressive Image Compression -->

<h1 align="center"> Hierarchical Visual Description Learning for Ultra-low Rate Progressive Image Compression </h1>

<p align="center">
  Hanfei&nbsp;Li &ensp; <b>&middot;</b> &ensp; 
  Anle&nbsp;Ke &ensp; <b>&middot;</b> &ensp; 
  Junqi&nbsp;Shi &ensp; <b>&middot;</b> &ensp; 
  Tong&nbsp;Chen&ensp; <b>&middot;</b> &ensp; 
  Zhan&nbsp;Ma&ensp; <b>&middot;</b> &ensp; 
</p>

<p align="center">
  Nanjing University &emsp;
</p>

<!-- <p align="center">
  <a href="https://njuvision.github.io/ResULIC/">🌐 Project Page</a> &ensp;
  <a href="https://arxiv.org/abs/2505.08281">📃 Paper</a> &ensp;
</p> -->

<p align="center">
    <img src="./figs/framework.png" style="border-radius: 15px"><br>
</p>


## 📢 News
- **2026.10.14** — Our paper has been accepted by **IEEE JETCAS**!



<!--
## ⚙️ Environment Setup

```bash
- conda create -n ResULIC python=3.10
- conda activate ResULIC
- pip install -r requirements.txt
```-->

## <a name="Visual Results"></a>✨ Visual Results
<p align="center">
    <img src="./figs/visual.png" style="border-radius: 15px"><br>
</p>

<!--
## <a name="train"></a>⏳ Train

**Note:** The numbers in the yaml filenames (e.g., `1_1_1`) represent $\lambda_{\text{diffusion}}$, $\lambda_{\text{mse}}$, and $\lambda_{\text{bpp}}$ respectively.

#### Stage 1: Initial Training

1. **Download Pretrained Model**  
   Download the pretrained **Stable Diffusion v2.1** model into the `./weight` directory:
   ```bash
   wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate -P ./weight

2. **Modify the configuration file**`./configs/train_zc_eps.yaml` and `./configs/model/stage1/xx.yaml` accordingly.

3. Start training.
   ```
   bash stage1.sh 
   ```
#### Stage 2: 

1. Modify the configuration file `./configs/train_stage2.yaml` and `./configs/model/stage2/xx.yaml` accordingly.

2. Start training.
   ```
   bash stage2.py 
   ```
-->

<!--
## <a name="inference"></a>😀 Inference
 -->




## 🌊 TODO
- [x] Release quantitative results
- [ ] Release inference code & pretrained models
- [ ] Release training code

## <a name="acknowledgement">❤ Acknowledgement
This work is based on [Infinity]([https://github.com/lllyasviel/ControlNet](https://github.com/FoundationVision/Infinity)), thanks to this invaluable contributions.

## <a name="cite"></a>🙇‍ Citation


If you find our work useful, please consider citing:

```bibtex
@article{Li2026Hierarchical,
    author  = {Li, Hanfei and Ke, Anle and Shi, Junqi and Chen, Tong and Ma, Zhan},
    title   = {Hierarchical Visual Description Learning for Ultra-low Rate Progressive Image Compression},
    journal = {IEEE Journal on Emerging and Selected Topics in Circuits and Systems},
    year    = {2026}
}
```
