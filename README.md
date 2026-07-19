<div align="center">

<h1>GeoStereo</h1>

<h3>A Unified Stereo Geometry Estimation Framework for Disparity and Surface Normal</h3>

<p>
  Qizhe Wei, Xianda Guo, Shaocong Xu, Hong Li, Runyi Yang, Hao Zhao
</p>

<a href="https://qz-wei.github.io/GeoStereo.github.io/">
  <img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page">
</a>
<a href="https://huggingface.co/qz-wei/GeoStereo">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_Weights-yellow" alt="Hugging Face">
</a>

</div>

---

<p align="center">
  <img src="assets/teaser.png" alt="GeoStereo Teaser" width="95%">
</p>

<p align="center">
  <img src="assets/background.gif" alt="GeoStereo Demo" width="95%">
</p>

---

## Overview

**GeoStereo** is a unified stereo geometry estimation framework for disparity and surface normal prediction. Given a rectified stereo pair, GeoStereo first estimates dense binocular correspondence and converts the predicted disparity into an initial surface-normal prior. Meanwhile, the right image is warped into the left camera view to provide explicit stereo appearance and correspondence cues.

These complementary geometric signals are encoded in the latent space and injected into a Lotus-style diffusion UNet through two dedicated conditioning branches:

- a **warped-view ControlNet**, which provides appearance and correspondence information from the right image warped to the left view;
- a **normal-prior ControlNet**, which provides geometry-aware initialization derived from stereo disparity.

The diffusion model combines stereo geometry with strong generative visual priors to predict spatially coherent surface normals while preserving scene structure and object boundaries. The unified design also enables disparity and surface-normal estimation within the same stereo geometry framework.

<p align="center">
  <img src="assets/pipe.png" alt="GeoStereo Pipeline" width="95%">
</p>

### Highlights

- **Unified stereo geometry estimation:** supports both disparity and surface-normal prediction.
- **Stereo-guided geometry prior:** converts dense stereo correspondence into an explicit surface-normal initialization.
- **Dual conditioning branches:** separately injects warped-view appearance cues and stereo-derived normal cues.
- **Diffusion-based refinement:** combines geometric constraints with strong generative priors for robust normal prediction.
- **Joint supervision:** optimizes latent-space reconstruction together with decoded normal-direction consistency.
- **Complete workflow:** provides scripts for training, inference, and evaluation.
- **Public checkpoints:** pretrained model weights are hosted on [Hugging Face](https://huggingface.co/qz-wei/GeoStereo).

---

## Installation

The environment was tested with:

- Python 3.11
- PyTorch 2.4.1
- CUDA 12.1
- Linux

### 1. Clone the repository

```bash
git clone https://github.com/qz-wei/GeoStereo.git
cd GeoStereo
```

### 2. Create a Conda environment

```bash
conda create -n geostereo python=3.11 -y
conda activate geostereo
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `flash-attn` fails to build during installation, install it separately without build isolation:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

---

## Pretrained Weights

The pretrained GeoStereo checkpoints are available on Hugging Face:

- [qz-wei/GeoStereo](https://huggingface.co/qz-wei/GeoStereo)

Download the complete repository using the Hugging Face CLI:

```bash
mkdir -p checkpoints/GeoStereo

hf download qz-wei/GeoStereo \
  --local-dir checkpoints/GeoStereo
```

Alternatively, download the weights with Git LFS:

```bash
git lfs install

git clone https://huggingface.co/qz-wei/GeoStereo \
  checkpoints/GeoStereo
```

The downloaded checkpoint is expected to contain the main diffusion and conditioning components, for example:

```text
checkpoints/GeoStereo/
├── unet/
├── controlnet/
├── normal_controlnet/
├── vae/
├── text_encoder/
├── tokenizer/
└── scheduler/
```

---

## Inference

Run inference on rectified stereo image pairs with:

```bash
python infer.py \
  --pair_txt ./left_right.txt
```

The input text file should contain the paths of the left and right images. Before running inference, update the checkpoint paths and output directory according to your local environment.

A typical inference configuration should specify:

- the left and right image paths;
- the GeoStereo checkpoint;
- the FoundationStereo checkpoint and configuration;
- the output directory.

Use the following command to inspect all supported options:

```bash
python infer.py --help
```

---

## Evaluation

Run evaluation with:

```bash
python eval.py
```

Before evaluation, update the dataset paths, checkpoint paths, and evaluation configuration according to your local directory structure.

Use the following command to inspect all supported options:

```bash
python eval.py --help
```

---

## Training

GeoStereo supports distributed training through Hugging Face Accelerate. The following example launches training on four GPUs:

```bash
accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision no \
  --dynamo_backend no \
  train.py \
  --vis_step 50
```

Important training arguments include:

```text
--model_path                 Base Lotus diffusion checkpoint
--restore_ckpt               GeoStereo checkpoint used for initialization
--ckpt_dir                   FoundationStereo checkpoint
--cfg_yaml                   FoundationStereo configuration file
--output_dir                 Directory for model checkpoints
--logging_dir                Directory for TensorBoard logs
--train_batch_size           Batch size per GPU
--max_train_steps            Number of optimization steps
--learning_rate              Peak learning rate
--lr_total_iter_length       Learning-rate schedule length
--lambda_ang                 Weight of decoded normal cosine loss
--checkpointing_steps        Checkpoint saving interval
--vis_step                   TensorBoard visualization interval
```

Use the following command to inspect all supported options:

```bash
python train.py --help
```

Monitor training with TensorBoard:

```bash
tensorboard \
  --logdir outputs/logs \
  --port 6006
```

---

## Project Page

More qualitative results, visual comparisons, and method details are available on the project page:

- [GeoStereo Project Page](https://qz-wei.github.io/GeoStereo.github.io/)

---

## Acknowledgements

GeoStereo is built upon several excellent open-source projects, including:

- [FoundationStereo](https://github.com/NVlabs/FoundationStereo)
- [Diffusers](https://github.com/huggingface/diffusers)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [StableNormal](https://github.com/Stable-X/StableNormal)

We sincerely thank the authors and contributors for making their work publicly available.