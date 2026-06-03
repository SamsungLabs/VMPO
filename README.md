<h1 align="center">VMPO: Diffusion Alignment with Variance Minimisation</h1>

This repo contains PyTorch implementation of the paper "[Diffusion Alignment Beyond KL: Variance Minimisation as Effective Policy Optimiser](https://arxiv.org/abs/2602.12229)" by [Zijing Ou](https://j-zin.github.io/), [Jacob Si](https://jacobyhsi.github.io/), [Junyi Zhu](https://junyizhu-ai.github.io/), [Ondrej Bohdal](https://ondrejbohdal.github.io/), [Mete Ozay](https://dblp.org/pid/26/5515.html), [Taha Ceritli](https://tahaceritli.github.io/), and [Yingzhen Li](http://yingzhenli.net/home/en/).

We propose Variance Minimisation Policy Optimiser (VMPO), a framework that reformulates diffusion alignment through the lens of Sequential Monte Carlo (SMC) sampling. VMPO treats the denoising process as a proposal distribution and reward guidance as importance weights, shifting the optimization goal from standard KL-divergence to the minimization of log-importance weight variance. We theoretically demonstrate that this variance objective is minimized by the optimal reward-tilted target and that its gradient aligns with KL-based methods under on-policy sampling. By varying potential functions and minimization strategies, VMPO provides a unified perspective that recovers several existing alignment methods while opening new design paths for non-KL-based diffusion refinement.


## Environment Setup
Our implementation is based on the [Flow-GRPO](https://github.com/yifan123/flow_grpo) and the [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) codebase, with most environments aligned.

Clone this repository and install packages by:
```bash
# Environment setup
conda create -n DiffusionNFT python=3.10.16
conda install -c nvidia cuda-toolkit=12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e . --no-build-isolation

# OCR reward preparation
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein
```

## Dataset

The repo currently supports the OCR task from [Flow-GRPO](https://github.com/yifan123/flow_grpo). The training and testing data can be downloaded from [Flow-GRPO](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr) and should be placed within `dataset/ocr` directory.

## Training
The default configuration file `config/vmpo.py` is set for 4 GPUs, and you can customize it as needed.
```bash
torchrun --nproc_per_node=4 -m scripts.train_vmpo_sd3 --config config/vmpo.py:sd3_ocr
```

## Citation
If you find this repo useful, please consider citing our paper:
```
@article{ou2026diffusion,
  title={Diffusion Alignment Beyond KL: Variance Minimisation as Effective Policy Optimiser},
  author={Ou, Zijing and Si, Jacob and Zhu, Junyi and Bohdal, Ondrej and Ozay, Mete and Ceritli, Taha and Li, Yingzhen},
  journal={arXiv preprint arXiv:2602.12229},
  year={2026}
}
```