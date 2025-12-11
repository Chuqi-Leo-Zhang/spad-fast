SPAD-FAST : Fast Multi-View Diffusion via Geometry-Aware Consistency Distillation
===================================================
<h4>
Chuqi Zhang, Binghong Chen, Jie Wu
</br>
<span style="font-size: 14pt; color: #555555">
</span>
</h4>
<hr>

## Acknowledgement

We have intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.

- [SPAD: https://github.com/yashkant/spad](https://github.com/yashkant/spad)
- [LCM-LoRA: https://github.com/luosiallen/latent-consistency-model](https://github.com/luosiallen/latent-consistency-model)

## Main Contribution

(a) We distill SPAD into a Latency Consistency Model by LCM-LoRA. 
(b) To maintain multi-view consistency, we introduce geometry-aware distillation losses.
(c) We achieve 12x speed-up at the inference stage comparing to the original SPAD while maintain comparable visual quality and consistency.

## Repository Setup

Create a fresh conda environment, and install all dependencies.

```text
conda create -n spad python=3.8 -y
conda activate spad
```

Clone the repository. Then, install pytorch (tested with CUDA 11.8), dependencies, pytorch3d, taming-transformers, and ldm:
```
git clone https://github.com/yashkant/spad
cd spad

# get pytorch (select correct CUDA version)
pip install --ignore-installed torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# get dependencies
pip install -r requirements.txt

# get and install pytorch3d (from source)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# get and install taming-transformers (from source)
git clone git@github.com:CompVis/taming-transformers.git
cd taming-transformers && pip install -e . && cd ..

# install ldm (from spad)
pip install -e .
```
If you run into dependency mismatch issues, take a look at this issue: https://github.com/yashkant/spad/issues/13#issuecomment-2640721687

## Pretrained Model

We utilize the checkpoints provided by [SPAD](https://github.com/yashkant/spad). To download it, run this command:
```
python scripts/download.py
```

## Distillation & Inference

To get dataset for training, follow this command:
```
bash ./render_data/render_batch.py
```
This command is actually just for testing. Change the code of line 44 in ./render_data/render_batch.py to get all the dataset.

To start LCM-LoRA distillation for SPAD, follow this command:
```
bash ./run_train.sh
```
To test and eveluate the distilled model, follow this command:
```
bash ./run_inference.sh
```
We use same example caption following [SPAD](https://github.com/yashkant/spad) for easier camparisons. Some settings might be unreasonable due to our limited computation resources, feel free to change that.

