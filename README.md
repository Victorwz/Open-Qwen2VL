# Open-Qwen2VL

[![arXiv](https://img.shields.io/badge/arXiv-2504.00595-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2504.00595)

Official code repo for our work [Open-Qwen2VL: Compute-Efficient Pre-Training of Fully-Open Multimodal LLMs on Academic Resources](https://victorwz.github.io/Open-Qwen2VL/).

## Introduction
This repo supports:
- Data quality score generation with DFN/CLIP and [MLM-Filter](https://github.com/Victorwz/MLM_Filter)
- High quality data selection based on the quality scores and resharding into webdataset format
- Multimodal Sequence Packing towards large-scale image-text dataset in webdataset format (supporting both caption data and interleaved data)
- Pre-training with packed multimodal sequences
- Supversied fine-tuning on both small-scale SFT data like [LLaVA-665k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) and large-scale SFT data like [MAmmoTH-VL-10M](https://huggingface.co/datasets/MAmmoTH-VL/MAmmoTH-VL-Instruct-12M)
- Evaluation on a series of multimodal benchmarks


## Release
- [3/31/2025] 🔥 We released all pre-trained model and instruction-tuned model checkpoints at [Open-Qwen2VL](https://huggingface.co/weizhiwang/Open-Qwen2VL) and [Open-Qwen2VL-Base](https://huggingface.co/weizhiwang/Open-Qwen2VL-Base)
- [3/31/2025] 🔥 We released all pre-training data in webdataset format at [Open-Qwen2VL-Data](https://huggingface.co/datasets/weizhiwang/Open-Qwen2VL-Data).
- [3/31/2025] 🔥 We released the technical report for [**Open-Qwen2VL**](https://arxiv.org/abs/2504.00595).

## Install

```Shell
conda create -n openqwen2vl python=3.10
conda activate openqwen2vl
pip install -e prismatic-vlms
```

If you need to pre-train or SFT the MLLM, install flash-attention:
```
pip install flash-attn==2.5.2 --no-build-isolation
```

## Directly Use or Trial
```python
import requests
import torch
from PIL import Image
from prismatic import load

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub)
vlm = load("Open-Qwen2VL")
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
image_url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image = [vlm.vision_backbone.image_transform(Image.open(requests.get(image_url, stream=True).raw).convert("RGB")).unsqueeze(0)]
user_prompt = '<image>\nDescribe the image."

# Generate!
generated_text = vlm.generate_batch(
    image,
    [user_prompt],
    do_sample=False,
    max_new_tokens=512,
    min_length=1,
)
print(generated_text[0])
```

## Multimodal Sequence Packing
We have released all our pre-training image-text caption data in webdataset format at [Open-Qwen2VL-Data](https://huggingface.co/datasets/weizhiwang/Open-Qwen2VL-Data). Please download it with ```huggingface-cli download``` or directly ```git clone```.

Then please run
```shell
bash mm_sequence_packing/multiprocess_sequence_packing_image_to_pil.sh 0 4 504 datacomp
bash mm_sequence_packing/multiprocess_sequence_packing_image_to_pil.sh 0 4 326 ccs
```

[Update on 5/16/2025]
We support sequence packing into a single json file which contains base64-string images and texts. Storing in json files can save 95% more disk space. You can run
```shell
bash mm_sequence_packing/multiprocess_sequence_packing_image_to_json.sh 0 4 504 /path/to/datacomp_wds /path/to/datacomp_packed_files
```

## Compute-Efficient MLLM Pre-Training
Prior to training, please follow the sequence packing instructions in the README to prepare the pickle files for each subdataset.

Then please run the training script
```Shell
bash prismatic-vlms/train.sh ${CKPTID} ${STAGE} ${BSZ} ${PER_GPU_BSZ}
```
Here are the parameters for training:
- `CKPTID`: id for the saved checkpoint;
- `STAGE`: choose between `pretrain` and `full-pretrain`, in which the full-pretrain will make the vision encoder trainable as well;
- `BSZ`: global batch size;
- `PER_GPU_BSZ`: the batch size for each gpu. If the global_bsz != num_gpus * per_gpu_bsz, then the gradient accumulation will be applied.

## Visual SFT
### Large-Scale SFT on MammoTH-VL-10M
Please firstly download and unzip the images of [MAmmoTH-VL-10M](https://huggingface.co/datasets/MAmmoTH-VL/MAmmoTH-VL-Instruct-12M). Then run ```python data_prepare/split_mammoth_10m.py``` to split each instruction examples into single json files.

Please run the training script
```Shell
bash prismatic-vlms/fine_tune_mammoth.sh ${CKPT_PATH} ${CKPTID}
```
Here are the parameters for training:
- `CKPT_PATH`: the path to the pre-trained MLLM checkpoint after the pre-training stage;
- `CKPTID`: id for the saved checkpoint

### Normal SFT Scripts

Then please run the training script
```Shell
bash prismatic-vlms/fine_tune.sh ${CKPT_PATH} ${CKPTID} ${DATAPATH}
```
Here are the parameters for training:
- `CKPT_PATH`: the path to the pre-trained MLLM checkpoint after the pre-training stage;
- `CKPTID`: id for the saved checkpoint;
- `DATAPATH`: the path to the SFT dataset json file.

## Evaluations
Please follow the doc [docs/Eval.md](docs/Eval.md) for evaluation instructions.

## Data Quality Score Generation and Data Filtering
Please follow the doc [docs/DATA_Filter.md](docs/DATA_Filter.md) for data filtering instructions.

## Citation

Please cite our paper if you find this repository interesting or helpful:
```bibtex
@article{Open-Qwen2VL,
  title={Open-Qwen2VL: Compute-Efficient Pre-Training of Fully-Open Multimodal LLMs on Academic Resources},
  author={Wang, Weizhi and Tian, Yu and Yang, Linjie and Wang, Heng and Yan, Xifeng},
  journal={arXiv preprint arXiv:2504.00595},
  year={2025}
}
```


## Credits
Our codebase is developed based on [prismatic-vlms](https://github.com/TRI-ML/prismatic-vlms) and [vlm-evaluation](https://github.com/TRI-ML/vlm-evaluation).
