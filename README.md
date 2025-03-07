# Diss Benchmarks
A small set of python scripts that run some ML models so that I can measure their power consumption.

To enable the python environment:
`conda activate trt`

# Models
All models and datasets are from HuggingFace

## Resnet-50 (Image Classification)
Model: `microsoft/resnet-50`
Dataset: `zh-plus/tiny-imagenet`
Dataset split:
train (100k images)
valid (10k images)

Can change which one it uses in the code (and truncate if needed)

## {Choose a text generation model} (Text Generation)
Model: `` (find the largest model that can be run on our device)
Dataset: {to choose}

##  RMBG 2.0 (Background Removal) (Image Segmentation)
Model: `briaai/RMBG-2.0`
Dataset: {to choose}

## Stable Diffusion (Text to Image) (Image Generation)
Model: `` (find a stable diffusion model that's small enough)
Dataset: {to choose}
