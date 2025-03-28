# Diss Benchmarks
A small set of python scripts that run some ML models so that I can measure their power consumption.
Python package requirements in `requirements.txt`

Warning! There might be some setup of CUDA and Nvidia and Tensorrt needed outside of PIP that can be a bit buggy!!!

# Models
All models are from HuggingFace. I chose these models as they were small and fit onto the 6GB of VRAM that the GPU had.
If repeating these experiments and you have a more powerful GPU, use bigger models. Should be as simple as changing the hugging face directory for each model.

1. Image Classification (ResNet50)
2. Image Generation (Stable Diffusion)
3. Text Generation (Llama-3.201B)

# How it works
Prompts from datasets from hugging face are passed into the models to perform inference until a time limit is reached.
