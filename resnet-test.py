#!/usr/bin/env python
import numpy as np
from PIL import Image

import torch
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset

# Name of the HF model to load
MODEL_NAME = "microsoft/resnet-50"

DATASET_NAME = "zh-plus/tiny-imagenet"  # Example image dataset
DATASET_SPLIT = "train[:5]"  # Subset for demonstration


def main():
    # 1) Load model & feature extractor from Hugging Face
    print(f"Loading model and feature extractor from '{MODEL_NAME}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ResNetForImageClassification.from_pretrained(MODEL_NAME).to(device).eval()

    # 2) Collect images either from a local folder or from a Hugging Face dataset
    print("Collecting images...")
    images = []
    ds = load_dataset(DATASET_NAME, split="valid")
    for sample in ds:
        img = sample["image"]
        # If it's a NumPy array, convert to PIL
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.uint8(img))
        images.append(img)

    print(f"Dataset has {len(images)} images")

    # 3) Run inference on each image using plain PyTorch
    print("Running inference...")
    for idx, img in enumerate(images):
        # Preprocess with feature extractor → torch.FloatTensor
        
        # If image only has 2 dimensions (Black and White), convert to 3 dimensions
        if len(np.array(img).shape) == 2:
            img = img.convert("RGB")

        inputs = feature_extractor(img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)  # shape [1, 3, H, W]

        # Forward pass
        with torch.no_grad():
            outputs = model(pixel_values)
        # logits = outputs.logits  # shape [1, num_labels]

        print(f"Image {idx}")


if __name__ == "__main__":
    main()
