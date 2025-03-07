#!/usr/bin/env python
import numpy as np
from PIL import Image

import torch
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset
import time

# Name of the HF model to load
MODEL_NAME = "microsoft/resnet-50"

DATASET_NAME = "zh-plus/tiny-imagenet"  # Example image dataset

RUN_TIME = 0.5 # minutes

if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Please check your PyTorch installation.")

# 1) Load model & feature extractor from Hugging Face
print(f"Loading model and feature extractor from '{MODEL_NAME}'...")
      
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = ResNetForImageClassification.from_pretrained(MODEL_NAME).to("cuda").eval()

# 2) Collect images either from a local folder or from a Hugging Face dataset
print("Collecting images...")
images = []
ds = load_dataset(DATASET_NAME, split="train")
for sample in ds:
    img = sample["image"]
    # If it's a NumPy array, convert to PIL
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.uint8(img))
    if len(np.array(img).shape) == 2:
        img = img.convert("RGB")
    images.append(img)

print(f"Dataset has {len(images)} images")
has_labels = hasattr(model.config, "id2label")
# 3) Run inference on each image using plain PyTorch
print("Running inference...")
start = time.time()
end = start + (RUN_TIME * 60)  # Run for 10 seconds
for idx, img in enumerate(images):
    inputs = feature_extractor(img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to("cuda")  # shape [1, 3, H, W]

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    # Get logits and predicted class
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()
    
    # Display results
    if has_labels:
        class_name = model.config.id2label[predicted_class_idx]
        print(f"Image {idx}: Predicted '{class_name}' (class {predicted_class_idx}) with confidence: {confidence:.4f}")
    else:
        print(f"Image {idx}: Predicted class {predicted_class_idx} with confidence: {confidence:.4f}")

    if time.time() > end:
        print("Time's up!")
        break