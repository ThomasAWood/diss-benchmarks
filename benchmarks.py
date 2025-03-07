import torch
import torch_tensorrt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoFeatureExtractor, ResNetForImageClassification
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
import time
import numpy as np
from PIL import Image

if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Please check your PyTorch installation.")

class TextGenBenchmark:
    def prepare_data(self):
        print("    Preparing data...")
        dataset_name = "fka/awesome-chatgpt-prompts"
        dataset = load_dataset(dataset_name, split="train")
        prompts = []
        for example in dataset:
            text = example["prompt"]
            # Use a minimal slice of the text as the prompt
            prompt = text[:1024]  # truncating just to demonstrate
            prompts.append(prompt)
        
        self.prompts = prompts

    def load_model(self):
        print("    Loading model...")
        # Load tokenizer and model in float16 for GPU
        model_name = "meta-llama/Llama-3.2-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to("cuda")

        # Convert PyTorch model to TensorRT-optimized module
        self.model = torch.compile(
            model,
            backend="tensorrt",
        ).to("cuda").eval()
    
    def run(self, run_time):
        start = time.time()
        end = start + (run_time * 60)  # Run for 60 seconds
        for i, prompt in enumerate(self.prompts):
            print(f"    Processing prompt {i}...")
            inputs = self.tokenizer(
                prompt, 
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    )

            # Print the generated text
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # print("Prompt:")
            # print(prompt)
            # print('Response')
            # print(generated_text[len(prompt):])

            if time.time() > end:
                break

class ImageGenBenchmark:
    def prepare_data(self):
        print("    Preparing data...")
        dataset_name = "Gustavosta/Stable-Diffusion-Prompts"
        dataset = load_dataset(dataset_name, split="test[:300]")
        prompts = []
        for example in dataset:
            prompt = example["Prompt"]
            prompts.append(prompt)

        self.prompts = prompts

    def load_model(self):
        print("    Loading model...")
        model = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, framework="pt")
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        self.pipe = pipe

    def run(self, run_time):
        start = time.time()
        end = start + (run_time * 60)
        for idx, prompt in enumerate(self.prompts):
            print(f"    Processing prompt {idx}...")
            self.pipe(prompt).images

            if time.time() > end:
                break

class ImageClassificationBenchmark:
    def prepare_data(self):
        print("    Preparing data...")
        dataset_name = "zh-plus/tiny-imagenet"
        images = []
        ds = load_dataset(dataset_name, split="train")
        for sample in ds:
            img = sample["image"]
            # If it's a NumPy array, convert to PIL
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.uint8(img))
            if len(np.array(img).shape) == 2:
                img = img.convert("RGB")
            images.append(img)

        self.images = images

    def load_model(self):
        print("    Loading model...")
        model_name = "microsoft/resnet-50" 
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = ResNetForImageClassification.from_pretrained(model_name).to("cuda").eval()

    def run(self, run_time):
        has_labels = hasattr(self.model.config, "id2label")
        start = time.time()
        end = start + (run_time * 60)

        for idx, img in enumerate(self.images):
            print(f"    Processing image {idx}...")
            inputs = self.feature_extractor(img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to("cuda")  # shape [1, 3, H, W]

            # Forward pass
            with torch.no_grad():
                outputs = self.model(pixel_values)

            # Get logits and predicted class
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()
            
            # Display results
            # if has_labels:
            #     class_name = self.model.config.id2label[predicted_class_idx]
            #     # print(f"Image {idx}: Predicted '{class_name}' (class {predicted_class_idx}) with confidence: {confidence:.4f}")
            # else:
            #     # print(f"Image {idx}: Predicted class {predicted_class_idx} with confidence: {confidence:.4f}")

            if time.time() > end:
                break 

class Benchmarks:
    def load_benchmark(self, benchmark):
        if benchmark == "text_gen":
            print("Loading Text Gen Benchmark...")
            self.text_gen_benchmark = TextGenBenchmark()
            self.text_gen_benchmark.prepare_data()
            self.text_gen_benchmark.load_model()
        elif benchmark == "image_gen":
            print("Loading Image Gen Benchmark...")
            self.image_gen_benchmark = ImageGenBenchmark()
            self.image_gen_benchmark.prepare_data()
            self.image_gen_benchmark.load_model()
        elif benchmark == "image_classification":
            print("Loading Image Classification Benchmark...")
            self.image_classification_benchmark = ImageClassificationBenchmark()
            self.image_classification_benchmark.prepare_data()
            self.image_classification_benchmark.load_model()
        print("Benchmark Loaded!")

    def run(self, benchmark, run_time):
        if benchmark == "text_gen":
            self.text_gen_benchmark.run(run_time)
        elif benchmark == "image_gen":
            self.image_gen_benchmark.run(run_time)
        elif benchmark == "image_classification":
            self.image_classification_benchmark.run(run_time)
        else:
            raise Exception("Invalid benchmark name. Please choose from 'text_gen', 'image_gen', 'image_classification'.")

    def free_memory(self, benchmark):
        if benchmark == "text_gen":
            del self.text_gen_benchmark
        elif benchmark == "image_gen":
            del self.image_gen_benchmark
        elif benchmark == "image_classification":
            del self.image_classification_benchmark
        torch.cuda.empty_cache()

    def load_all_benchmarks(self):
        self.load_benchmark("text_gen")
        self.load_benchmark("image_gen")
        self.load_benchmark("image_classification")
