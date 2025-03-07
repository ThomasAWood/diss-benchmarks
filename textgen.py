import torch
import torch_tensorrt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time

if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Please check your PyTorch installation.")

#
# Create the dataset
#
print("Downloading dataset...")
dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train")

RUN_TIME = 1  # Run for 1 minute


print("Creating promt set...")
prompts = []
for example in dataset:
    text = example["prompt"]
    # Use a minimal slice of the text as the prompt
    prompt = text[:1024]  # truncating just to demonstrate
    prompts.append(prompt)

max_seq_length = 1024

#
# Load the model
# 
model_name = "meta-llama/Llama-3.2-1B"

print("Loading Tokenizer and Model...")
# Load tokenizer and model in float16 for GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to("cuda")

print("Converting model to TensorRT...")
# Convert PyTorch model to TensorRT-optimized module
model_trt = torch.compile(
    model,
    backend="tensorrt",
).to("cuda").eval()

#
# Run inference
# 
start = time.time()
end = start + (RUN_TIME * 60)  # Run for 60 seconds
print("Running inference...")
for i, prompt in enumerate(prompts):
    print(f"\n\nProcessing prompt {i}...")
    inputs = tokenizer(
        prompt, 
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model_trt.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            )

    # Print the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Prompt:")
    print(prompt)
    print('Response')
    print(generated_text[len(prompt):])

    if time.time() > end:
        print("Time's up!")
        break
