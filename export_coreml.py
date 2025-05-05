import os
import torch
import coremltools as ct
from diffusers import StableDiffusionPipeline

def main():
    # Load the model with low CPU memory usage
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    pipe.to("cpu")

    # Dummy input for tracing
    sample_input = pipe.tokenizer("a cat in a hat", return_tensors="pt").input_ids

    # Export to CoreML with PyTorch as source
    mlmodel = ct.convert(
        lambda x: pipe(prompt="a cat in a hat").images[0],
        inputs=[ct.TensorType(shape=sample_input.shape)],
        source="pytorch"
    )

    # Save as .mlpackage
    os.makedirs("ColoringBook.mlpackage", exist_ok=True)
    mlmodel.save("ColoringBook.mlpackage")

if __name__ == "__main__":
    main()