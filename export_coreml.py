import os
import torch
import coremltools as ct
from diffusers import StableDiffusionPipeline

def main():
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
    )
    pipe.to("cpu")

    # Dummy input for tracing
    sample_input = pipe.tokenizer("a cat in a hat", return_tensors="pt").input_ids

    # Export to CoreML
    mlmodel = ct.convert(
        lambda x: pipe(prompt="a cat in a hat").images[0],
        inputs=[ct.TensorType(shape=sample_input.shape)],
    )

    # Save as .mlpackage
    os.makedirs("ColoringBook.mlpackage", exist_ok=True)
    mlmodel.save("ColoringBook.mlpackage")

if __name__ == "__main__":
    main()