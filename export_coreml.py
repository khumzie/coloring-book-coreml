import os
import torch
import coremltools as ct
from diffusers import StableDiffusionPipeline
import numpy as np

def main():
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
    )
    pipe.to("cpu")

    # Create a scripted model
    class StableDiffusionWrapper(torch.nn.Module):
        def __init__(self, pipe):
            super().__init__()
            self.pipe = pipe

        def forward(self, prompt):
            return self.pipe(prompt).images[0]

    model = StableDiffusionWrapper(pipe)
    scripted_model = torch.jit.script(model)

    # Save the scripted model
    torch.jit.save(scripted_model, "model.pt")

    # Convert to CoreML
    mlmodel = ct.convert(
        "model.pt",
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        inputs=[
            ct.TensorType(
                name="prompt",
                shape=[1, 77],  # Default token sequence length for SD
                dtype=np.int32
            )
        ]
    )

    # Save as .mlpackage
    os.makedirs("ColoringBook.mlpackage", exist_ok=True)
    mlmodel.save("ColoringBook.mlpackage")

if __name__ == "__main__":
    main()