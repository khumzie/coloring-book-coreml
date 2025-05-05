import os
import torch
import coremltools as ct
import numpy as np
from diffusers import StableDiffusionPipeline

def main():
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
    )
    pipe.to("cpu")
    
    # Extract UNET for conversion
    unet = pipe.unet
    
    # Create a traced model of just the UNET component
    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
            
        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(sample, timestep, encoder_hidden_states).sample
            
    # Prepare example inputs for tracing
    batch_size = 1
    height = 512
    width = 512
    
    sample = torch.randn(batch_size, 4, height // 8, width // 8)
    timestep = torch.tensor([999])
    encoder_hidden_states = torch.randn(batch_size, 77, 768)
    
    # Create and trace the model
    wrapped_unet = UNetWrapper(unet)
    traced_unet = torch.jit.trace(
        wrapped_unet,
        (sample, timestep, encoder_hidden_states)
    )
    
    # Save the traced model
    torch.jit.save(traced_unet, "unet.pt")
    
    # Convert to CoreML
    mlmodel = ct.convert(
        "unet.pt",
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        inputs=[
            ct.TensorType(
                name="sample",
                shape=[batch_size, 4, height // 8, width // 8],
                dtype=np.float32
            ),
            ct.TensorType(
                name="timestep",
                shape=[1],
                dtype=np.int32
            ),
            ct.TensorType(
                name="encoder_hidden_states",
                shape=[batch_size, 77, 768],
                dtype=np.float32
            )
        ]
    )
    
    # Save as .mlpackage
    os.makedirs("ColoringBook.mlpackage", exist_ok=True)
    mlmodel.save("ColoringBook.mlpackage")
    
    print("Model conversion completed successfully!")

if __name__ == "__main__":
    main()