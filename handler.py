import runpod
import torch
import base64
import io
from diffusers import AutoPipelineForText2Image

# Global variable to hold the model
pipe = None

def init_model():
    global pipe
    print("Loading Z-Image-Turbo model...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    # Optimize for 8-step inference (Turbo)
    pipe.scheduler.config.num_inference_steps = 8
    print("Model loaded successfully.")

def handler(job):
    global pipe
    
    # Get job input
    job_input = job["input"]
    prompt = job_input.get("prompt", "A futuristic city with flying cars")
    negative_prompt = job_input.get("negative_prompt", "")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    # Default to 8 steps if not specified, but model is turbo so 8 is good
    num_inference_steps = job_input.get("num_inference_steps", 8) 
    
    # Generate image
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            guidance_scale=job_input.get("guidance_scale", 7.5) # Standard guidance
        ).images[0]
    
    # Convert to Base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": img_str}

# Initialize the model on startup
if torch.cuda.is_available():
    init_model()

runpod.serverless.start({"handler": handler})
