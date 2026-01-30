import runpod
import torch
import base64
import io
from diffusers import ZImagePipeline # Use the specific class for 2026 Z-Image support

# Global variable to hold the model
pipe = None

def init_model():
    global pipe
    if pipe is None:
        print("Loading Z-Image-Turbo (6B)...")
        # Use bfloat16 for the best performance on RTX 4090 / A100
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False
        ).to("cuda")
        print("Model loaded successfully.")

def handler(job):
    global pipe
    # Ensure model is initialized (safe-guard for certain worker configurations)
    if pipe is None:
        init_model()
    
    try:
        job_input = job.get("input", {})
        prompt = job_input.get("prompt", "A futuristic city with flying cars")
        negative_prompt = job_input.get("negative_prompt", "")
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        
        # TURBO RULE: Standard is 9 steps (which equals 8 actual forwards) 
        # and 0.0 guidance scale.
        steps = job_input.get("num_inference_steps", 9) 
        guidance = job_input.get("guidance_scale", 0.0) 

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height
            ).images[0]
        
        # Convert to Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"image": img_str}

    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

# Initialize immediately on cold start
init_model()

runpod.serverless.start({"handler": handler})
