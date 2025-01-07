from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import logging
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import Union
import aiohttp
from contextlib import asynccontextmanager
import uvicorn
from io import BytesIO
import logging
import colorlog
from models import VLLMModel, VLLMModelForCausalLM
from utils import check_dtype

# Initialize logger
logger = logging.getLogger(__name__)

# Create a colored stream handler
handler = colorlog.StreamHandler()

# Set the log level
logger.setLevel(logging.DEBUG)

# Define the log format with colors
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)

# Set formatter to the handler
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)



# Set device and dtype for model loading
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = check_dtype(device)

# Define global model and processor variables
model = None
processor = None

# Model and processor loading at startup
@asynccontextmanager
async def load_model(app: FastAPI):
    global model, processor
    try:
        
        model, processor = VLLMModelForCausalLM.from_pretrained("microsoft/Florence-2-base", device=device, dtype=torch_dtype)
        logger.info(f"Model loaded successfully.")
        yield

        model = None
        processor = None

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

# Image handler function
async def handle_image(path_or_url: Union[str, bytes]) -> Image:
    if isinstance(path_or_url, str) and path_or_url.startswith("http"):
        async with aiohttp.ClientSession() as session:
            async with session.get(path_or_url) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=404, detail="Image not found")
                image_data = await resp.read()
                image = Image.open(BytesIO(image_data)).convert("RGB")  # Use BytesIO to handle the byte data
    else:
        image = Image.open(path_or_url).convert("RGB")
    return image

# FastAPI app
app = FastAPI(lifespan=load_model)

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to a list of domains to restrict access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request
class Request(BaseModel):
    task_prompt: str
    path_or_url: Union[str, bytes]
    text_input: str = None

# Inference endpoint
@app.post("/api/inference")
async def inference(request: Request):
    try:
        task_prompt = request.task_prompt
        path_or_url = request.path_or_url
        text_input = request.text_input

        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        image = await handle_image(path_or_url=path_or_url)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

        return parsed_answer

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    # test content example
    # path_or_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    # task_prompt = "<OD>"" 
    uvicorn.run("florence2base:app", host="0.0.0.0", port=8080, reload=True)
    # curl -X POST http://localhost:8080/api/inference -H "Content-Type: application/json" -d '{"task_prompt": "<OCR>", "path_or_url": "image.png"}'

    # {
    #     "task_prompt": "<OCR>",
    #     "path_or_url": "https://huggingface.co/5CD-AI/Vintern-1B-v2/resolve/main/ex_images/1.png"
    # }