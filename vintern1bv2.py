from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import torch
import logging
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import Optional, Union
import aiohttp
from contextlib import asynccontextmanager
import uvicorn
from io import BytesIO
import logging
import colorlog
from starlette.datastructures import UploadFile as StarletteUploadFile
from models import VLLMModel
from utils import check_dtype, load_image


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
generation_config = dict(max_new_tokens=2048, do_sample=False, num_beams=3, repetition_penalty=2.5)

# Define global model and tokenizer variables
model = None
tokenizer = None

# Model and tokenizer loading at startup
@asynccontextmanager
async def load_model(app: FastAPI):
    global model, tokenizer
    try:
        model, tokenizer = VLLMModel.from_pretrained("5CD-AI/Vintern-1B-v2", device=device, dtype=torch_dtype)
        logger.info(f"Model loaded successfully.")
        yield
        model = None
        tokenizer = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

# Image handler function
async def handle_image(file_or_url: Union[str, UploadFile]) -> Image:
    if isinstance(file_or_url, str):  # Handle URL input
        async with aiohttp.ClientSession() as session:
            async with session.get(file_or_url) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=404, detail="Image not found")
                image_data = await resp.read()
                image = load_image(BytesIO(image_data), max_num=8)  # Use BytesIO for byte data
    elif isinstance(file_or_url, UploadFile) or isinstance(file_or_url, StarletteUploadFile):  # Handle file upload
        file_content = await file_or_url.read()
        image = load_image(BytesIO(file_content), max_num=8)  # Load image from memory
    else:
        raise HTTPException(status_code=400, detail="Invalid input type")
    return image.to(torch_dtype).to(device)

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

# Inference endpoint
@app.post("/api/inference")
async def inference(
    prompt: Union[str, bytes] = Form(...),
    file: Optional[UploadFile] = File(None),  # File input (optional)
    url: Optional[Union[str, bytes]] = Form(None),  # URL input (optional)
    ):
    
    """Call this to use OCR"""
    try:
        # Choose file or URL based on input
        logger.error(f"Error during inference: {prompt}")
        prompt = "<image>\n" + prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        file_or_url = file if file else url

        image = await handle_image(file_or_url)
        logger.info(f"Image loaded successfully.")

        # Model inference
        response, history = model.chat(tokenizer, image, prompt, generation_config, history=None, return_history=True)
        return response

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")



if __name__ == "__main__":
    # "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    # "https://huggingface.co/5CD-AI/Vintern-1B-v2/resolve/main/ex_images/1.png"
    uvicorn.run("vintern1bv2:app", host="0.0.0.0", port=8080, reload=True)
    # curl -X 'POST' \
    #   'https://30b3-34-142-236-251.ngrok-free.app/api/inference' \
    #   -H 'accept: application/json' \
    #   -H 'Content-Type: multipart/form-data' \
    #   -F 'prompt=<image>\nHãy đọc các văn bản có trong ảnh.' \
    #   -F 'file=@/home/lavender/My-Project/CV_Project/05_receipt_recognition/image.png'
