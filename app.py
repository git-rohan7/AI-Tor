# Updated app.py with OCR (Tesseract) integration for image files

import os
import requests
import logging
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import io

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
groq_api_key = os.getenv("GROQ_API_KEY", "your-groq-api-key")
headers = {
    "Authorization": f"Bearer {groq_api_key}",
    "Content-Type": "application/json"
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_and_query")
async def upload_and_query(request: Request, file: UploadFile = File(...), user_query: str = Form(...)):
    try:
        content = await file.read()

        # OCR step: Convert image bytes to text using Tesseract
        image = Image.open(io.BytesIO(content))
        text_from_file = pytesseract.image_to_string(image)

        messages = [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": f"Document: {text_from_file}\n\nQuestion: {user_query}"}
        ]

        payload = {
            "model": LLM_MODEL,
            "messages": messages
        }

        response = requests.post(groq_api_url, headers=headers, json=payload)

        if response.status_code != 200:
            logger.error(f"Error from llama API: {response.status_code} - {response.text}")
            answer = "Error from LLM API."
        else:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]

    except Exception as e:
        logger.exception("An error occurred during processing.")
        answer = "An error occurred while processing your request."

    return templates.TemplateResponse("index.html", {"request": request, "answer": answer})



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)