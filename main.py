import os
import io
from dotenv import load_dotenv # 1. Import the loader
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from deep_translator import GoogleTranslator
load_dotenv()

app = FastAPI()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found! Did you create the .env file?")

client = Groq(api_key=api_key)

class TranslationRequest(BaseModel):
    text: str
    target_lang: str

@app.get("/")
async def root():
    return {"message": "Server is running!"}

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    print(f"Translating: {request.text} to {request.target_lang}")
    try:
        translated = GoogleTranslator(source='auto', target=request.target_lang).translate(request.text)
        return {"translated_text": translated}
    except Exception as e:
        print(f"Translation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Connection opened")
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            if not audio_data or len(audio_data) < 100:
                continue
            
            try:
                buffer = io.BytesIO(audio_data)
                buffer.name = "audio.wav"
                
                transcription = client.audio.transcriptions.create(
                    file=buffer,
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )
                
                if transcription.strip():
                    print(f"Heard: {transcription.strip()}")
                    await websocket.send_json({"text": transcription.strip()})
            
            except Exception as api_err:
                print(f"Groq API Error: {api_err}")
                await websocket.send_json({"error": "Transcription failed"})
                
    except Exception as ws_err:
        print(f"WebSocket Error: {ws_err}")
    finally:
        print("Connection closed")