import os
import io
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from deep_translator import GoogleTranslator

load_dotenv()
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env!")

client = Groq(api_key=api_key)

class TranslationRequest(BaseModel):
    text: str
    target_lang: str

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    try:
        translated = GoogleTranslator(source='auto', target=request.target_lang).translate(request.text)
        return {"translated_text": translated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🚀 Connection opened")
    audio_buffer = bytearray()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            # We wait for a larger chunk (~5-7 seconds of audio) 
            # to ensure high-quality transcription
            if len(audio_buffer) > 120000: 
                try:
                    buffer = io.BytesIO(audio_buffer)
                    buffer.name = "audio.webm" 
                    
                    transcription = client.audio.transcriptions.create(
                        file=buffer,
                        model="whisper-large-v3-turbo",
                        response_format="text"
                    )
                    
                    if transcription.strip():
                        print(f"👂 Heard: {transcription.strip()}")
                        await websocket.send_json({"text": transcription.strip()})
                    
                    audio_buffer.clear() # Clear backend memory after sending
            
                except Exception:
                    continue 
                    
    except Exception as e:
        print(f"❌ WebSocket Closed: {e}")
    finally:
        print("🔌 Connection closed")