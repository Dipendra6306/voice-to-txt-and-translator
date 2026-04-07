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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

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
        print(f"Translation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🚀 Connection opened")
    audio_chunks = bytearray()
    
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            audio_chunks.extend(audio_data)
            
            if len(audio_chunks) > 40000: 
                print(f"📦 Processing buffer: {len(audio_chunks)} bytes")
                try:
                    buffer = io.BytesIO(audio_chunks)
                    buffer.name = "audio.webm" 
                   
                    transcription = client.audio.transcriptions.create(
                        file=buffer,
                        model="whisper-large-v3-turbo",
                        response_format="text"
                    )
                    
                    text = transcription.strip()
                    if text:
                        print(f"👂 Heard: {text}")
                        await websocket.send_json({"text": text})
                    else:
                        print("❓ Groq returned empty text (Maybe silence?)")
                    
                    audio_chunks.clear()
            
                except Exception as api_err:
                    print(f"⚠️ Groq API Error: {api_err}")
                    
                    continue
                    
    except Exception as ws_err:
        print(f"❌ WebSocket Closed: {ws_err}")
    finally:
        print("🔌 Connection closed")