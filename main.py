from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import json
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "CRITICAL: OPENAI_API_KEY is missing from Render environment variables."}

        client = openai.OpenAI(api_key=api_key)

        # Read file contents
        contents = await file.read()
        if len(contents) == 0:
            return {"error": "Uploaded file is completely empty."}

        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        try:
            # 1. Test OpenAI Whisper Call
            with open(temp_path, "rb") as audio_file:
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
        except Exception as openai_w_err:
            return {"error": f"OpenAI Whisper API Failed: {str(openai_w_err)}"}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        transcript_text = getattr(transcript_response, "text", "")
        if not transcript_text:
            return {"error": "Whisper transcription returned empty text."}

        # 2. Test GPT-4o Call
        prompt = f"Analyze this transcript and return valid JSON with a 'clips' array containing 3 objects with keys: title, duration, score, start, end, caption, hashtags, subtitles. Transcript: {transcript_text}"

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
        except Exception as openai_gpt_err:
            return {"error": f"OpenAI GPT-4o API Failed: {str(openai_gpt_err)}"}

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        # Return full traceback directly in the JSON response so you can read it instantly
        return {"error": "Server Exception", "details": str(e), "trace": traceback.format_exc()}

@app.post("/export-clip")
async def export_clip(file: UploadFile = File(...), start: float = 0.0, end: float = 30.0):
    return {"status": "ok"}
