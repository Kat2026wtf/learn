from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import json

app = FastAPI()

# Enable CORS for your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Transcribes uploaded video audio via OpenAI Whisper, identifies top 15-30s viral clips,
    and returns matching captions, hooks, and hashtags based on actual video transcript.
    """
    temp_path = f"temp_{file.filename}"
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on Render environment variables.")

        client = openai.OpenAI(api_key=api_key)

        # Save temporary video file safely
        contents = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(contents)

        # 1. Transcribe actual audio via OpenAI Whisper with timestamp segments
        with open(temp_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        transcript_text = getattr(transcript_response, "text", str(transcript_response))
        
        if not transcript_text or len(transcript_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Audio transcription came back empty. Ensure the video has sound.")

        # 2. Prompt GPT-4o to analyze transcript & return JSON with 3 distinct 15-30s clip options
        prompt = f"""
        Analyze this video transcript and return valid JSON containing 3 distinct 15-30 second viral clip segments.
        
        Transcript: "{transcript_text}"

        Return a JSON object with a key "clips" containing an array of 3 objects with these exact fields:
        - "title": Short descriptive title (e.g., "Option A: The Core Strategy")
        - "duration": String duration (e.g., "24s")
        - "score": Virality score percentage (e.g., "96% Virality")
        - "start": start time in seconds (number, based on transcript segment timings if possible, or 0-indexed estimates)
        - "end": end time in seconds (number)
        - "caption": Engaging social media caption matching the exact topic discussed in this segment
        - "hashtags": An object with keys "broad", "niche", and "growth" containing tailored hashtag strings
        - "subtitles": An array of objects with "time" (seconds number) and "text" (short subtitle phrase for video overlay)
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content
        
        # Safely parse JSON to ensure it returns clean structured JSON object to frontend
        parsed_json = json.loads(raw_content)
        return parsed_json

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Always cleanup temp file even if errors happen
        if os.path.exists(temp_path):
            os.remove(temp_path)
