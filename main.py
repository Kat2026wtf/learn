from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import json
import subprocess
import tempfile
import traceback
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_trace = traceback.format_exc()
    print("GLOBAL EXCEPTION:", error_trace)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "trace": error_trace},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

def seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    normalized_mp4_path = os.path.join(temp_dir, "normalized.mp4")

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing from Render environment variables.")

        client = openai.OpenAI(api_key=api_key)

        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        with open(input_path, "wb") as f:
            f.write(contents)

        # Normalize video/MOV via FFmpeg
        conversion_cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            normalized_mp4_path, "-y"
        ]
        subprocess.run(conversion_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Transcribe once via Whisper
        with open(normalized_mp4_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )

        transcript_text = getattr(transcript_response, "text", "")
        segments = getattr(transcript_response, "segments", [])

        if not transcript_text or not transcript_text.strip():
            raise HTTPException(status_code=400, detail="Transcription came back empty.")

        prompt = f"""
        You are an expert social media growth hacker. Analyze this full video transcript and identify the 3 HIGHEST VIRALITY moments.
        Transcript: "{transcript_text}"
        CRITICAL RULES:
        1. Each identified clip MUST have a duration between 15 and 30 seconds.
        2. Provide exact "start" and "end" timestamps in seconds.
        Return a valid JSON object with a key "clips" containing an array of exactly 3 objects:
        - "title": Short viral title
        - "duration": String duration (e.g., "24s")
        - "score": Virality score percentage (e.g., "97% Virality")
        - "start": start time in seconds (number)
        - "end": end time in seconds (number)
        - "caption": Engaging social media caption
        - "hashtags": An object with keys "broad", "niche", and "growth"
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        parsed_data = json.loads(response.choices[0].message.content)
        parsed_data["segments"] = segments  # Pass segments so export doesn't re-run Whisper!
        return parsed_data

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/export-clip")
async def export_clip(
    file: UploadFile = File(...), 
    start: float = Form(0.0), 
    end: float = Form(30.0),
    segments_json: str = Form("[]")
):
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    normalized_path = os.path.join(temp_dir, "normalized.mp4")
    srt_path = os.path.join(temp_dir, "subtitles.srt")
    output_path = os.path.join(temp_dir, "final_viral_clip.mp4")

    try:
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)

        conversion_cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            normalized_path, "-y"
        ]
        subprocess.run(conversion_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            segments = json.loads(segments_json)
        except:
            segments = []

        srt_content = ""
        for idx, seg in enumerate(segments, start=1):
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            if seg_end >= start and seg_start <= end:
                s_time = max(0.0, seg_start - start)
                e_time = max(0.0, seg_end - start)
                text = seg.get("text", "").strip()
                srt_content += f"{idx}\n{seconds_to_srt_time(s_time)} --> {seconds_to_srt_time(e_time)}\n{text}\n\n"

        with open(srt_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        duration = max(5.0, min(30.0, end - start))

        ffmpeg_cmd = [
            "ffmpeg", "-ss", str(start), "-i", normalized_path, "-t", str(duration),
            "-vf", f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,subtitles={srt_path}:force_style='FontSize=24,PrimaryColour=&H00FFFF&,OutlineColour=&H000000&,BorderStyle=3'",
            "-c:v", "libx264", "-crf", "23", "-c:a", "aac", output_path, "-y"
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        return JSONResponse(
            content={"status": "success", "clip_duration": duration, "size": len(video_bytes)},
            headers={"Access-Control-Allow-Origin": "*"}
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        print("EXPORT ERROR:\n", error_detail)
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "trace": error_detail},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
