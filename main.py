from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import json
import subprocess
import tempfile
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing from Render environment variables.")

        client = openai.OpenAI(api_key=api_key)

        # Save uploaded file
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)

        # 1. Transcribe via OpenAI Whisper (safely without crashing on older SDK versions)
        with open(input_path, "rb") as audio_file:
            try:
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
            except Exception as openai_err:
                raise HTTPException(status_code=500, detail=f"OpenAI Whisper Error: {str(openai_err)}")

        transcript_text = getattr(transcript_response, "text", "")
        segments = getattr(transcript_response, "segments", [])

        if not transcript_text or not transcript_text.strip():
            raise HTTPException(status_code=400, detail="Transcription came back empty. Ensure your video file has clear audio.")

        # 2. GPT-4o Prompt for High-Virality Clips
        prompt = f"""
        You are an expert social media growth hacker. Analyze this full video transcript and identify the 3 HIGHEST VIRALITY moments (hooks, surprising statements, or core breakthroughs).

        Transcript: "{transcript_text}"

        CRITICAL RULES:
        1. Each identified clip MUST have a duration between 15 and 30 seconds (15 <= (end - start) <= 30).
        2. Provide exact "start" and "end" timestamps in seconds based on the transcript timeline.

        Return a valid JSON object with a key "clips" containing an array of exactly 3 objects with these exact fields:
        - "title": Short viral title for the hook
        - "duration": String duration (e.g., "24s")
        - "score": Virality score percentage (e.g., "97% Virality")
        - "start": start time in seconds (number)
        - "end": end time in seconds (number)
        - "caption": Engaging social media caption / hook matching this specific high-virality window
        - "hashtags": An object with keys "broad", "niche", and "growth" containing tailored hashtags
        - "subtitles": An array of objects with "time" (seconds number relative to clip start) and "text" (short subtitle phrase)
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
        except Exception as gpt_err:
            raise HTTPException(status_code=500, detail=f"OpenAI GPT-4o Error: {str(gpt_err)}")

        parsed_data = json.loads(response.choices[0].message.content)
        return parsed_data

    except HTTPException as he:
        raise he
    except Exception as e:
        error_detail = traceback.format_exc()
        print("CRITICAL BACKEND ERROR:\n", error_detail)
        raise HTTPException(status_code=500, detail=f"Server Exception: {str(e)}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.post("/export-clip")
async def export_clip(file: UploadFile = File(...), start: float = 0.0, end: float = 30.0):
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    srt_path = os.path.join(temp_dir, "subtitles.srt")
    output_path = os.path.join(temp_dir, "final_viral_clip.mp4")

    try:
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)

        api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)

        with open(input_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )

        segments = getattr(transcript_response, "segments", [])
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
            "ffmpeg", "-ss", str(start), "-i", input_path, "-t", str(duration),
            "-vf", f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,subtitles={srt_path}:force_style='FontSize=24,PrimaryColour=&H00FFFF&,OutlineColour=&H000000&,BorderStyle=3'",
            "-c:v", "libx264", "-crf", "23", "-c:a", "aac", output_path, "-y"
        ]

        subprocess.run(ffmpeg_cmd, check=True)

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        return {"status": "success", "clip_duration": duration, "size": len(video_bytes)}

    except Exception as e:
        error_detail = traceback.format_exc()
        print("EXPORT ERROR:\n", error_detail)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in [input_path, srt_path, output_path]:
            if os.path.exists(path):
                os.remove(path)
