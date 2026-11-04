import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="learnwtf Backend")

# Enable CORS for frontend communication (Vercel / Localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase & OpenAI
SUPABASE_URL = os.getenv("https://sryjtwofoxyuscntjrbt.supabase.co")
SUPABASE_KEY = os.getenv("sb_publishable_E8_wuBS8vHmlEoG8fxXhHA_kfAkHIuI")
OPENAI_API_KEY = os.getenv("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")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

@app.get("/")
def health_check():
    return {"status": "online", "system": "learnwtf backend active"}

@app.post("/api/upload-and-rename")
async def upload_and_rename(file: UploadFile = File(...)):
    try:
        # 1. Read file bytes
        file_bytes = await file.read()
        
        # 2. Use OpenAI to generate a clean title
        prompt = f"Given the filename '{file.filename}', generate a concise, professional, snake_case or clean title for a knowledge management system (max 5 words, no extension, no quotes)."
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        auto_title = response.choices[0].message.content.strip().replace('"', '')

        # 3. Upload file to Supabase Storage bucket ('documents')
        storage_path = f"{auto_title}_{file.filename}"
        supabase.storage.from_('documents').upload(
            storage_path, 
            file_bytes, 
            file_options={"content-type": file.content_type, "upsert": "true"}
        )
        
        # 4. Get public URL
        public_url_resp = supabase.storage.from_('documents').get_public_url(storage_path)

        # 5. Save record to 'knowledge_files' table
        db_record = {
            "original_name": file.filename,
            "auto_renamed_title": auto_title,
            "file_url": public_url_resp,
            "file_type": "document",
            "access_tier": "private"
        }
        supabase.table("knowledge_files").insert(db_record).execute()

        return {"status": "success", "auto_renamed_title": auto_title, "file_url": public_url_resp}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag-chat")
async def rag_chat(query: str):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are learnwtf AI, an analytical assistant grounded strictly in the user's uploaded knowledge ecosystem."},
                {"role": "user", "content": query}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))