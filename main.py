from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from claim_processor import ClaimProcessor
import shutil
import uuid
import os
import tempfile  # move import here, globally is fine
import traceback

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_claim/")
async def process_claim(query: str = Form(...), file: UploadFile = File(...)):
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[INFO] PDF saved to: {temp_path}")

        processor = ClaimProcessor(temp_path)
        result = processor.process_claim(query)

        print("[INFO] Claim processed. Result:", result)

        os.remove(temp_path)
        return JSONResponse(content=result)

    except Exception as e:
        print("[ERROR] An exception occurred:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
