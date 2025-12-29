from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import json
import multiprocessing as mp
from multiprocessing import Semaphore

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict

from indextts.infer_v2 import IndexTTS2
from boto3_utils import download_s3_file, upload_s3_file
from datetime import datetime


MAX_CONCURRENT_JOBS = 1
job_semaphore = Semaphore(MAX_CONCURRENT_JOBS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

active_connections: Dict[str, WebSocket] = {}

class AudioRequest(BaseModel):
    text_prompt: str
    audio_ref_s3_key: str
    

def now_local_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tts_worker(job_id: str, request: dict):
    bucket_name = os.getenv("S3_BUCKET_NAME")
    output_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        def send(msg):
            ws = active_connections.get(job_id)
            if ws:
                ws.send_text(json.dumps(msg))

        send({"status": "downloading_reference"})
        print("Downloading reference audio from S3...", now_local_str())

        audio_ref_path = download_s3_file(
            bucket=bucket_name,
            key=request["audio_ref_s3_key"],
            local_path=output_dir
        )
        
        print("Loading IndexTTS2 model...", now_local_str())

        # for some reason 
        tts = IndexTTS2(
            cfg_path=os.path.join(BASE_DIR, "checkpoints", "config.yaml"),
            model_dir=os.path.join(BASE_DIR, "checkpoints"),
            use_fp16=True,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )

        send({"status": "generating_audio"})
        print("Generating audio...", now_local_str())

        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

        tts.infer(
            spk_audio_prompt=audio_ref_path,
            text=request["text_prompt"],
            output_path=output_path,
            emo_alpha=0.6,
            use_emo_text=True,
            use_random=False,
            verbose=True,
        )

        generated_file = os.path.basename(output_path)

        send({"status": "uploading_to_s3"})
        print("Uploading generated audio to S3...", now_local_str())

        upload_s3_file(
            local_path=os.path.join(output_dir, generated_file),
            bucket=bucket_name,
        )

        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{generated_file}"

        send({
            "status": "completed",
            "s3_url": s3_url
        })
        
        print("Job completed.", now_local_str())

    except Exception as e:
        send({
            "status": "error",
            "error": str(e)
        })

    finally:
        try:
            # cleanup
            os.remove(audio_ref_path)
            os.remove(output_path)
            job_semaphore.release()
        except Exception:
            pass


@app.post("/generate-audio")
def generate_audio(request: AudioRequest):
    acquired = job_semaphore.acquire(block=False)
    if not acquired:
        raise HTTPException(429, "Server busy")
    
    job_id = str(uuid.uuid4())

    process = mp.Process(
        target=tts_worker,
        args=(job_id, request.dict())
    )
    process.start()

    return {
        "job_id": job_id,
        "ws_url": f"/ws/{job_id}"
    }



@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    active_connections[job_id] = websocket

    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        active_connections.pop(job_id, None)
