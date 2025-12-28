from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
import sys
from indextts.infer_v2 import IndexTTS2
import os
import uuid
from boto3_utils import download_s3_file, upload_s3_file


class AudioRequest(BaseModel):
    text_prompt: str
    audio_ref_s3_link: str

app = FastAPI()

@app.post("/generate-audio")
def read_root(request: AudioRequest):
  
    bucket_name = os.getenv("S3_BUCKET_NAME")
    
    if not bucket_name:
        raise RuntimeError("S3_BUCKET_NAME environment variable is not set")
    
    bucket_folder = "generated_audios"
  
    tts = IndexTTS2(
        cfg_path=os.path.join("checkpoints", "config.yaml"),
        model_dir=os.path.join("checkpoints"),
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )

    output_path = os.path.join("output", f"{uuid.uuid4()}.wav")
    
    audio_ref_path = download_s3_file(
        bucket=bucket_name,
        key=request.audio_ref_s3_link.replace(f"s3://{bucket_name}/", ""),
        local_path=os.path.join("temp", f"{uuid.uuid4()}_ref.wav")
    )

    tts.infer(
        spk_audio_prompt=audio_ref_path,
        text=request.text_prompt,
        output_path=output_path,
        emo_alpha=0.6,
        use_emo_text=True,
        use_random=False,
        verbose=True,
    )

    print("generated audio path:", output_path)
    
    audio_file_name = os.path.basename(output_path)
    
    upload_s3_file(
        bucket=bucket_name,
        key=f"{bucket_folder}/{audio_file_name}",
        local_path=output_path,
    )
    
    # Clean up local files
    os.remove(audio_ref_path)
    os.remove(output_path)
    
    full_generated_audio_url = f"https://{bucket_name}.s3.amazonaws.com/{bucket_folder}/{audio_file_name}"
    
    return {"generated_audio_s3_link": f"{full_generated_audio_url}"}