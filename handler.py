import multiprocessing as mp
import os
from boto3_utils import download_s3_file, upload_s3_file
from utils import now_local_str
from indextts.infer_v2 import IndexTTS2
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def tts_worker(audio_ref_s3_key: str, text_prompt: str):

  try:
    bucket_name = os.getenv("S3_BUCKET_NAME")
    
    if not bucket_name:
        raise RuntimeError("S3_BUCKET_NAME environment variable is not set")
    
    output_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading reference audio from S3...", now_local_str())

    audio_ref_path = download_s3_file(
        bucket=bucket_name,
        key=audio_ref_s3_key,
        local_path=output_dir
    )
    
    print("Loading IndexTTS2 model...", now_local_str())

    tts = IndexTTS2(
        cfg_path=os.path.join("checkpoints", "config.yaml"),
        model_dir=os.path.join("checkpoints"),
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )

    print("Generating audio...", now_local_str())

    output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

    tts.infer(
        spk_audio_prompt=audio_ref_path,
        text=text_prompt,
        output_path=output_path,
        emo_alpha=0.6,
        use_emo_text=True,
        use_random=False,
        verbose=True,
    )

    generated_file = os.path.basename(output_path)

    print("Uploading generated audio to S3...", now_local_str())

    upload_s3_file(
        local_path=os.path.join(output_dir, generated_file),
        bucket=bucket_name,
    )

    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{generated_file}"
    print("Job completed.", now_local_str())
    print("Generated audio S3 URL:", s3_url)
    
    return({
        "status": "completed",
        "s3_url": s3_url
    })
  except Exception as e:
    print(f"Error in tts_worker: {e}")
    return({
        "status": "error",
        "error": str(e)
    })
  finally:
    os.remove(audio_ref_path)
    os.remove(output_path)
  



