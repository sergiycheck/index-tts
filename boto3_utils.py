import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()

# loading env variables from .env file
# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

s3 = boto3.client("s3")


def download_s3_file(bucket: str, key: str, local_path: str) -> str:
    """
    Download an S3 object to a local file.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3.download_file(bucket, key, local_path)
    except ClientError as e:
        raise RuntimeError(f"Failed to download s3://{bucket}/{key}: {e}")

    return local_path


def upload_s3_file(bucket: str, key: str, local_path: str, content_type: str | None = None):
    """
    Upload a local file to S3.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)

    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type

    try:
        s3.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
    except ClientError as e:
        raise RuntimeError(f"Failed to upload {local_path} to s3://{bucket}/{key}: {e}")
