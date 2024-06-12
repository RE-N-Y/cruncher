import os
from dotenv import load_dotenv

load_dotenv()

def get_s3_key():
    return os.getenv("S3_KEY")

def get_s3_secret():
    return os.getenv("S3_SECRET")

def get_s3_endpoint():
    return os.getenv("S3_ENDPOINT")