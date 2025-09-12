from dotenv import load_dotenv
import os, boto3, botocore

# Load .env from current directory
load_dotenv(".env")

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    region_name=os.getenv("MINIO_REGION", "us-east-1"),
    verify=False,
)

print("Endpoint:", os.getenv("MINIO_ENDPOINT"))
print("Bucket:", os.getenv("MINIO_BUCKETS"))

try:
    resp = s3.list_objects_v2(Bucket=os.getenv("MINIO_BUCKETS"), MaxKeys=5)
    print("ListObjects OK:", [o["Key"] for o in resp.get("Contents", [])])
except botocore.exceptions.ClientError as e:
    print("ListObjects FAILED:", e.response["Error"])
