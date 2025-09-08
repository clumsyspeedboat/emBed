import pytest
from moto import mock_aws
import boto3
from src.storage import MinioClient

@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

@pytest.fixture
def s3_client(aws_credentials):
    with mock_aws():
        yield boto3.client("s3", region_name="us-east-1")

def test_minio_client_ensure_bucket(s3_client):
    """Tests that a bucket is created if it doesn't exist."""
    minio_client = MinioClient(endpoint="https://s3.amazonaws.com", access_key="testing", secret_key="testing")
    
    # Check that the bucket does not exist
    with pytest.raises(Exception):
        s3_client.head_bucket(Bucket="test-bucket")
        
    # Create the bucket
    minio_client.ensure_bucket("test-bucket")
    
    # Check that the bucket now exists
    response = s3_client.head_bucket(Bucket="test-bucket")
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200

def test_minio_client_list_objects(s3_client):
    """Tests listing objects in a bucket."""
    minio_client = MinioClient(endpoint="https://s3.amazonaws.com", access_key="testing", secret_key="testing")
    bucket_name = "list-test-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    s3_client.put_object(Bucket=bucket_name, Key="file1.txt", Body="hello")
    s3_client.put_object(Bucket=bucket_name, Key="folder/file2.txt", Body="world")
    
    keys = minio_client.list_objects(bucket_name)
    assert len(keys) == 2
    assert "file1.txt" in keys
    assert "folder/file2.txt" in keys
    
    keys_with_prefix = minio_client.list_objects(bucket_name, prefix="folder/")
    assert len(keys_with_prefix) == 1
    assert "folder/file2.txt" in keys_with_prefix