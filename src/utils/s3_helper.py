# src/utils/s3_helper.py
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

def upload_to_s3(local_path, s3_key, bucket_name):
    """
    Uploads a file to S3 and provides detailed AWS error codes on failure.
    """
    s3 = boto3.client('s3')
    
    # 🎯 สิ่งที่กำลังทำ: พยายามโยนไฟล์ขึ้น S3
    print(f"🚀 Uploading {os.path.basename(local_path)} to s3://{bucket_name}/{s3_key} ...")
    
    try:
        s3.upload_file(local_path, bucket_name, s3_key)
        return True
    
    except FileNotFoundError:
        print(f"❌ UPLOAD FAILED: Local file not found: {local_path}")
        return False
    
    except NoCredentialsError:
        print("❌ UPLOAD FAILED: AWS Credentials not available in environment.")
        return False
        
    except ClientError as e:
        # 🔥 จุดที่สำคัญ: ดึง Error Code ของ AWS ออกมา
        error_code = e.response.get('Error', {}).get('Code')
        error_message = e.response.get('Error', {}).get('Message')
        
        print(f"❌ UPLOAD FAILED (AWS Error Code: {error_code})")
        print(f"   Reason: {error_message} (Bucket: {bucket_name})")
        return False

def download_from_s3(s3_key, local_path, bucket_name):
    """
    Downloads a file from S3 (Used for fetching YOLO/Teacher if needed)
    """
    s3 = boto3.client('s3')
    
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket_name, s3_key, local_path)
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == '404' or error_code == 'NoSuchKey':
            # นี่ไม่ใช่ความผิดพลาด แค่ไฟล์ไม่ได้อยู่ตรงนั้น
            print(f"⚠️ DOWNLOAD SKIPPED: File not found on S3 (Code: 404)")
        else:
             print(f"❌ DOWNLOAD FAILED: S3 Client Error Code: {error_code}")
        return False