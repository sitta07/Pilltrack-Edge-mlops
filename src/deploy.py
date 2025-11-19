import os
import sys
import json
import hashlib
import yaml
from dotenv import load_dotenv

# Add path to find src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

load_dotenv()

# Import s3_helper (ต้องมั่นใจว่าไฟล์นี้มีอยู่จริงและ import ได้)
try:
    from utils.s3_helper import upload_to_s3
except ImportError:
    print("❌ Critical Error: Cannot import 'upload_to_s3' from src.utils.s3_helper")
    sys.exit(1)

def load_params(param_path="params.yaml"):
    with open(param_path, "r") as f: return yaml.safe_load(f)

def calculate_md5(file_path):
    """สร้าง MD5 Hash เพื่อใช้แปะใน Metadata"""
    if not os.path.exists(file_path): return None
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def main():
    print("🚀 Starting Deployment Stage...")
    
    # เช็ก Environment Variable ก่อนเลย
    TARGET_BUCKET = os.getenv("S3_BUCKET_NAME")
    if not TARGET_BUCKET:
        print("❌ Error: 'S3_BUCKET_NAME' env var is missing!")
        sys.exit(1)

    params = load_params()
    
    # กำหนด Path
    MODEL_PATH = "models/student_quant_int8.tflite" # หรืออ่านจาก params ก็ได้
    INDEX_PATH = params['enrollment']['index_file']
    LABELS_PATH = params['enrollment']['labels_file']
    META_PATH = params['enrollment']['metadata_file']
    S3_PREFIX = params['enrollment']['s3_prefix']

    # 1. Generate Metadata (ทำสดๆ ก่อนส่ง)
    print("📝 Generating Deployment Metadata...")
    metadata = {
        "version": "v1-auto-deploy", # อาจจะรับค่าจาก GitHub SHA ก็ได้ถ้าอยาก Advance
        "bucket": TARGET_BUCKET,
        "files": {
            "student_model.tflite": calculate_md5(MODEL_PATH),
            "pill_db.index": calculate_md5(INDEX_PATH),
            "labels.json": calculate_md5(LABELS_PATH),
        }
    }
    
    # Save Metadata ลงไฟล์ก่อน
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # 2. List รายการของที่จะส่ง
    files_to_upload = {
        MODEL_PATH: "student_model.tflite",
        INDEX_PATH: "pill_db.index",
        LABELS_PATH: "labels.json",
        META_PATH: "model_metadata.json"
    }

    # 3. Upload Loop
    print(f"☁️ Uploading artifacts to S3 Bucket: {TARGET_BUCKET}...")
    success_count = 0
    
    for local_path, s3_filename in files_to_upload.items():
        if os.path.exists(local_path):
            s3_dest = f"{S3_PREFIX}{s3_filename}"
            print(f"   Process: {local_path} -> {s3_dest}")
            
            if upload_to_s3(local_path, s3_dest, bucket_name=TARGET_BUCKET):
                print(f"   ✅ Uploaded: {s3_filename}")
                success_count += 1
            else:
                print(f"   ❌ Failed to upload: {s3_filename}")
        else:
            print(f"   ⚠️ File not found (Skipping): {local_path}")

    # 4. Summary
    total_files = len(files_to_upload)
    if success_count == total_files:
        print("\n🎉🎉 Deployment Successful! All files are on S3. 🎉🎉")
        sys.exit(0)
    else:
        print(f"\n⚠️ Deployment Incomplete! ({success_count}/{total_files} files uploaded)")
        sys.exit(1)

if __name__ == "__main__":
    main()