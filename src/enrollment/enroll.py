import os
import sys
import yaml
import json
import hashlib
import numpy as np
import tensorflow as tf
import faiss
from glob import glob
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv 

# เพิ่ม path ให้ Python มองเห็น module src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

load_dotenv()

try:
    from src.utils.s3_helper import upload_to_s3, download_from_s3
except ImportError:
    print("⚠️ Warning: s3_helper not found. S3 Upload/Download will fail.")
    def upload_to_s3(path, key, bucket_name=None): return False
    def download_from_s3(key, path, bucket_name=None): return False

def load_params(param_path="params.yaml"):
    with open(param_path, "r") as f: return yaml.safe_load(f)

def calculate_md5(file_path):
    """คำนวณ Hash MD5 ของไฟล์"""
    if not os.path.exists(file_path): return None
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_embedding(interpreter, image_path, input_details, output_details):
    """สกัด Feature Vector จากรูปภาพโดยใช้ TFLite"""
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    scale, zero_point = input_details[0]['quantization']

    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((input_shape[1], input_shape[2]))
        
        input_data = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_data = (input_data - mean) / std
        input_data = np.expand_dims(input_data, axis=0)

        if input_shape[3] == 3: pass
        elif input_shape[1] == 3: input_data = np.transpose(input_data, (0, 3, 1, 2))

        if input_dtype == np.int8:
            input_data = (input_data / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        return interpreter.get_tensor(output_index).flatten()
        
    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        return None

def main():
    params = load_params()
    
    # --- Configs ---
    DATA_DIR = params['data']['extract_path']
    MODEL_PATH = "models/student_quant_int8.tflite" 
    
    INDEX_OUT = params['enrollment']['index_file']
    LABELS_OUT = params['enrollment']['labels_file']
    META_OUT = params['enrollment']['metadata_file']
    
    S3_PREFIX = params['enrollment']['s3_prefix']
    
    TARGET_BUCKET = os.getenv("S3_BUCKET_NAME")
    
    if not TARGET_BUCKET:
        print("❌ Error: 'S3_BUCKET_NAME' not found in .env")
        sys.exit(1)

    print(f"🎯 Target S3 Bucket: {TARGET_BUCKET}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ ไม่พบไฟล์โมเดล {MODEL_PATH} กรุณารัน dvc repro convert ก่อน")
        sys.exit(1)

    # --- 1. Load TFLite Interpreter ---
    print("⚙️ Loading TFLite Interpreter...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # --- 2. Build Database (Enrollment) ---
    embeddings = []
    labels_map = {}
    current_id = 0
    
    train_dir = os.path.join(DATA_DIR, "pills_dataset_resnet", "train")
    if not os.path.exists(train_dir): train_dir = DATA_DIR
    
    try:
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    except FileNotFoundError:
        print(f"❌ หาโฟลเดอร์ข้อมูลไม่เจอที่: {train_dir}"); sys.exit(1)

    print(f"🚀 Start Enrollment for {len(classes)} classes...")

    for pill_name in tqdm(classes, desc="Indexing"):
        folder_path = os.path.join(train_dir, pill_name)
        images = glob(os.path.join(folder_path, "*.jpg"))[:30] 
        
        if not images: continue

        for img_path in images:
            vector = get_embedding(interpreter, img_path, input_details, output_details)
            if vector is not None:
                embeddings.append(vector)
                labels_map[current_id] = pill_name
                current_id += 1

    # --- 3. Save Index & Labels ---
    if len(embeddings) > 0:
        print(f"\n💾 Saving Database (Total Vectors: {len(embeddings)})...")
        
        embeddings_matrix = np.array(embeddings).astype('float32')
        d = embeddings_matrix.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings_matrix)
        
        os.makedirs(os.path.dirname(INDEX_OUT), exist_ok=True)
        faiss.write_index(index, INDEX_OUT)
        
        with open(LABELS_OUT, 'w') as f: json.dump(labels_map, f, indent=2)
            
        print(f"✅ Saved Index & Labels")
    else:
        print("❌ Error: No embeddings generated!"); sys.exit(1)

    # --- 4. Generate Metadata ---
    print("\n📝 Generating Metadata (Clean Version)...")
    
    metadata = {
        "version": "v1-auto-deploy",
        "description": "Auto-generated from MLOps Pipeline (Minimal Artifact Set)",
        "files": {
            # ✅ มีแค่ 3 ไฟล์หลักสำหรับ Metric Learning
            "student_model.tflite": calculate_md5(MODEL_PATH),
            "pill_db.index": calculate_md5(INDEX_OUT),
            "labels.json": calculate_md5(LABELS_OUT),
        }
    }
    
    with open(META_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata Saved")

    # --- 5. Deploy to S3 (Atomic Upload) ---
    print(f"\n🚀 Uploading Artifacts to S3 ({TARGET_BUCKET})...")
    
    files_to_upload = {
        MODEL_PATH: "student_model.tflite",
        INDEX_OUT: "pill_db.index",
        LABELS_OUT: "labels.json",
        META_OUT: "model_metadata.json"
    }

    success_count = 0
    for local_path, s3_filename in files_to_upload.items():
        if os.path.exists(local_path):
            s3_dest = f"{S3_PREFIX}{s3_filename}"
            # ส่ง Bucket Name ที่ได้จาก .env ไปให้ฟังก์ชัน upload
            if upload_to_s3(local_path, s3_dest, bucket_name=TARGET_BUCKET):
                success_count += 1
        else:
            print(f"⚠️ Missing file: {local_path}")

    if success_count == len(files_to_upload):
        print("\n🎉🎉🎉 MISSION COMPLETE! System Ready on Edge! 🎉🎉🎉")
    else:
        print("\n⚠️ Some files failed to upload.")

if __name__ == "__main__":
    main()