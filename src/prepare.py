# src/prepare.py
import os
import zipfile
import yaml
import sys

def load_params(param_path="params.yaml"):
    with open(param_path, "r") as f: return yaml.safe_load(f)

def main():
    params = load_params()
    
    
    
    zip_path = "data/pills_dataset_resnet.zip" # หรืออ่านจาก params
    extract_to = "data/raw" # หรืออ่านจาก params
    
    print(f"🔨 Preparing data: Unzipping {zip_path} -> {extract_to}")

    if not os.path.exists(zip_path):
        print(f"❌ Error: Zip file not found at {zip_path}")
        sys.exit(1)

    # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    print("✅ Data extraction complete.")

if __name__ == "__main__":
    main()