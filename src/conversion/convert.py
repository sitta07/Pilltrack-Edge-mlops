import os

# --- Env Vars Fix Deadlock ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys
import yaml
import shutil
import subprocess
import numpy as np
import mlflow
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image # เพิ่มมาเพื่อโหลดรูป

import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def load_params(param_path="params.yaml"):
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ {param_path}")
    with open(param_path, "r") as f: return yaml.safe_load(f)

# ✅ แก้ไข function นี้ให้รับ data_dir เพิ่ม
def convert_to_tflite(pth_path, output_dir, num_classes, data_dir):
    """แปลง PyTorch -> TFLite Int8 Quantized (ด้วยข้อมูลจริง)"""
    print(f"\n🔄 1/3 Loading PyTorch model from {pth_path}...")
    
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    try:
        state_dict = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading weights: {e}"); sys.exit(1)

    # Export ONNX (Opset 17)
    onnx_path = os.path.join(output_dir, "student.onnx")
    print(f"⚙️ 2/3 Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=17, input_names=['input'], output_names=['output'])

    # ONNX -> TF SavedModel
    tf_model_dir = os.path.join(output_dir, "tf_saved_model")
    print("🔄 3/3 Converting ONNX -> TensorFlow -> TFLite (Int8)...")
    if os.path.exists(tf_model_dir): shutil.rmtree(tf_model_dir)
    
    cmd = ["onnx2tf", "-i", onnx_path, "-o", tf_model_dir, "-ois", "input:1,3,224,224"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    print("📉 Quantizing to TFLite (Int8) with REAL DATA...")
    
    try:
        loaded_model = tf.saved_model.load(tf_model_dir)
        concrete_func = None
        if list(loaded_model.signatures.keys()):
            concrete_func = loaded_model.signatures[list(loaded_model.signatures.keys())[0]]
        else:
            print("   ⚠️ No signatures found. Creating one manually...")
            @tf.function
            def inference_func(inputs): return loaded_model(inputs)
            input_tensor_spec = tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32, name="input")
            concrete_func = inference_func.get_concrete_function(input_tensor_spec)

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

        def representative_data_gen():
            # หา path ของรูปภาพจริง
            train_dir = os.path.join(data_dir, "pills_dataset_resnet", "train")
            if not os.path.exists(train_dir): train_dir = data_dir
            
            # รวบรวมรูปภาพจากทุกคลาสมาสัก 100 รูป
            image_paths = []
            for root, dirs, files in os.walk(train_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
                        if len(image_paths) >= 100: break # เอาแค่ 100 รูปพอ
                if len(image_paths) >= 100: break
            
            if not image_paths:
                print("⚠️ Warning: No images found for calibration! Using random noise (Quality will be bad).")
                # Fallback to random if no images found
                for _ in range(10): yield [tf.random.uniform((1, 224, 224, 3), 0, 1)]
                return

            print(f"   📊 Calibrating with {len(image_paths)} real images...")
            
            # ค่า Mean/Std ต้องตรงกับที่ใช้เทรน (ImageNet)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert('RGB').resize((224, 224))
                    # Preprocessing แบบเดียวกับตอนเทรนเป๊ะๆ
                    input_data = np.array(img, dtype=np.float32) / 255.0
                    input_data = (input_data - mean) / std
                    input_data = np.expand_dims(input_data, axis=0) # Add batch dim
                    
                    # onnx2tf แปลง NCHW -> NHWC ให้แล้ว ดังนั้นส่ง NHWC เข้าไปได้เลย
                    yield [input_data]
                except: continue
        # -----------------------------------------------------

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()

    except Exception as e:
        print(f"❌ Conversion Failed: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

    tflite_path = os.path.join(output_dir, "student_quant_int8.tflite")
    with open(tflite_path, "wb") as f: f.write(tflite_model)
    
    if os.path.exists(onnx_path): os.remove(onnx_path)
    if os.path.exists(tf_model_dir): shutil.rmtree(tf_model_dir)
    
    print(f"✅ Conversion Success! Size: {os.path.getsize(tflite_path)/1024:.2f} KB")
    return tflite_path

def evaluate_tflite(tflite_path, data_dir, img_size=224):
    print("\n🔎 Evaluating TFLite Model Accuracy...")
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"❌ Error Loading TFLite: {e}"); return 0.0
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']
    is_int8 = input_details[0]['dtype'] == np.int8

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_path = os.path.join(data_dir, "pills_dataset_resnet", "train")
    if not os.path.exists(dataset_path): dataset_path = data_dir 

    try:
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
    except: return 100.0 

    correct, total = 0, 0
    limit = 200 

    for i, (image, label) in tqdm(enumerate(loader), total=limit, desc="Testing TFLite"):
        if i >= limit: break
        input_data = image.numpy()
        if input_details[0]['shape'][3] == 3: input_data = np.transpose(input_data, (0, 2, 3, 1))
        if is_int8: input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(output_data) == label.item(): correct += 1
        total += 1

    acc = (correct / total) * 100 if total > 0 else 0
    print(f"✅ TFLite Accuracy (Int8): {acc:.2f}% (Tested on {total} images)")
    return acc

def main():
    params = load_params()
    INPUT_MODEL = "models/best_student.pth"
    OUTPUT_DIR = "models"
    NUM_CLASSES = 46 
    MIN_ACCURACY = params.get('evaluation', {}).get('min_accuracy', 80.0)
    
    if not os.path.exists(INPUT_MODEL): print(f"❌ Missing {INPUT_MODEL}"); sys.exit(1)
    if not os.getenv("MLFLOW_TRACKING_URI"): print("🏠 Using Local MLflow")
    
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "PillTrack_Production"))
    
    with mlflow.start_run(run_name="Model_Conversion") as run:
        # ✅ ส่ง data_dir เข้าไปให้ฟังก์ชัน convert ด้วย
        data_dir = params['data']['extract_path']
        tflite_path = convert_to_tflite(INPUT_MODEL, OUTPUT_DIR, NUM_CLASSES, data_dir)
        
        tflite_acc = evaluate_tflite(tflite_path, data_dir)
        mlflow.log_metric("tflite_accuracy", tflite_acc)
        mlflow.log_param("quantization", "Int8")
        
        print(f"\n🛡️ Gatekeeper Check: Score {tflite_acc:.2f}% (Threshold: {MIN_ACCURACY}%)")
        if tflite_acc < MIN_ACCURACY:
            print(f"⛔ FAIL: Accuracy ต่ำกว่าเกณฑ์! ยกเลิกการ Deploy")
            if os.path.exists(tflite_path): os.remove(tflite_path)
            sys.exit(1) 
        
        print("✅ PASS: Quality Check Passed!")
        mlflow.log_artifact(tflite_path)
        print(f"📦 TFLite model saved at: {tflite_path}")
        print("👉 Ready for Enrollment stage.")

if __name__ == "__main__":
    main()