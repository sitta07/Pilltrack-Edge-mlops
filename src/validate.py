import os
import sys
import json
import faiss
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image
from glob import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

def load_params(param_path="params.yaml"):
    with open(param_path, "r") as f: return yaml.safe_load(f)

# ♻️ Reuse ฟังก์ชันนี้ (ควรแยกไปไว้ใน utils แต่แปะที่นี่เพื่อความครบจบในไฟล์เดียว)
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
    print("🔍 Starting Performance Validation Stage...")
    params = load_params()
    
    # --- Load Configs ---
    TEST_DIR = params['validation']['test_data_path']
    THRESHOLD = params['validation']['accuracy_threshold']
    METRIC_FILE = params['validation']['metric_file']
    
    INDEX_FILE = params['enrollment']['index_file']
    LABELS_FILE = params['enrollment']['labels_file']
    MODEL_PATH = "models/student_quant_int8.tflite" # หรืออ่านจาก params

    # 1. Check Files Existence
    if not (os.path.exists(INDEX_FILE) and os.path.exists(LABELS_FILE) and os.path.exists(MODEL_PATH)):
        print("❌ Error: Missing artifacts (index, labels, or model).")
        sys.exit(1)
        
    if not os.path.exists(TEST_DIR):
        print(f"❌ Error: Test directory not found at {TEST_DIR}")
        sys.exit(1)

    # 2. Load Artifacts
    print("⚙️ Loading resources...")
    index = faiss.read_index(INDEX_FILE)
    with open(LABELS_FILE, 'r') as f:
        labels_map = json.load(f) # key=str(id), value=class_name
        # Convert keys to int for mapping
        labels_map = {int(k): v for k, v in labels_map.items()}

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Run Inference on Test Set
    y_true = []
    y_pred = []
    
    classes = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
    print(f"🧪 Testing on {len(classes)} classes from {TEST_DIR}...")

    for class_name in tqdm(classes, desc="Validating"):
        folder_path = os.path.join(TEST_DIR, class_name)
        images = glob(os.path.join(folder_path, "*.jpg"))
        
        for img_path in images:
            # Get Vector
            query_vec = get_embedding(interpreter, img_path, input_details, output_details)
            
            if query_vec is not None:
                # Search in Faiss (Top-1)
                query_vec = np.expand_dims(query_vec, axis=0).astype('float32')
                distances, indices = index.search(query_vec, k=1)
                
                pred_idx = indices[0][0]
                pred_label = labels_map.get(pred_idx, "Unknown")
                
                y_true.append(class_name)
                y_pred.append(pred_label)

    # 4. Calculate Metrics
    if not y_true:
        print("❌ Error: No test images found or processed.")
        sys.exit(1)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n📊 Validation Results:")
    print(f"   Accuracy: {accuracy:.4f} (Threshold: {THRESHOLD})")
    
    # Save metrics for DVC
    metrics = {"accuracy": accuracy}
    with open(METRIC_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    # 5. Decide Pass/Fail (The Gatekeeper 🛡️)
    if accuracy >= THRESHOLD:
        print("✅ Validation PASSED! Model meets the criteria.")
        sys.exit(0)
    else:
        print(f"❌ Validation FAILED! Accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
        sys.exit(1) # ทำให้ DVC / CI Pipeline หยุดทำงานทันที!

if __name__ == "__main__":
    main()