import argparse
import os
import yaml
import sys
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# folder src/utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import ฟังก์ชันช่วยโหลด S3 
try:
    from src.utils.s3_helper import download_from_s3
except ImportError:
    # Fallback function กรณีหาไฟล์ s3_helper ไม่เจอ (เพื่อให้รันเทสได้)
    print("⚠️ Warning: ไม่พบ src.utils.s3_helper หากไฟล์ Teacher ไม่มีในเครื่อง โปรแกรมอาจหยุดทำงาน")
    def download_from_s3(key, path): return False

# --- 1. Helper Functions ---
def load_params(param_path="params.yaml"):
    """โหลดไฟล์ Config YAML"""
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ {param_path}")
    with open(param_path, "r") as f:
        return yaml.safe_load(f)

def get_device(force_cpu=False):
    """เลือก Device อัตโนมัติ (CUDA > MPS > CPU)"""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # สำหรับ Mac Apple Silicon
    else:
        return torch.device("cpu")

def prepare_data(zip_path, extract_to):
    """แตกไฟล์ Zip อัตโนมัติ"""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ Zip ต้นฉบับที่: {zip_path} (กรุณาเช็ค dvc pull หรือ path ใน params.yaml)")
    
    # ถ้าโฟลเดอร์ปลายทางยังไม่มี ให้แตกไฟล์
    if not os.path.exists(extract_to):
        print(f"📦 Unzipping data from '{zip_path}' to '{extract_to}'...")
        os.makedirs(extract_to, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print("✅ Unzip Complete!")
        except zipfile.BadZipFile:
            print("❌ Error: ไฟล์ Zip เสียหาย")
            sys.exit(1)
    else:
        print(f"✅ Data already extracted at '{extract_to}'")

# --- 2. Model Definitions ---
def get_teacher(num_classes, device, weights_path=None):
    """สร้างและโหลดโมเดล Teacher (ResNet50)"""
    print(f"👨‍🏫 Initializing Teacher (ResNet50) for {num_classes} classes...")
    model = models.resnet50(weights=None) # ไม่ใช้ default weights เพราะเราจะโหลด custom
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if weights_path and os.path.exists(weights_path):
        print(f"   - Loading weights from: {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print("   ✅ Load Teacher weights complete.")
        except Exception as e:
            print(f"   ❌ Error loading weights: {e}")
            sys.exit(1)
    else:
        print("   ⚠️ Warning: No custom weights found for Teacher. Using random initialization.")
    
    model.to(device)
    model.eval() # Teacher ต้องนิ่งเสมอ
    return model

def get_student(num_classes, device):
    """สร้างโมเดล Student (MobileNetV2)"""
    print("👶 Initializing Student (MobileNetV2)...")
    # Student ใช้ Pretrained ImageNet
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(device)
    return model

# --- 3. KD Loss Function ---
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """คำนวณ Loss แบบผสม (Soft Target + Hard Target)"""
    # 1. Soft Loss (เลียนแบบครู)
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    soft_prob = F.log_softmax(student_logits / T, dim=1)
    loss_soft = nn.KLDivLoss(reduction='batchmean')(soft_prob, soft_targets) * (T * T)
    
    # 2. Hard Loss (เรียนจากเฉลยจริง)
    loss_hard = nn.CrossEntropyLoss()(student_logits, labels)
    
    # ผสมกันตามน้ำหนัก Alpha
    return alpha * loss_soft + (1. - alpha) * loss_hard

# --- 4. Validation Function ---
def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return val_loss / len(loader.dataset), 100 * correct / total

# --- 5. Main Execution ---
def main(args):
    # 1. Setup Environment
    params = load_params()
    device = get_device(args.cpu)
    print(f"🚀 Starting Training on Device: {device}")

    # Extract Configs
    TRAIN_CFG = params['train']
    DATA_CFG = params['data']
    MODEL_CFG = params['model']
    
    # 2. Prepare Data (Auto Unzip)
    print("⚙️ Preparing Dataset...")
    prepare_data(
        zip_path=DATA_CFG['source_zip'],
        extract_to=DATA_CFG['extract_path']
    )

    # 3. Prepare Teacher Model (Auto-Download)
    teacher_local_path = MODEL_CFG.get('teacher_local_path', 'models/teacher_resnet50.pth')
    
    if not os.path.exists(teacher_local_path):
        print(f"🔍 Teacher model not found locally at {teacher_local_path}")
        if 'teacher_s3_key' in MODEL_CFG:
            print("☁️ Attempting to download from S3...")
            success = download_from_s3(MODEL_CFG['teacher_s3_key'], teacher_local_path)
            if not success:
                print("❌ Failed to download Teacher. Cannot proceed with KD training.")
                # sys.exit(1) 
        else:
            print("⚠️ No S3 key defined for Teacher.")

    # 4. Load Dataset
    transform = transforms.Compose([
        transforms.Resize((DATA_CFG['img_size'], DATA_CFG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"📂 Loading dataset from: {DATA_CFG['extract_path']}")
    
    # หาโฟลเดอร์ที่มีรูปภาพจริงๆ (รองรับโครงสร้างทั้งแบบมีและไม่มี subfolder)
    dataset_full_path = os.path.join(DATA_CFG['extract_path'], "pills_dataset_resnet", "train")
    if not os.path.exists(dataset_full_path):
        dataset_full_path = DATA_CFG['extract_path'] # Fallback
        
    try:
        full_dataset = datasets.ImageFolder(root=dataset_full_path, transform=transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_CFG['batch_size'], shuffle=True, num_workers=DATA_CFG.get('num_workers', 2))
        val_loader = DataLoader(val_dataset, batch_size=TRAIN_CFG['batch_size'], num_workers=DATA_CFG.get('num_workers', 2))
        
        num_classes = len(full_dataset.classes)
        print(f"✅ Data Loaded: {len(full_dataset)} images, {num_classes} classes.")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"👉 ตรวจสอบว่าใน {dataset_full_path} มีโฟลเดอร์แยกตาม Class หรือไม่")
        sys.exit(1)

    # 5. Initialize Models
    teacher = get_teacher(num_classes, device, teacher_local_path)
    student = get_student(num_classes, device)
    
    optimizer = optim.Adam(student.parameters(), lr=TRAIN_CFG['lr'])
    
    # 6. MLflow Tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

    print("\n🔥 Starting Training Loop...")
    with mlflow.start_run(run_name=args.run_name) as run:
        # Log Params
        mlflow.log_params(TRAIN_CFG)
        mlflow.log_params(MODEL_CFG)
        mlflow.log_param("device", device.type)
        
        best_acc = 0.0
        
        for epoch in range(TRAIN_CFG['epochs']):
            student.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CFG['epochs']}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Get Teacher Logits (No Grad)
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                
                # Get Student Logits
                student_logits = student(inputs)
                
                # Calculate KD Loss
                loss = distillation_loss(
                    student_logits, 
                    teacher_logits, 
                    labels, 
                    TRAIN_CFG['temperature'], 
                    TRAIN_CFG['alpha']
                )
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

            # Validation Phase
            train_loss = running_loss / len(train_loader)
            val_loss, val_acc = validate(student, val_loader, device)
            
            print(f"   📝 Stats: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.2f}%")
            
            # Log Metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)
            
            # Save Best Model
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"   ⭐ New Best Model! ({best_acc:.2f}%) Saving...")
                os.makedirs("models", exist_ok=True)
                torch.save(student.state_dict(), "models/best_student.pth")
        
        # End of Training
        print(f"\n🏆 Training Complete. Best Accuracy: {best_acc:.2f}%")
        
        # Log Best Model Artifact
        if os.path.exists("models/best_student.pth"):
            print("📦 Uploading model artifact to MLflow...")
            mlflow.log_artifact("models/best_student.pth", artifact_path="models")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PillTrackEdge Training Pipeline")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--run_name", type=str, default="CI_Training_Run", help="MLflow Run Name")
    
    args = parser.parse_args()
    main(args)