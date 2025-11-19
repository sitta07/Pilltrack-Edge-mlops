# pilltrack-edge-mlops
End-to-end MLOps pipeline for real-time pill identification on Edge devices using Deep Metric Learning, Knowledge Distillation, and Vector Search.

## Create Python environment
conda env create -f environment.yaml
conda activate pilltrack-conda

## update (Please uncomment setup in environment.yaml if you use macOS )
conda env update --file environment.yaml --prune

## Train Model 
dvc repro 

## Push Command
## Add Data
dvc add --force data/pills_dataset_resnet.zip
dvc add data/raw/pills_dataset_resnet/ 

## Run CI.yaml
dvc status          
dvc push           
git add .          
git commit -m "Add trained models and data"
git push          


## 2. Development Workflow (Git Flow)
## Always create a new branch for new features or fixes. Do not push directly to main.

## Step 1: Create a new branch

git checkout main
git pull origin main

# 2. สร้าง Branch ใหม่ (ตั้งชื่อให้สื่อความหมาย เช่น feature/new-model, fix/validation-bug)
git checkout -b feature/your-feature-name

## 3. Train & Reproduce Pipeline
## Run the DVC pipeline to train, convert, and enroll (Build Artifacts):

dvc repro
## (Note: This runs locally and updates dvc.lock, but does NOT deploy to S3)

## 4. Commit & Push Changes
## Once your experiment is successful locally:

## Case A: Code or Params Changed ONLY (Normal case)

## 1. Check status (Ensure dvc.lock is modified)
dvc status

## 2. Push large files to S3 (if any changes in dvc tracked files)
dvc push

## 3. Git Commit & Push
git add .
git commit -m "feat: Describe what you changed"
git push -u origin feature/your-feature-name


## Case B: Dataset/Raw Data Changed (Only if you updated the .zip file)
dvc add --force data/pills_dataset_resnet.zip
dvc push
git add data/pills_dataset_resnet.zip.dvc .gitignore
git commit -m "chore: Update dataset version"


## need to set up in action variable first
AWS_ACCESS_KEY_ID
MLFLOW_TRACKING_URI
AWS_SECRET_ACCESS_KEY
AWS_ACCESS_KEY_ID
