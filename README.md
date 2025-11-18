# pilltrack-edge-mlops
End-to-end MLOps pipeline for real-time pill identification on Edge devices using Deep Metric Learning, Knowledge Distillation, and Vector Search.

# Create Python environment
conda env create -f environment.yaml
conda activate pilltrack-conda

# update (Please uncomment setup in environment.yaml if you use macOS )
conda env update --file environment.yaml --prune

# Train Model 
dvc repro 

# Push Command
# Add Data
dvc add --force data/pills_dataset_resnet.zip
dvc add data/raw/pills_dataset_resnet/ 

# Run Ci.yaml
dvc status          
dvc push           
git add .          
git commit -m "Add trained models and data"
git push          
