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

## need to set up in action variable first
AWS_ACCESS_KEY_ID
MLFLOW_TRACKING_URI
AWS_SECRET_ACCESS_KEY
AWS_ACCESS_KEY_ID
