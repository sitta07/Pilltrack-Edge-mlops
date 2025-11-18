# pilltrack-edge-mlops
End-to-end MLOps pipeline for real-time pill identification on Edge devices using Deep Metric Learning, Knowledge Distillation, and Vector Search.

# Create Python environment
conda env create -f environment.yaml
conda activate pilltrack-conda

# update
conda env update --file environment.yaml --prune

# export Close OpenMP
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=""