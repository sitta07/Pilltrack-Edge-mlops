# PillTrack: Edge MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![DVC](https://img.shields.io/badge/Data%20Ops-DVC-purple)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![AWS S3](https://img.shields.io/badge/Storage-AWS%20S3-orange)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black)

## Overview

**PillTrack** is an end-to-end MLOps pipeline designed for **real-time pill identification on Edge devices**.

Unlike traditional classification, this system leverages **Deep Metric Learning** to generate robust vector embeddings for pills, allowing for few-shot identification of new pill types without full retraining. To ensure low-latency inference on edge hardware, we utilize **Knowledge Distillation** to compress heavy teacher models (ResNet) into lightweight student models.

## MLOps Architecture

The pipeline follows a reproducible **Data-centric AI** approach using DVC for data versioning and Git for code versioning.

Tech Stack & Engineering Decisions

Data Version Control (DVC):

Decision: Decouples large datasets (.zip) from the codebase while maintaining version history aligned with Git commits. Ensures reproducibility of every experiment.

Deep Metric Learning:

Decision: Used instead of Softmax classification to handle the "open-set" problem (new pills appearing in the future) via Vector Search similarity.

Knowledge Distillation:

Decision: Compresses model size by transferring knowledge from a heavy Teacher network to a lightweight Student network, optimizing for Edge latency constraints.

GitHub Actions (CI/CD):

Decision: Automates the training pipeline (dvc repro) on Pull Requests to ensure model convergence before merging.


Getting Started
1. Environment Setup
Manage dependencies using Conda to ensure cross-platform compatibility.

## Create environment
```bash
conda env create -f environment.yaml
```


## Activate environment
```bash
conda activate pilltrack-conda
```

## (Optional) Update environment if yaml changes
## Note: Uncomment setup in yaml if using macOS Apple Silicon
```bash
conda env update --file environment.yaml --prune
```

2. Configuration (Secrets)
To run the pipeline locally or in CI/CD, ensure the following environment variables are set (e.g., in .env or GitHub Secrets):

```bash
export AWS_ACCESS_KEY_ID="your_key"

export AWS_SECRET_ACCESS_KEY="your_secret"

export AWS_REGION="ap-southeast-1"

export MLFLOW_TRACKING_URI="your_mlflow_server"
```

Development Workflow (Git Flow + DVC)

We follow a strict Feature Branch Workflow. Direct pushes to main are prohibited to maintain pipeline integrity.

Step 1: Start a New Feature

Always create a new branch for model experiments or bug fixes.

```bash
git checkout main
git pull origin main
git checkout -b feature/improved-resnet-backbone
```

Step 2: Reproduce Pipeline & Train

Run the DVC pipeline to execute stages (train, convert, enroll) defined in dvc.yaml.

## Runs the pipeline locally, updates artifacts and dvc.lock
```bash
dvc repro
```

Step 3: Commit & Push Changes
Case A: Code or Hyperparameters Changed ONLY

If you only modified .py files or params.yaml:
## 1. Check status (Ensure dvc.lock is modified)
```bash
dvc status
```

## 2. Push tracked artifacts to S3
```bash
dvc push
```

## 3. Git Commit & Push Code
```bash
git add .
git commit -m "feat: optimize distillation temperature"
git push -u origin feature/improved-resnet-backbone
```

Case B: Dataset Changed

If you updated the raw dataset (e.g., data/pills_dataset_resnet.zip):
## 1. Update DVC tracking
```bash
dvc add --force data/pills_dataset_resnet.zip
```

## 2. Push data to Remote Storage
```bash
dvc push
```

## 3. Commit the pointer file (.dvc) to Git
```bash
git add data/pills_dataset_resnet.zip.dvc .gitignore
git commit -m "chore: update dataset v2 with new pill types"
git push
```

CI/CD Pipeline

On every git push, the CI pipeline executes:

DVC Pull: Fetches data from AWS S3.

Reproduction: Runs dvc repro to validate the training pipeline.

Reporting: Pushes metrics to MLflow and comments results on the PR.


👨‍💻 Author
sitta07


AI Engineer Intern @ AI SmartTech

© 2025 AI SmartTech. All Rights Reserved.
