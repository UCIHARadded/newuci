## 📝 Diversify UCI Pipeline

### 📌 Overview
This repository implements diversified training algorithms for activity recognition using UCI datasets. It focuses on enhancing generalisation by applying algorithmic diversification strategies on top of neural architectures for sensor-based HAR (Human Activity Recognition).
This project of using diversify algorithm on UCI-HAR dataset is being an extension of this paper OUT-OF-DISTRIBUTION REPRESENTATION LEARNING FOR TIME SERIES CLASSIFICATION at ICLR 2023.

Link- https://paperswithcode.com/paper/generalized-representations-learning-for-time

---

### ⚙️ Core Pipelines

#### 🔧 Training Pipeline (train.py)

Configuration Parsing: Reads parameters (dataset paths, algorithm choice, hyperparameters).

Dataset Loading: Uses uci_loader.py and datautil to prepare train-test splits.

Model Initialization: Loads models from alg/algs, e.g., diversified networks.

Optimizer & Loss Setup: Uses custom losses in loss/common_loss.py.

Training Loop: Epoch-wise training with evaluation hooks.

Diversification: If enabled, integrates diversity-promoting regularisation.

#### 🧪 Evaluation Pipeline
Evaluation is embedded within train.py post each epoch, reporting metrics like accuracy and loss on validation/test splits.

Algorithms and evaluation logic reside in:

alg/algs/diversify.py

alg/modelopera.py (model operations for evaluation)

---

### ✨ Key Features
✅ Diversification algorithms for robust learning
✅ Modular dataset loader supporting UCI HAR data
✅ Extensible model operations and optimizer integrations
✅ PyTorch-based implementation for rapid experimentation
✅ Structured utility functions for reproducibility

---

### 📁 File Structure

              diversify/
              │
              ├── alg/                     # Algorithms and model operations
              │   ├── alg.py
              │   ├── modelopera.py
              │   ├── opt.py
              │   └── algs/
              │       ├── base.py
              │       └── diversify.py
              │
              ├── datautil/                # Dataset loaders and utilities
              │   ├── getdataloader_single.py
              │   └── actdata/
              │       ├── cross_people.py
              │       └── util.py
              │
              ├── loss/                    # Loss functions
              │   └── common_loss.py
              │
              ├── network/                 # Model architectures (details not listed)
              │
              ├── utils/                   # Miscellaneous utilities
              │
              ├── train.py                 # Main training entrypoint
              ├── uci_loader.py            # UCI dataset loader
              ├── requirements.txt         # Python dependencies
              └── env.yml                  # Conda environment file

---

### 📊 Dataset Supported
UCI HAR Dataset

Human Activity Recognition Using Smartphones Data Set

Loaded via uci_loader.py with preprocessing and splitting scripts under datautil/actdata/.

---

### ▶️ How to Run
Install dependencies

bash
Copy
Edit
conda env create -f env.yml
conda activate diversify
or using pip:

bash
Copy
Edit
pip install -r requirements.txt
Train a model

bash
Copy
Edit
python train.py --dataset <dataset_path> --alg diversify --epochs 100 --batch_size 64
Replace <dataset_path> with your local path to UCI HAR dataset.

---

### 🏆 Outputs and Artifacts
Model checkpoints: Saved periodically (if implemented in train.py).

Training logs: Epoch-wise accuracy, loss, and evaluation metrics.

Diversification reports: Metrics showing the effect of diversification (implementation dependent in alg/algs/diversify.py).

---

### 🔬 Analysis (In-Depth)
The diversification strategy introduces regularisation losses that encourage diverse feature learning, improving generalisation across subjects.

Uses cross-validation splits (cross_people.py) to benchmark robustness.

Model operations (modelopera.py) implement forward, backward, and evaluation utilities for streamlined experimentation.

Supports algorithmic extension via base classes in alg/algs/base.py, making it straightforward to integrate novel diversification methods.

---

### 📝 License
This repository is currently missing an explicit license file. Please add a LICENSE file (e.g. MIT, Apache 2.0) to define usage terms. Without it, the code remains copyrighted and re-use is restricted.

---

### 💡 Note
For academic or production use, ensure dataset paths and preprocessing steps in uci_loader.py align with your local data organisation.

Extend algorithms by adding new files under alg/algs/ and registering them in alg.py.
