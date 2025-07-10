## 📝 Diversify UCI Pipeline

<img width="415" height="375" alt="image" src="https://github.com/user-attachments/assets/655d037e-024c-4267-a81c-1929408c0a39" />


### 📌 Overview
This repository implements diversified training algorithms for activity recognition using UCI datasets. It focuses on enhancing generalisation by applying algorithmic diversification strategies on top of neural architectures for sensor-based HAR (Human Activity Recognition).
This project, of using diversify algorithm on UCI-HAR dataset, is an extension of the paper OUT-OF-DISTRIBUTION REPRESENTATION LEARNING FOR TIME SERIES CLASSIFICATION at ICLR 2023.

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

<img width="359" height="539" alt="image" src="https://github.com/user-attachments/assets/429253e8-f880-4afb-b919-1ff4430db15b" />



#### 🧪 Evaluation Pipeline
Evaluation is embedded within train.py post each epoch, reporting metrics like accuracy and loss on validation/test splits.

Algorithms and evaluation logic reside in:

alg/algs/diversify.py

alg/modelopera.py (model operations for evaluation)

<img width="361" height="541" alt="image" src="https://github.com/user-attachments/assets/3100b6b4-37b8-4c3c-9d7f-11f62fc7b760" />


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

direct link to download- https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip

Human Activity Recognition Using Smartphones Data Set

Loaded via uci_loader.py with preprocessing and splitting scripts under datautil/actdata/.

---

### ▶️ How to Run

#### Install dependencies


        conda env create -f env.yml
        conda activate diversify
or using pip:

        pip install -r requirements.txt

#### Dataset Download

        # Clean any existing files or folders
        rm -rf data/UCI\ HAR\ Dataset data/UCI-HAR-dataset UCI-HAR.zip
        
        # Recreate the data directory
        mkdir -p data/
        
        # Download the UCI HAR dataset
        wget -O UCI-HAR.zip https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
        
        # Check contents of the ZIP (to confirm folder name inside ZIP)
        unzip -l UCI-HAR.zip | head -n 20
        
        # Extract outer ZIP
        unzip -o UCI-HAR.zip -d data/
        
        # Extract the inner ZIP
        unzip -o "data/UCI HAR Dataset.zip" -d data/
        
        # Rename folder to standard name
        mv "data/UCI HAR Dataset" data/UCI-HAR-dataset
        
        # Verify structure
        ls data/UCI-HAR-dataset
        
#### Train a model

          python train.py --data_dir ./data/UCI-HAR-dataset/ --task cross_people --test_envs 0 --dataset uci_har --algorithm diversify --latent_domain_num 10 --alpha1 1.0 --alpha 1.0 --lam 0.0 --local_epoch 3 --max_epoch 50 --lr 0.01 --output ./data/train_output/act/cross_people-uci_har-Diversify-0-10-1-1-0-3-2-0.01
          
          python train.py --data_dir ./data/UCI-HAR-dataset/ --task cross_people --test_envs 1 --dataset uci_har --algorithm diversify --latent_domain_num 2 --alpha1 0.1 --alpha 10.0 --lam 0.0 --local_epoch 10 --max_epoch 15 --lr 0.01 --output ./data/train_output/act/cross_people-uci_har-Diversify-1-2-0.1-10-0-10-15-0.01
          
          python train.py --data_dir ./data/UCI-HAR-dataset/ --task cross_people --test_envs 2 --dataset uci_har --algorithm diversify --latent_domain_num 20 --alpha1 0.5 --alpha 1.0 --lam 0.0 --local_epoch 1 --max_epoch 150 --lr 0.01 --output ./data/train_output/act/cross_people-uci_har-Diversify-2-20-0.5-1-0-1-150-0.01
          
          python train.py --data_dir ./data/UCI-HAR-dataset/ --task cross_people --test_envs 3 --dataset uci_har --algorithm diversify --latent_domain_num 5 --alpha1 5.0 --alpha 0.1 --lam 0.0 --local_epoch 5 --max_epoch 30 --lr 0.01 --output ./data/train_output/act/cross_people-uci_har-Diversify-3-5-5-0.1-0-5-30-0.01

---

### 🏆 Outputs and Artifacts


                | **Test Environment** | **Target Accuracy** |
                | -------------------- | ------------------- |
                | Test Envis 0         | 93%                 |
                | Test Envis 1         | 88%                 |
                | Test Envis 2         | 90%                 |
                | Test Envis 3         | 92%                 |

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
### Contact

E-mail- rishabhgupta8218@gmail.com

### 📝 License
This project is free for academic and commercial use with attribution.

        @misc{UCI-HAR2025,
          title={Diversify UCI Pipeline},
          author={Rishabh Gupta et al.},
          year={2025},
          note={https://github.com/UCIHARadded/newuci}
        }

---

### 💡 Note
For academic or production use, ensure dataset paths and preprocessing steps in uci_loader.py align with your local data organisation.

Extend algorithms by adding new files under alg/algs/ and registering them in alg.py.
