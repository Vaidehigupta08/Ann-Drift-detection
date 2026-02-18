# Ann-Drift-detection
# 🔍 Autoencoder Based Data Drift Detection (ANN + FastAPI + Streamlit)

This project detects **data drift** using an **Autoencoder (Artificial Neural Network)** trained on historical (normal) data.  
It supports:
- ✅ Model training
- ✅ Drift threshold calculation
- ✅ FastAPI backend (API)
- ✅ Streamlit frontend (UI)

---

## 🧠 Core Idea (Simple Language)

1. Autoencoder is trained only on **normal / historical data**
2. Model learns to **reconstruct normal data**
3. New data is passed to the same model
4. If reconstruction error is **high**, data drift exists
5. Drift is quantified using **Drift Ratio**

---

## 🏗️ Model Architecture (ANN Autoencoder)

Input Layer
↓
Dense (bottleneck_dim × 2) → ReLU
↓
Dense (bottleneck_dim) → ReLU ← Bottleneck
↓
Dense (bottleneck_dim × 2) → ReLU
↓
Dense (input_dim) → Linear


- Total layers: **4 Dense layers**
- Loss: **Mean Squared Error (MSE)**
- Optimizer: **Adam**
- This architecture is sufficient for tabular data

---

## 📁 Project Folder Structure

Ann Drift Detection Project/
│
├── model/
│ ├── autoencoder_drift_model.h5
│ ├── scaler.pkl
│ ├── feature_columns.pkl
│ └── drift_threshold.json
│
├── train_model.py # Model training + threshold creation
├── main.py # FastAPI backend
├── app.py # Streamlit frontend
└── README.md


---

