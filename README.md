# Deep Learning-Based Fertilizer Recommendation for Sustainable Crop Production

An end-to-end Deep Learning application designed for precision agriculture and sustainable crop management. The system accepts environmental parameters, soil characteristics, and crop requirements to accurately recommend the optimal fertilizer using a feed-forward Artificial Neural Network (ANN / MLP).

---

## 📌 1. Project Overview

* **Project Title**: Deep Learning-Based Fertilizer Recommendation for Sustainable Crop Production
* **Problem**: Over-fertilization and improper nutrient application degrade soil health, contaminate groundwater, and reduce crop yields. 
* **Solution**: A Deep Learning classification pipeline that maps multi-variable soil and weather metrics directly to the most effective fertilizer formulation.
* **Architecture**: Input Preprocessing Pipeline $\rightarrow$ Dense (64, ReLU) $\rightarrow$ Dropout (0.2) $\rightarrow$ Dense (32, ReLU) $\rightarrow$ Dense (7, Softmax).

---

## 📊 2. Dataset Information

* **Dataset Name**: Fertilizer Prediction Dataset
* **Source**: Publicly available agricultural benchmark dataset hosted on Kaggle and GitHub ([Suryam-2023/Crop-and-Fertilizer-Recommendation-System](https://github.com/Suryam-2023/Crop-and-Fertilizer-Recommendation-System)).
* **Location in Project**: `dataset/Fertilizer_Prediction.csv`
* **Input Features (8)**:
  1. `Temperature` (°C): Ambient environmental temperature.
  2. `Humidity` (%): Ambient relative humidity.
  3. `Moisture` (%): Soil moisture content.
  4. `Soil Type` (Categorical): `Loamy`, `Sandy`, `Clayey`, `Black`, `Red`.
  5. `Crop Type` (Categorical): `Sugarcane`, `Cotton`, `Millets`, `Paddy`, `Pulses`, `Wheat`, `Tobacco`, `Barley`, `Oil seeds`, `Ground Nuts`, `Maize`.
  6. `Nitrogen (N)` (mg/kg): Soil nitrogen level.
  7. `Potassium (K)` (mg/kg): Soil potassium level.
  8. `Phosphorous (P)` (mg/kg): Soil phosphorus level.
* **Target Classes (7)**:
  * `Urea` (High Nitrogen: 46% N)
  * `DAP` (Diammonium Phosphate: 18% N, 46% P)
  * `14-35-14` (High Phosphorus NPK complex)
  * `28-28` (Equal High Nitrogen & Phosphorus formulation)
  * `17-17-17` (Balanced NPK fertilizer)
  * `20-20` (Nitrogen-Phosphorus complex + Sulphur)
  * `10-26-26` (High Phosphorus & Potassium complex)

---

## 🏗️ 3. Project Folder Structure

```
fertilizer-recommendation/
│
├── dataset/
│   └── Fertilizer_Prediction.csv       # Cleaned benchmark dataset
│
├── models/
│   ├── fertilizer_ann_model.keras      # Trained Keras Deep Learning ANN model
│   ├── preprocessor.joblib             # Scikit-learn ColumnTransformer (Scaler + OneHotEncoder)
│   ├── label_encoder.joblib            # Scikit-learn LabelEncoder for target classes
│   └── training_evaluation.png         # Saved accuracy, loss curves & confusion matrix
│
├── train_model.py                      # Complete model training & evaluation script
├── predict.py                          # Standalone prediction script (CLI & module)
├── app.py                              # Interactive Streamlit Web Application
├── requirements.txt                    # Project Python dependencies
└── README.md                           # Documentation & execution guide
```

---

## ⚙️ 4. Technical Pipeline & Preprocessing Rationale

1. **Handling Whitespace & Inconsistent Column Names**:
   - Strips leading/trailing spaces from headers (e.g., `'Humidity '` $\rightarrow$ `'Humidity'`) and standardizes `'Temparature'` to `'Temperature'`.
2. **Missing Value Imputation**:
   - Ensures zero null records to prevent computation errors during neural network backpropagation.
3. **Categorical Feature Encoding (`OneHotEncoder`)**:
   - Categorical fields (`Soil Type`, `Crop Type`) lack inherent ordinal order. One-hot encoding creates binary indicators without imposing false mathematical hierarchies.
4. **Numerical Feature Scaling (`StandardScaler`)**:
   - Standardizes continuous metrics (Temperature, Humidity, Moisture, N, K, P) to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$), ensuring balanced gradient updates across features with vastly different scales.
5. **Target Encoding (`LabelEncoder`)**:
   - Converts discrete string labels into integers ($0 \dots 6$) for training with `sparse_categorical_crossentropy`.
6. **Stratified Train-Test Split (80/20)**:
   - Preserves balanced class proportions across training and validation holdout sets.

---

## 🧠 5. Deep Learning Model Design Rationale

* **Dense Layer 1 (64 units, ReLU)**: Expands input feature representation into higher-dimensional non-linear feature space.
* **Dropout (0.2)**: Randomly drops 20% of activations during training to prevent co-adaptation and mitigate overfitting on small datasets.
* **Dense Layer 2 (32 units, ReLU)**: Condenses features into high-level representations.
* **Output Layer (7 units, Softmax)**: Normalizes logits into probability distributions $\sum P(y_i) = 1.0$.
* **Loss (`sparse_categorical_crossentropy`)**: Numerically stable multi-class loss function.
* **Optimizer (`Adam`, lr=0.01)**: Adaptive moment estimation provides fast convergence with momentum.
* **Epochs (60) & Batch Size (8)**: Small mini-batch size allows frequent gradient updates and optimal generalization.

---

## 🚀 6. Step-by-Step Running Instructions

### Step 6.1: Set up Virtual Environment

**On Windows (PowerShell / Command Prompt):**
```bash
# Navigate to project folder
cd c:\deep_learning_project

# Create virtual environment named 'venv'
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
```

### Step 6.2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6.3: Train the Deep Learning Model

```bash
python train_model.py
```
*Output*:
- Prints dataset summary and class distribution.
- Displays neural network architecture.
- Trains over 60 epochs.
- Outputs final Test Accuracy (typically $\ge 95\%$) and detailed Classification Report.
- Generates `models/training_evaluation.png`, `models/fertilizer_ann_model.keras`, `models/preprocessor.joblib`, and `models/label_encoder.joblib`.

### Step 6.4: Test Predictions via CLI

**Run Automated Demo Cases:**
```bash
python predict.py
```

**Run Interactive Command-Line Input:**
```bash
python predict.py --interactive
```

### Step 6.5: Launch the Streamlit Web Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the interactive dashboard.

---

## 🛠️ 7. Troubleshooting Common Errors

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'tensorflow'` | Dependencies not installed in active environment | Run `pip install -r requirements.txt` inside the activated virtual environment. |
| `FileNotFoundError: Trained model artifacts not found` | Inference run before training | Run `python train_model.py` first to generate the `.keras` and `.joblib` files in `models/`. |
| `ValueError: Found unknown categories ... in column 0` | Input soil/crop type does not match training vocabulary | Ensure input values match supported categories listed in `predict.py` or dropdowns in `app.py`. |
| `TensorFlow oneDNN custom operations notice` | Informational log from CPU optimization | Normal log message. Set environment variable `$env:TF_ENABLE_ONEDNN_OPTS="0"` if you wish to silence it. |

---

## 📜 8. License & Academic Attribution
Created for Deep Learning PBL coursework in precision agriculture and sustainable crop production.
