# 🧪 Drug–Drug Interaction Side Effect Predictor (Flask App)

This project predicts the **top 10 possible side effects** when two drugs are taken together.  
It uses:

- RDF2Vec embeddings (128-dim each)  
- XGBoost MultiOutputClassifier  
- TwoSides multi-label dataset  
- DrugName → DrugBank ID mapping  
- Flask-based web interface  

The app loads drug embeddings, combines them, runs the trained model, and displays side-effect probabilities with risk levels.

---

## 🚀 Features

- Enter two drug names  
- Convert drug names → DrugBank IDs  
- Load RDF2Vec embeddings  
- Build 256-dim combined vector  
- Predict side-effect probabilities  
- Display top 10 side effects  
- Auto-classify risk: High / Medium / Low  
- Clean and simple Flask UI  

---

## 📂 Project Structure

```
project/
│── app.py
│── requirements.txt
│── final_xgb_model.pkl   ← Download separately (see below)
│── DrugNamee.csv
│── TWO_SIDES_50000_multilabel_with_names.csv
│── mock_rdf2vec_embeddings.csv
│
├── templates/
│     └── index.html
│
└── static/   (optional)
      ├── style.css
      └── script.js
```

---

## 🔽 Download the Trained Model (Required)

Because GitHub does not allow uploading files >25MB directly,  
the trained model `final_xgb_model.pkl` is available in **GitHub Releases**.

👉 **Download here:**  
https://github.com/Logicrithm/Drug-Drug-interaction
/releases/latest

After downloading, **place the file in the project root**, next to:

```
app.py  
DrugNamee.csv  
mock_rdf2vec_embeddings.csv  
```

---

## 📦 Requirements

Minimal dependencies:

```
Flask==3.0.3
pandas==2.2.3
numpy==2.2.4
joblib==1.4.2
scikit-learn==1.6.1
xgboost==3.0.0
```

Save these into **requirements.txt**.

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```
git clone https://github.com/Logicrithm/Drug-Drug-interaction
.git
cd Drug-Drug-interaction

```

### 2️⃣ Create a virtual environment

#### Windows:
```
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux:
```
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Place the downloaded model file
Put `final_xgb_model.pkl` next to `app.py`.

### 5️⃣ Run the Flask app
```
python app.py
```

App runs at:  
👉 **http://127.0.0.1:8000/**

---

## 🧠 How The App Works

### ✔ Step 1 — User enters two drug names  
Example: `"aspirin"` + `"warfarin"`

### ✔ Step 2 — Convert names → DrugBank IDs  
Using mapping file: `DrugNamee.csv`

### ✔ Step 3 — Fetch embeddings  
Vectors loaded from: `mock_rdf2vec_embeddings.csv`

### ✔ Step 4 — Combine embeddings  
256-dim final vector (128 + 128)

### ✔ Step 5 — Model prediction  
XGBoost model predicts probabilities for **all** side effects.

### ✔ Step 6 — Display top 10  
Sorted by probability + risk levels.

---

## 🖼️ UI Screenshot (Add your screenshot)
```
![App Screenshot](screenshots/main_ui.png)
```

---

## 📁 Required Files

| File | Description |
|------|-------------|
| `final_xgb_model.pkl` | Trained XGBoost model (download separately) |
| `DrugNamee.csv` | Drug name → DrugBank ID mapping |
| `TWO_SIDES_50000_multilabel_with_names.csv` | Rebuilds multi-label column list |
| `mock_rdf2vec_embeddings.csv` | 128-dim embeddings for each drug |
| `index.html` | Frontend UI |

---

## 🧪 Example Prediction

| Side Effect | Probability | Risk |
|------------|-------------|------|
| GI Bleeding | 0.92 | High |
| Bruising | 0.76 | Medium |
| Headache | 0.41 | Low |

---

## 🐞 Troubleshooting

### ❌ Drug not found  
Drug name not present in `DrugNamee.csv`.  
Try lowercase or check spelling.

### ❌ Embedding not found  
DrugBank ID missing in `mock_rdf2vec_embeddings.csv`.

---

## 🤝 Contributing
Pull requests are welcome!

---

## 📜 License
MIT License

---

## 🙌 Acknowledgements
- TwoSides Dataset  
- RDF2Vec embedding methodology  
- DrugBank mapping  
- XGBoost MultiOutputClassifier  

