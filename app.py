import numpy as np
from flask import Flask, render_template, request
import joblib
import pandas as pd  

# Load trained model
model = joblib.load('final_xgb_model.pkl')

# Load data files
mapping_df = pd.read_csv("DrugNamee.csv")

# Load the same dataset used during training
twosides = pd.read_csv('TWO_SIDES_50000_multilabel_with_names.csv')

# Rebuild label_cols EXACTLY as done in preprocessing
label_cols = [col for col in twosides.columns if col not in ['drug_1_rxnorn_id', 'drug_1_concept_name', 'drug_2_rxnorm_id', 'drug_2_concept_name']]


# Normalize drug names
mapping_df['Name'] = mapping_df['Name'].str.lower()


# Build name → DrugBank ID mapping
drug_name_to_dbid = mapping_df.set_index('Name')['DrugBank ID'].to_dict()
 
embed_df = pd.read_csv('mock_rdf2vec_embeddings.csv', index_col=0)



# Retrieve DrugBank ID
def get_drugbank_id(drug_name):
    return drug_name_to_dbid.get(drug_name.lower())

# Retrieve embedding vector
def get_embedding_vector(drugbank_id):
    try:
        return embed_df.loc[drugbank_id].values
    except KeyError:
        return np.zeros(128)  # if ID not found, return zero vector

# Prepare combined embedding for two drugs
def get_embedding(drug1, drug2):
    id1 = get_drugbank_id(drug1)
    id2 = get_drugbank_id(drug2)

    if not id1 or not id2:
        return None, id1, id2

    vec1 = get_embedding_vector(id1)
    vec2 = get_embedding_vector(id2)
    combined = np.concatenate([vec1, vec2])
    return combined, id1, id2

# Flask app setup
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    all = []
    error = None
    prediction = None

    if request.method == 'POST':
        drug1 = request.form['drug1']
        drug2 = request.form['drug2']
        X, id1, id2 = get_embedding(drug1, drug2)

        if X is not None:
            X = np.array([X])
            probs = model.predict_proba(X)

            # Extract class-1 probabilities from MultiOutputClassifier
            probabilities = np.array([est[:, 1] if len(est.shape) > 1 else est for est in probs]).T[0]
            
            
            side_effect_probs = [(label_cols[i], float(prob)) for i, prob in enumerate(probabilities)]
            side_effect_probs.sort(key=lambda x: x[1], reverse=True)
            all = side_effect_probs[:10]
            # Add risk level to each tuple
            all = [(name, prob, categorize_risk(prob)) for name, prob in all]

            prediction = f"All Side Effects for {drug1} and {drug2}:"
        else:
            error = f"Drug not found: {drug1 if not id1 else drug2}"

    return render_template("index.html", prediction=prediction, all=all, error=error)

# Categorize risk
def categorize_risk(prob):
    if prob >= 0.8:
        return "High"
    elif prob >= 0.5:
        return "Medium"
    else:
        return "Low"




if __name__ == "__main__":
    app.run(debug=True, port=8000)
