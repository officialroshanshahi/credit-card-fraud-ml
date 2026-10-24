import json
import joblib
import pandas as pd
import numpy as np
import uuid
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.v1.preprocessing import load_data, preprocessor
from src.v1.train import trainer
from src.v1.evaluate import evaluator
from src.v1.config import (
    TARGET, MIN_RECALL, BASE, ACTIVE, MODELS, 
    PARAMS, SCORES, PROD, ver, MIN_LABELS
)

# getting project root: two directory behind the current one
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# independant function for handling path resolvement which is unnecessary but this way it is easier
def get_resolved_path(path_str: str) -> Path:
    path_obj = Path(path_str)
    return path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj).resolve()


# MAIN TRAINING PIPELINE
def train_pipeline(ver=1, retrain='no', data=None):
    
    if retrain == 'yes' and data is not None:
        # for retraining
        X = data.drop(columns=[TARGET])
        y = data[TARGET]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    else:
        Xtr, Xte, ytr, yte = load_data(ver=ver, base_only='no')

    X_train_processed, X_train_std = preprocessor(Xtr, "yes", "yes", ver)
    X_test_processed, X_test_std = preprocessor(Xte, "no", "yes", ver)

    train_output = trainer(X_train_processed, X_train_std, ytr, MODELS, PARAMS, SCORES, ver)
    
    if isinstance(train_output, tuple):
        models, _ = train_output
    else:
        models = train_output

    best_model_name, best_info, _ = evaluator(X_test_processed, X_test_std, yte, models, plot='yes', ver=ver)
    
    test_metrics = best_info['test_metrics']

    result = {
        "version": int(ver),
        "model_name": best_model_name,
        "accuracy": None if pd.isna(test_metrics["accuracy"]) else float(test_metrics["accuracy"]),
        "f1": None if pd.isna(test_metrics["f1"]) else float(test_metrics["f1"]),
        "val_recall": None if pd.isna(best_info["val_recall"]) else float(best_info["val_recall"]),
        "roc_auc": None if pd.isna(test_metrics["roc_auc"]) else float(test_metrics["roc_auc"]),
        "precision": None if pd.isna(test_metrics["precision"]) else float(test_metrics["precision"]),
        "recall": None if pd.isna(test_metrics["recall"]) else float(test_metrics["recall"])
    }

    # checking if the model is performing as required
    if float(result["recall"]) < MIN_RECALL: # currently 0.75
        print(f"Model rejected: Recall {result['recall']:.3f} < {MIN_RECALL}")
        return {"status": "model didn't cross the threshold", **result}

    print(f"Model trained: Recall {result['recall']:.3f} >= {MIN_RECALL}")

    # for saving the required ACTIVE_MODEL.JSON
    active_path = get_resolved_path(ACTIVE)
    active_path.parent.mkdir(parents=True, exist_ok=True)

    with open(active_path, "w") as f:
        json.dump(result, f, indent=2)

    return {"status": "model is completely trained and functional", **result}


# MAIN RETRAINING LOGIC
def retrain(ver=1):
    prod_p = get_resolved_path(PROD) # necessary
    base_p = get_resolved_path(BASE) # our current dataset
    active_p = get_resolved_path(ACTIVE) # current active model

    # checking if production dataset exists
    if not prod_p.exists():
        return {"status": "pending", "reason": f"Missing file: {prod_p}"}

    # getting the dataset
    prod = pd.read_csv(prod_p).dropna(subset=[TARGET])
    
    # if not labled
    if TARGET not in prod.columns:
        return {"status": "pending", "reason": "No verified labels"}
    
    # if there are not required number of labels: currently 100
    if len(prod) < MIN_LABELS:
        return {"status": "pending", "reason": f"Need {MIN_LABELS} labels; got {len(prod)}"}

    # only getting columns we need
    cols = [c for c in pd.read_csv(base_p).columns if c != TARGET]
    valid = [c for c in cols + [TARGET] if c in prod.columns]

    old_ver = ver
    
    # reading the current active model
    if active_p.exists():
        with open(active_p, "r") as f:
            old_ver = json.load(f).get("version", 1)

    # initialising the training pipeline and of course, sending the required dataset
    result = train_pipeline(ver=old_ver + 1, retrain='yes', data=prod[valid])

    # checking if the model performance is as required
    if result.get("recall", 0) < MIN_RECALL:
        return {"status": "rejected", "reason": f"Recall {result.get('recall', 0):.3f} < {MIN_RECALL}", **result}

    print(f"New model promoted: Recall {result['recall']:.3f} >= {MIN_RECALL}")
    
    return {"status": "promoted", **result}


# MAIN PREDICTION PIPELINE
def run_prediction(data: dict, m_inst, high_risk_threshold: float = 0.85):
    
    model, ver, name = m_inst
    
    is_scaled_model = any(k in name.lower() for k in ["logisticregression", "kneighbor", "knclassifier"])
    
    if is_scaled_model:
        _, eval_data = preprocessor(pd.DataFrame([data]), "no", "yes", ver)
    else:
        eval_data = preprocessor(pd.DataFrame([data]), "no", "no", ver)
    
    probabilities = model.predict_proba(eval_data)[0] 
    
    # for threshold artifacts
    threshold_path = PROJECT_ROOT / f"models/models/v{ver}/{name}_threshold_{ver}.pkl"
    
    threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5
    
    if len(probabilities) == 2:
        highest_prob = float(probabilities[1])
    else:
        predicted_class = int(np.argmax(probabilities))
        highest_prob = float(probabilities[predicted_class])
        
    predicted_class = int(highest_prob >= threshold)
    prob_percentage = round(highest_prob * 100, 2)
    
    # review tier as this model has accuray of 60 to 70%
    # this makes it easier for our investigative team
    if highest_prob >= high_risk_threshold:
        review_tier = "High Risk"
        action = "AUTO_BLOCK"
    elif highest_prob >= threshold:
        review_tier = "Medium Risk"
        action = "MANUAL_REVIEW"
    else:
        review_tier = "Low Risk"
        action = "APPROVE"
    
    
    inverse_label_map = {0: "Legitimate", 1: "Fraud"}
    
    predicted_label = inverse_label_map.get(predicted_class, str(predicted_class))
    
    dt = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    # for saving the production dataset
    filepath = PROJECT_ROOT / f"data/new/v{ver}/production_data_{dt}.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    base_row = {
        "request_id": str(uuid.uuid4()),
        "timestamp": pd.Timestamp.now(),
        "prediction": predicted_label,
        "confidence_score": highest_prob,
        "confidence_percentage": f"{prob_percentage}%",
        "review_tier": review_tier,
        "action": action,
        "all_probabilities": str(probabilities.tolist())
    }
    
    for k, val in data.items():
        base_row[k.lower()] = val
        
    df_row = pd.DataFrame([base_row])
    
    if filepath.exists(): 
        df_row = df_row.reindex(columns=pd.read_csv(filepath, nrows=0).columns, fill_value="")
    
    # saving the dataset
    df_row.to_csv(filepath, mode="a", header=not filepath.exists(), index=False, quoting=1)
    
    return {
        "request_id": base_row["request_id"],
        "prediction": predicted_label,
        "confidence_score": highest_prob,
        "confidence_percentage": f"{prob_percentage}%",
        "review_tier": review_tier,
        "action": action,
        "model_version": ver,
        "model": name,
        "applied_threshold": threshold,
        "high_risk_threshold": high_risk_threshold
    }

if __name__ == '__main__':
    result = train_pipeline(ver=1, retrain='no', data=None)