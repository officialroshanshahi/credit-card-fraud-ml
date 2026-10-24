import joblib
import os, glob, pandas as pd
from pathlib import Path
import numpy as np
from src.v1.preprocessing import preprocessor, load_data
from src.v1.monitor import monitor_model
from src.v1.main import retrain
from src.v1.config import TARGET

# getting the root of the project
current_path = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in current_path.parents if p.name == "credit-card-fraud-ml")


# checking if the production dataset exists if it doesn't using test set temporarily for this
def get_all_prod_data(v):
    DIR_VER = PROJECT_ROOT / f"data/new/v{v}"
    files = glob.glob(os.path.join(DIR_VER, "production_data_*.csv"))
    
    if not files:
        print(f"\n[MONITORING INFO] No live production data files found for v{v}.")
        print("[MONITORING INFO] Falling back to Test Data Split as a temporary production proxy...")
        # FIX 1: Pass the dynamic version variable 'v', not hardcoded '1'
        _, Xte, _, yt = load_data(ver=v) 
        
        test_prod_proxy = Xte.copy()
        test_prod_proxy[TARGET] = yt.squeeze()
        return test_prod_proxy
    
    df = pd.concat([pd.read_csv(f, on_bad_lines='skip') for f in files], ignore_index=True)
    
    # ensure TARGET exists so downstream labels tracking doesn't break
    if TARGET not in df.columns:
        df[TARGET] = np.nan
        
    return df



# MAIN monitoring logic
# PLEASE NOTE: this could be added to main.py as this is the logic to init the monitoring orchestration
# but for simplicity this is in here
# PLEASE ALSO NOTE: that since there isn't production dataset to monitor the model 
# we can't be run monitoring properly, please keep that in mind while testing the endpoints
def run_monitoring(model_instances):
    
    model, ver, name = model_instances
    
    prod = get_all_prod_data(ver)
    
    REF = PROJECT_ROOT / f"data/processed/v{ver}/test_baseline.csv"
    
    # synchronized metadata columns to filter out operational logs from features
    meta_cols = {
        TARGET, "request_id", "timestamp", "prediction", "probability", 
        "risk_percentage", "confidence_score", "confidence_percentage", 
        "all_probabilities", "review_tier", "action"
    }
    
    if not REF.exists():
        print(f"[MONITORING INFO] Baseline dataset missing at {REF}. Generating test_baseline.csv on the fly from test set...")
        _, Xte, _, yte = load_data(ver=ver)
        
        raw_te_cols = [c for c in Xte.columns if c not in meta_cols]
        
        x_te, _ = preprocessor(Xte[raw_te_cols], "no", "yes", ver)
        
        baseline = x_te.copy()
        
        baseline = yte.squeeze()
        
        REF.parent.mkdir(parents=True, exist_ok=True)
        
        baseline.to_csv(REF, index=False)
        
        print(f"[MONITORING INFO] Saved test baseline dataset to {REF}")
        
        ref = baseline
    else:
        ref = pd.read_csv(REF)
        
        ref.columns = [c.lower() for c in ref.columns]
    
    if TARGET in ref.columns:
        ref[TARGET] = pd.to_numeric(ref[TARGET], errors="coerce").astype("int64")

    prod.columns = [c.lower() for c in prod.columns]
    
    raw_feature_cols = [
        c for c in prod.columns 
        if c not in meta_cols and prod[c].nunique() > 1
    ]
        
    x_prod, xs_prod = preprocessor(prod[raw_feature_cols], "no", "yes", ver)
    eval_data = xs_prod if any(k in name.lower() for k in ["logistic", "kneighbor", "knc"]) else x_prod
    
    threshold_path = PROJECT_ROOT / f"models/models/v{ver}/{name}_threshold_{ver}.pkl"
    threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5
    
    pred_probas = model.predict_proba(eval_data)
    pred_labels = (pred_probas[:, 1] >= threshold).astype(int) if pred_probas.ndim == 2 and pred_probas.shape[1] >= 2 else np.argmax(pred_probas, axis=1)
    
    labels = None
    if TARGET in prod.columns:
        labels = pd.to_numeric(prod[TARGET], errors="coerce").astype("int64")
    
    current_df = x_prod.copy()
    current_df[TARGET] = labels
    
    result = monitor_model(ref, current_df, pred_labels, pred_probas, labels, ver)
    
    if result.get("retrain_required"):
        print("[MONITORING ALERT] Drift or performance drop detected. Triggering automated retrain...")
        result["retraining"] = retrain(ver)
    else:
        print("[MONITORING INFO] Model performance and data distributions are stable.")
    
    return result

if __name__ == '__main__':
    print("Monitoring runner initialized.")