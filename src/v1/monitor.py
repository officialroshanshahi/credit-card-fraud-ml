import os, numpy as np, pandas as pd
from datetime import datetime
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, recall_score
from evidently import Report
from evidently.presets import DataDriftPreset
from src.v1.config import TARGET, PSI_LIMIT, KS_LIMIT, MISSING_LIMIT, PRED_LIMIT, TARGET_LIMIT, MIN_RECALL

# for figuring out if the populartion has shifted i.e. data drift
def psi(ref, cur, bins=10):
    # for creating bins
    e = np.unique(np.percentile(ref, np.linspace(0, 100, bins + 1)))
    
    # fallback for data's lack of variance
    if len(e) < 3:
        return 0
    
    # for calculating dataset falling into current dataset, if none fallbacks to 0.0001 (to avoid dividing by zero)
    a = np.clip(np.histogram(cur, e)[0] / len(cur), .0001, None)
    
    # for reference dataset
    b = np.clip(np.histogram(ref, e)[0] / len(ref), .0001, None)
    
    # population stability index sum((act-exp)*log(act/exp)
    return float(np.sum((a - b) * np.log(a / b)))


# main monitoring logics
def monitor_model(reference, current, predictions=None, pred_probas=None, labels=None, version=1):
    alerts = []
    
    # removing unncessary columns
    ignore_cols = {'request_id', 'timestamp', TARGET, 'prediction', 'probability', 'risk_percentage', 'confidence_score', 'confidence_percentage', 'all_probabilities', 'test_results'}
    cols = [c for c in reference if c in current and c not in ignore_cols]
    ref, cur = reference[cols], current[cols]
    
    # samping the dataset both ref and cur with 20K being upper limit
    sample_size = min(len(ref), len(cur), 20000)
    
    ref_sampled = ref.sample(n=sample_size, random_state=42)
    cur_sampled = cur.sample(n=sample_size, random_state=42)

    # for cols we are interested in
    for x in cols:
        
        # missing dataset
        miss = cur_sampled[x].isna().mean()
        
        if miss > MISSING_LIMIT:
            alerts.append(f"Missing: {x} {miss:.1%}")
        
        # for skipping datas not numeric i.e. cat, text, etc.
        if not pd.api.types.is_numeric_dtype(ref_sampled[x]):
            continue
        
        # for removing NaNs from dataset and skipping the dataset itself
        # if the number of values is less than 20
        a, b = ref_sampled[x].dropna(), cur_sampled[x].dropna()
        if len(a) < 20 or len(b) < 20: 
            continue
        
        # perform two sample Kolmogorov-Smirnov (K-S) test and getting the pvalue
        p = ks_2samp(a, b).pvalue
        
        s = psi(ref_sampled[x], cur_sampled[x])
        
        if s > PSI_LIMIT: # currently 0.2
            alerts.append(f"PSI drift: {x} {s:.3f}")
            
        elif p < KS_LIMIT: # currently 0.05
            if x in ["hour_sin", "hour_cos"]:
                print(f"[MONITORING INFO] Cyclical time drift noted on {x} (p={p:.4f})")
            else:
                print(f"[MONITORING WARNING] Significant KS drift detected on {x} (p={p:.4f}).")
                alerts.append(f"KS drift: {x} (p={p:.4f})")

    # making sure the prediction exists and if does checking prediction shift
    if predictions is not None and "prediction" in reference:
        
        ref_counts = pd.Series(reference["prediction"]).value_counts(normalize=True)
        
        cur_counts = pd.Series(predictions).value_counts(normalize=True)
        
        all_classes = set(ref_counts.index).union(set(cur_counts.index))
        
        d = max([abs(ref_counts.get(c, 0) - cur_counts.get(c, 0)) for c in all_classes])
        
        if d > PRED_LIMIT: # currently 0.1
            alerts.append(f"Prediction drift: {d:.2%}")

    has_valid_labels = labels is not None and not pd.Series(labels).isna().all()
    recall_score_val = None
    
    # checking recall on the new dataset
    if has_valid_labels and predictions is not None:
        try:
            valid_mask = ~pd.Series(labels).isna()
            
            y_true = np.array(labels)[valid_mask]
            
            y_pred = np.array(predictions)[valid_mask]
            
            recall_score_val = recall_score(y_true, y_pred, zero_division=0)
            
            if recall_score_val < MIN_RECALL: # currently 0.75
                alerts.append(f"Recall drop: {recall_score_val:.3f}")
                
        except Exception as e:
            print(f"Recall calculation warning during monitoring: {e}")

    # check for target/label drift using TARGET_LIMIT
    if has_valid_labels and TARGET in reference:
        
        valid_labels = pd.Series(labels).dropna()
        
        ref_target_counts = pd.Series(reference[TARGET]).value_counts(normalize=True)
        
        cur_target_counts = valid_labels.value_counts(normalize=True)
        
        all_targets = set(ref_target_counts.index).union(set(cur_target_counts.index))
        
        d = max([abs(ref_target_counts.get(c, 0) - cur_target_counts.get(c, 0)) for c in all_targets])
        
        if d > TARGET_LIMIT: 
            alerts.append(f"Target drift: {d:.2%}")

    path = ""
    
    # for saving the report into the disk as html file
    try:
        report = Report(metrics=[DataDriftPreset()])
        ev_ref = reference[cols].copy()
        ev_curr = current[cols].copy()

        if has_valid_labels:
            ev_ref[TARGET] = pd.to_numeric(reference[TARGET], errors='coerce').fillna(0).astype(int)
            curr_labels = pd.to_numeric(labels, errors='coerce')
            valid_idx = curr_labels.dropna().index
            ev_curr = ev_curr.loc[valid_idx]
            ev_curr[TARGET] = curr_labels.loc[valid_idx].astype(int)

        try:
            snapshot = report.run(reference_data=ev_ref, current_data=ev_curr)
        except TypeError:
            config_mapping = {
                "numerical_features": [c for c in cols if pd.api.types.is_numeric_dtype(reference[c])],
                "categorical_features": [c for c in cols if not pd.api.types.is_numeric_dtype(reference[c])]
            }
            if has_valid_labels:
                config_mapping["target"] = TARGET
            
            snapshot = report.run(
                reference_data=ev_ref, 
                current_data=ev_curr, 
                column_mapping=config_mapping
            )
        
        os.makedirs("reports", exist_ok=True)
        
        path = f"reports/monitor_v{version}_{datetime.now():%Y%m%d_%H%M%S}.html"
        
        snapshot.save_html(path)

    except Exception as e:
        print(f"Monitoring report generation warning: {e}")

    return {"report": path, "alerts": alerts, "retrain_required": bool(alerts), "recall": recall_score_val}