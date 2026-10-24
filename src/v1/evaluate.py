import datetime
import io
import joblib
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, precision_recall_curve, 
    recall_score, roc_curve
)
from pathlib import Path

# logic for finding project root: go back 2 times from current location
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# for plotting the evaluation metrics
def eval_plots(best_models, ver=1):
    print("\nGenerating evaluation plots...")
    rows = []
    
    # getting the metrics out of the best_models artifact
    for name, r in best_models.items():
        rows.append({
            "model": name,
            "val_recall": r["val_recall"],
            **r["test_metrics"]
        })

    df = pd.DataFrame(rows)
    
    # for saving those evaluation plots
    eval_dir = PROJECT_ROOT / f"plots/evaluate/v{ver}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # for giving unique name to every single metrics plots
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # this may be unnecessary as this is just saving the metrics as files
    # this can be removed if necessary
    df.to_csv(eval_dir / f"model_metrics_v{ver}_{timestamp}.csv", index=False)
    df.to_excel(eval_dir / f"model_metrics_v{ver}_{timestamp}.xlsx", index=False)

    generated_images = []

    # saving the plots and converting the fig for frontend
    def save_and_convert_fig(filename):
        disk_path = eval_dir / f"{filename}_v{ver}_{timestamp}.png"
        plt.savefig(disk_path, format='png', dpi=150, bbox_inches='tight')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    # plotting the required metrics
    
    fig, ax = plt.subplots(figsize=(10, 5))
    df.set_index("model")[["accuracy", "f1", "roc_auc", "precision", "recall"]].plot(
        kind="bar", ax=ax, figsize=(10, 5)
    )
    
    
    plt.title("Model Performance Comparison (Recall-Prioritized Fraud Detection)")
    
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=.3)
    plt.tight_layout()
    
    generated_images.append(save_and_convert_fig("performance_comparison"))

    plt.figure(figsize=(9, 6))
    for name, r in best_models.items():
        fpr, tpr = r["curves"]["roc"]
        plt.plot(fpr, tpr, label=f"{name} (Fraud Class)")
    
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.grid(alpha=.3)
    plt.tight_layout()
    
    generated_images.append(save_and_convert_fig("roc_curves"))

    plt.figure(figsize=(9, 6))
    for name, r in best_models.items():
        precision, recall = r["curves"]["pr"]
        plt.plot(recall, precision, label=f"{name} (Fraud Class)")
            
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc='lower left')
    plt.grid(alpha=.3)
    plt.tight_layout()
    
    generated_images.append(save_and_convert_fig("pr_curves"))

    print(f"Metrics and plots successfully saved to: {eval_dir}/")
    
    print(df.round(4))
    
    return generated_images


# main evaluation logic
def evaluator(x_test_processed, x_test_processed_std, y_test, trained_models, plot='no', ver=1):
    
    best_models = {}
    val_scores = {}
    
    # for saving the best model as well as metrics artifacts
    val_scores_path = PROJECT_ROOT / f"models/ev_search/v{ver}/val_scores_{ver}.joblib"
    
    if val_scores_path.exists():
        val_scores = joblib.load(val_scores_path)

    print("\nEvaluating models on test data using recall-optimized thresholds...")

    for model_name, model in trained_models.items():
        
        m_lower = model_name.lower()
        
        # logic for using scaled dataset if the model falls on any of them
        if "logisticregression" in m_lower or "kneighborclassifier" in m_lower or "knclassifier" in m_lower:
            X_te = x_test_processed_std
        else:
            X_te = x_test_processed

        threshold_path = PROJECT_ROOT / f"models/models/v{ver}/{model_name}_threshold_{ver}.pkl"
        
        # making sure the thresholds exists and if not, setting default threshold
        if threshold_path.exists():
            threshold = joblib.load(threshold_path)
            print(f"[{model_name}] Evaluated with threshold: {threshold:.4f}")
        else:
            threshold = 0.5
            print(f"[{model_name}] Threshold file not found. Falling back to: 0.5")

        # logic for getting out the probability,
        # and the model doesn't have predict_proba function, falling back to deecision_function
        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(X_te)
            y_probs = raw_probs[:, 1] if raw_probs.ndim == 2 and raw_probs.shape[1] >= 2 else raw_probs.ravel()
        else:
            y_probs = model.decision_function(X_te)

        y_pred = (y_probs >= threshold).astype(int)

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        
        # curves
        precision, recall, _ = precision_recall_curve(y_test, y_probs)

        # values
        rec_val = recall_score(y_test, y_pred, zero_division=0)
        prec_val = precision_score(y_test, y_pred, zero_division=0)

        print(f"[{model_name}] Test Results -> Recall: {rec_val:.4f} | Precision: {prec_val:.4f}")

        best_models[model_name] = {
            'model': model,
            'threshold': threshold,
            'val_recall': val_scores.get(model_name, 0.0),
            'test_metrics': {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_test, y_probs),
                'precision': prec_val,
                'recall': rec_val
            },
            'curves': {
                'roc': (fpr, tpr),
                'pr': (precision, recall)
            }
        }
    
    # for saving the metrics artifact
    best_models_path = PROJECT_ROOT / f'models/ev_metrics/v{ver}/best_models_metrics_{ver}.joblib'
    best_models_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_models, best_models_path)

    # figuring out the best model based on recall
    # recall because the dataset is highly imbalanced
    best_model_name, _ = max(
        best_models.items(), 
        key=lambda item: item[1]['test_metrics']['recall']
    )

    best_model = best_models[best_model_name]['model']
    print(f"\nBest model selected: {best_model_name} with Recall: {best_models[best_model_name]['test_metrics']['recall']:.4f}")
    
    # saving the best_model artifact
    best_model_path = PROJECT_ROOT / f"models/models/v{ver}/best_model_overall_{ver}.pkl"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, best_model_path)

    generated_plots = []
    
    # if plots =='yes' getting the plots
    if str(plot).lower() == "yes":
        generated_plots = eval_plots(best_models, ver=ver)
    
    return best_model_name, best_models[best_model_name], generated_plots