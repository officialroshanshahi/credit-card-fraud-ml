import io
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from PIL import Image

# explain_model function computes SHAP values for a binary classifier and test dataset,
# generating visualizations and saving feature importance metrics.
def explain_model(x_test, model, name, explainer_type="auto", ver=1, target_class=1):
    
    print(f"Calculating SHAP for {name} (Binary Classification - Target Class: {target_class})...")
    
    # ensure x_test is a DataFrame
    if not isinstance(x_test, pd.DataFrame):
        raise TypeError("x_test must be a pandas DataFrame")

    num_samples = len(x_test)
    X_sample = x_test.copy()

    # align feature names directly with model expectation
    expected_features = None
    
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        try:
            expected_features = model.get_booster().feature_names
        except Exception:
            pass
    
    # if expected features are known, ensure the test sample has the same columns
    if expected_features:
        missing_cols = [c for c in expected_features if c not in X_sample.columns]
        
        for c in missing_cols:
            X_sample[c] = 0.0
        
        X_sample = X_sample[expected_features]

    features = X_sample.columns.tolist()

    # build explicit background dataset to avoid E[f(X)] == f(x)
    project_root = Path(__file__).resolve().parents[2]
    baseline_path = project_root / f"data/processed/v{ver}/test_baseline.csv"
    
    if baseline_path.exists():
        bg_df = pd.read_csv(baseline_path)
        
        # unnecessary columns
        meta_cols = {
            "class",
            "request_id",
            "timestamp",
            "prediction",
            "probability",
            "risk_percentage",
            "risk_level"
        }
        
        bg_features = [c for c in bg_df.columns if c not in meta_cols]
        bg_data = bg_df[bg_features]
        
        if expected_features:
            for c in expected_features:
                if c not in bg_data.columns:
                    bg_data[c] = 0.0
            bg_data = bg_data[expected_features]
            
        bg_summary = shap.kmeans(bg_data, min(20, len(bg_data)))
    else:
        # dynamic noise matching the exact shape of X_sample
        noise = np.random.normal(0, 0.1, size=X_sample.shape)
        bg_summary = pd.DataFrame(X_sample.values + noise, columns=features)

    # dynamic explainer selection based on explainer_type or model characteristics
    lower_name = name.lower()
    
    # for tree based models using TreeExplainer
    is_tree_model = (
        explainer_type == "tree" or 
        any(t in lower_name for t in ["xgb", "xgboost", "rf", "randomforest", "lgb", "lightgbm", "catboost", "tree"]) or 
        hasattr(model, "get_booster")
    )

    # checking if the model is tree based model and if it is using TreeExplainer
    if is_tree_model:
        print("[SHAP INFO] Evaluating probabilities via TreeExplainer...")
        
        explainer = shap.TreeExplainer(model)
        
        raw_shap = explainer(X_sample)
        
        # handle multi-output or binary class formatting for TreeExplainer
        if len(raw_shap.values.shape) == 3:
            target_vals = raw_shap.values[..., target_class]
            target_base = raw_shap.base_values[:, target_class] if np.ndim(raw_shap.base_values) > 1 else raw_shap.base_values
        else:
            target_vals = raw_shap.values
            target_base = raw_shap.base_values
            
        shap_values = shap.Explanation(
            values=target_vals,
            base_values=target_base,
            data=X_sample.to_numpy(),
            feature_names=features
        )
    
    # if the model isn't tree based
    else:
        print("[SHAP INFO] Evaluating binary probabilities via KernelExplainer...")
        predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=features))[:, target_class]
        
        explainer = shap.KernelExplainer(predict_fn, bg_summary)
        vals = explainer.shap_values(X_sample)
        base = explainer.expected_value

        target_vals = vals[target_class] if isinstance(vals, list) else vals
        
        if isinstance(base, (list, np.ndarray)):
            target_base = base[target_class] if len(base) > target_class else base[0]
        else:
            target_base = base

        shap_values = shap.Explanation(
            values=target_vals,
            base_values=target_base,
            data=X_sample.to_numpy(),
            feature_names=features
        )

    generated_images = []

    # for frontend
    def fig_to_pil():
        
        buf = io.BytesIO()
        
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        
        plt.close()
        
        buf.seek(0)
        
        return Image.open(buf)

    # beeswarm summary plot
    plt.figure()
    
    if num_samples == 1:
        # for model that is not tree
        if isinstance(explainer, shap.KernelExplainer):
            
            bg_sample_eval = bg_summary.data if hasattr(bg_summary, "data") else bg_summary.iloc[:10]
            
            bg_vals = explainer.shap_values(bg_sample_eval)
            
            target_bg_vals = bg_vals[target_class] if isinstance(bg_vals, list) else bg_vals
            
            shap_batch = shap.Explanation(
                values=target_bg_vals,
                base_values=target_base,
                data=bg_sample_eval.to_numpy() if isinstance(bg_sample_eval, pd.DataFrame) else bg_sample_eval,
                feature_names=features
            )
        else:
            shap_batch = shap_values
        
        # plotting the most important features upto 15 of them
        shap.plots.beeswarm(shap_batch, max_display=15, show=False)
    else:
        shap.plots.beeswarm(shap_values, max_display=15, show=False)

    plt.title(f"SHAP Summary (Beeswarm - Class {target_class}) - {name}")
    
    plt.tight_layout()
    
    generated_images.append(fig_to_pil())


    # bar chart plot
    plt.figure()
    
    if num_samples == 1:
        shap.plots.bar(shap_values[0], max_display=15, show=False)
    else:
        shap.plots.bar(shap_values, max_display=15, show=False)
    
    plt.title(f"SHAP Feature Importance (Class {target_class}) - {name}")
    
    plt.tight_layout()
    
    generated_images.append(fig_to_pil())


    # waterfall plot
    plt.figure()
    single_exp = shap_values[0]
    
    shap.plots.waterfall(single_exp, max_display=15, show=False)
    plt.title(f"SHAP Waterfall (Class {target_class}) - {name}")
    plt.tight_layout()
    generated_images.append(fig_to_pil())


    # save feature importance CSV
    # technically this is unncessary. it can be removed
    out = Path(f"plots/shap/v{ver}")
    out.mkdir(parents=True, exist_ok=True)
    
    raw_vals = shap_values.values
    mean_abs = np.abs(raw_vals).mean(axis=0).ravel() if num_samples > 1 else np.abs(raw_vals).ravel()

    pd.DataFrame({"Feature": features,"Mean_Absolute_SHAP": mean_abs}).sort_values("Mean_Absolute_SHAP", ascending=False).to_csv(
        out / f"{name}_importance_{datetime.now():%Y%m%d_%H%M%S}.csv", index=False
    )
    
    print(f"{name} SHAP calculation completed successfully.")
    
    return generated_images