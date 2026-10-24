import joblib, numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, train_test_split
from sklearn.metrics import precision_recall_curve
from src.v1.config import MIN_RECALL

# finding out the PROJECT_ROOT to save our model artifacts
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# main training pipeline
def trainer(x_train, x_train_pr, y_train, model_dic, params, scoring_metrics, ver):
    
    print('\nStarting training pipeline with threshold optimization.\n')
    
    best_models, best_thresholds, val_scores = {}, {}, {}
    
    y_train_flat = y_train.squeeze()
    
    # splitting the training data into train and validation set,
    # technically this step can be removed but, to get the best out of dataset this is used.
    train, val = train_test_split(np.arange(len(x_train)), test_size=0.20, stratify=y_train_flat, random_state=42)
    
    split_fold = np.zeros(len(x_train))
    split_fold[train] = -1
    
    pds = PredefinedSplit(test_fold=split_fold)
    
    for m_name, init_model in model_dic.items():
        
        print(f"Searching space for {m_name}...")
        
        m_low = m_name.lower()
        
        # use standardized processed data for LR and others
        X_tr = x_train_pr.copy() if any(k in m_low for k in ["logisticregression", "kneighbor", "knclassifier"]) else x_train.copy()
        
        # change this as necessary
        n_workers = 1
        
        fit_params = {}
        
        # early stopping logic currently only for XGBClassifier as it it used in production
        if "xgbclassifier" in m_low:
            # hist as this increases the training time, and logloss as this dataset is binary
            # and stops early if the 15 consequetive trees are built but not improving the metrics
            init_model.set_params(tree_method="hist", eval_metric="logloss", early_stopping_rounds=15)
            fit_params = {"eval_set": [(X_tr.iloc[val], y_train_flat.iloc[val])], "verbose": False}

        search = RandomizedSearchCV(
            estimator=init_model,
            param_distributions=params[m_name],
            n_iter=10, # this could be increased but current 10 also outputs good model
            cv=pds,
            scoring=scoring_metrics,
            refit="recall",
            random_state=42,
            n_jobs=n_workers)
        
        search.fit(X_tr, y_train_flat, **fit_params)
        
        best_model = search.best_estimator_

        # optimizing threshold on validation set
        val_X = X_tr.iloc[val]
        val_y = y_train_flat.iloc[val]
        val_probs = best_model.predict_proba(val_X)[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(val_y, val_probs)
        
        target_recall_floor = MIN_RECALL # currently set to 0.75
        valid_indices = np.where(recall[:-1] >= target_recall_floor)[0]
        
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(precision[valid_indices])]
        else:
            best_idx = np.argmax(recall)
        
        # finding out optimal threshold
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        optimal_threshold = min(optimal_threshold, 0.80)
        
        print(f"[{m_name}] Optimal Threshold: {optimal_threshold:.4f} | Validation Recall: {recall[best_idx]:.4f} | Validation Precision: {precision[best_idx]:.4f}")

        # for saving model artifacts
        artifacts = [
            (f"models/models/v{ver}/{m_name}_best_{ver}.pkl", best_model),
            (f"models/ev_search/v{ver}/{m_name}_search_{ver}.pkl", search),
            (f"models/models/v{ver}/{m_name}_threshold_{ver}.pkl", optimal_threshold)
        ]
        
        for path, obj in artifacts:
            (PROJECT_ROOT / path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(obj, PROJECT_ROOT / path)

        best_models[m_name] = best_model
        best_thresholds[m_name] = optimal_threshold
        val_scores[m_name] = search.best_score_
        
        print(f"Search and threshold tuning complete for {m_name}.\n")

    print("All models have been trained and thresholds optimized.")
    
    # making sure the folder exists before saving as it can save alot of headache if the folder doesn't exist
    (PROJECT_ROOT / f"models/ev_search/v{ver}/val_scores_{ver}.joblib").parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(val_scores, PROJECT_ROOT / f"models/ev_search/v{ver}/val_scores_{ver}.joblib")
    
    joblib.dump(best_thresholds, PROJECT_ROOT / f"models/ev_search/v{ver}/thresholds_{ver}.joblib")
    
    return best_models, best_thresholds