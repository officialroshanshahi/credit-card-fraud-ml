import math
import numpy as np
from fastapi import status
import os, json, joblib, threading
from fastapi import BackgroundTasks, HTTPException
from sklearn.utils.validation import check_is_fitted
from src.v1.config import ACTIVE, VER
from src.v1.main import train_pipeline

TRAINING_STATE = {"is_training": False}
state_lock = threading.Lock()

# main logic for checking if the model exists, if it doesn't exists starting the training pipeline
def verify_and_get_model():
    
    # setting training_state as global variable to lock endpoints when the model is in training
    global TRAINING_STATE
    
    with state_lock:
        if TRAINING_STATE["is_training"]:
            raise HTTPException(503, "Model is currently training in the backend. Please try again shortly.")
    
    # checking if ACTIVE i.e. active_model.json exists,
    # if it does return the required model instance alongside required metadata
    if os.path.exists(ACTIVE):
        try:
            with open(ACTIVE) as f:
                a = json.load(f)
            
            ver, name = a["version"], a["model_name"]
            
            model = joblib.load(f"models/models/v{ver}/best_model_overall_{ver}.pkl")
            
            check_is_fitted(model)
            
            return model, ver, name
        except Exception as e:
            print(f"[BACKEND] Model corrupted or unreadable: {e}")

    # state lock logic for endpoints
    with state_lock:
        if TRAINING_STATE["is_training"]:
            raise HTTPException(503, "Model training already in progress.")
        TRAINING_STATE["is_training"] = True

    raise HTTPException(status_code=503, detail="START_TRAINING")


# this handles the training pipeline initializing
def handle_train_err(e: HTTPException, bg_tasks: BackgroundTasks):
    if e.detail == "START_TRAINING":
        def _bg():
            try:
                # training pipeline from main.py
                train_pipeline(retrain='no', ver=VER)
            finally:
                with state_lock:
                    TRAINING_STATE["is_training"] = False
        
        # training the model in the background i.e. async
        bg_tasks.add_task(_bg)
        
        return {"status": "training", "message": "Pipeline initialized."}, status.HTTP_202_ACCEPTED
    
    raise e

# for recursively cleaning the dataset before saving them
# mainly for json as json doesn't support NaN
def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or np.isnan(obj)):
        return None
    return obj