from contextlib import asynccontextmanager
import os, glob, time, joblib, pandas as pd, uvicorn, io
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from anyio import to_thread
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from api.database.schema import UserAuth, ReqSchema, Batch, Label
from api.database.database import register_user, verify_user, init_db
from api.middleware.auth import auth, create_access_token
from api.middleware.train import verify_and_get_model, TRAINING_STATE, state_lock, handle_train_err
from api.controller.retrain import run_monitoring
from src.v1.preprocessing import preprocessor
from src.v1.main import run_prediction
from src.v1.config import ACTIVE, HOST, PORT
from src.v1.explain import explain_model
from src.v1.evaluate import eval_plots

# initialising the database at the start of the server
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Credit Card Fraud Detection API", lifespan=lifespan)

# dynamically checking current path
current_path = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in current_path.parents if p.name == "credit-card-fraud-ml")

# for logging
L_DIR = PROJECT_ROOT / "logs"
L_FILE = L_DIR / "logs.csv"

# logic for logging server activity
# it can be its own middleware but for simplicity it is inside api
def append_log_sync(log_row: dict):
    L_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([log_row]).to_csv(L_FILE, mode="a", header=not L_FILE.exists(), index=False)


@app.middleware("http")
async def log_server_activity(request: Request, call_next):
    st = time.time()
    try: 
        res = await call_next(request)
        sc = res.status_code
    except Exception as e: 
        sc = 500; raise e
    finally:
        log = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "client_ip": request.client.host if request.client else "unknown",
            "method": request.method,
            "endpoint": request.url.path,
            "status_code": sc,
            "latency_seconds": round(time.time() - st, 4)}
        
        await to_thread.run_sync(append_log_sync, log)
        
    return res


# endpoint for registering the user
@app.post("/register", status_code=201)
def register(user: UserAuth):
    if not register_user(user.username, user.password): 
        raise HTTPException(400, "Username already registered.")
    return {"status": "success"}


# endpoint for login
@app.post("/login")
def login(u: UserAuth):
    if not verify_user(u.username, u.password): 
        raise HTTPException(401, "Invalid credentials.")
    return {"access_token": create_access_token(u.username), "token_type": "bearer"}


# endpoint for getting model health
@app.get("/health")
async def health(b: BackgroundTasks, u=Depends(auth)):
    try: 
        _, version, model_name = verify_and_get_model()
        return {
            "status": "healthy", 
            "version": version, 
            "model": model_name
        }
    except HTTPException as e:
        return handle_train_err(e, b)


# endpoint for predicting
@app.post("/predict")
async def predict_api(r: ReqSchema, b: BackgroundTasks, _=Depends(auth)):
    try:
        return await to_thread.run_sync(run_prediction, r.data.dict(by_alias=True), verify_and_get_model())
    except HTTPException as e:
        return handle_train_err(e, b)


# endpoint for batch prediction
@app.post("/predict/batch")
async def batch(r: Batch, b: BackgroundTasks, _=Depends(auth)):
    try:
        return [await to_thread.run_sync(run_prediction, x.dict(by_alias=True), verify_and_get_model()) for x in r.data]
    except HTTPException as e:
        return handle_train_err(e, b)


# labeling the current dataset based on req_id
@app.post("/label")
async def label(r: Label, _=Depends(auth)):
    def sync_l():
        _, ver, _ = verify_and_get_model()
        
        target_dir = PROJECT_ROOT / f"data/new/v{ver}"
        
        for f in glob.glob(os.path.join(target_dir, "production_data_*.csv")):
            
            d = pd.read_csv(f)
            
            mask = d.request_id.astype(str) == r.request_id
            
            if mask.any():
                d.loc[mask, "class"] = r.class_label
                
                d.to_csv(f, index=False)
                
                return {"status": "labeled"}
                
        raise HTTPException(404, "Request ID not found.")
    
    return await to_thread.run_sync(sync_l)


# endpoint for monitoring the model performance
# of course, it could be made automatic but for simplicity it is an endpoint
@app.post("/monitor")
async def monitor(_=Depends(auth)):
    try:
        model_instances = verify_and_get_model()
        
        results = await to_thread.run_sync(run_monitoring, model_instances)
        
        return {
            "status": "success",
            "message": "Monitoring analysis completed successfully.",
            "data_summary": results
        }
        
    except Exception as e:
        print(f"[API ROUTE ERROR] /monitor execution block failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Monitoring error: {str(e)}")


# endpoint for status of the model ready, untrained or training
@app.get("/status")
def get_status():
    with state_lock:
        cur = TRAINING_STATE["is_training"]
    return {"status": "training" if cur else ("ready" if os.path.exists(ACTIVE) else "untrained")}


# endpoint for explaining the model performance based on the given data
# AND sync_explain could be its own file but for simplicity it is in here
@app.post("/explain", responses={200: {"content": {"image/png": {}}}})
async def explain(r: ReqSchema, b: BackgroundTasks, _=Depends(auth)):
    try: 
        model, version, name = verify_and_get_model()
    except HTTPException as e: 
        return handle_train_err(e, b)

    def sync_explain():
        # preprocessing the given dataset
        x, xs = preprocessor(pd.DataFrame([r.data.dict(by_alias=True)]), "no", "yes", version)
        
        # using xs or scaled dataset if the model fall in any of them
        eval_data = xs if any(k in name.lower() for k in ["logistic", "kneighbor", "knc"]) else x
        
        if hasattr(model, "feature_names_in_"):
            
            missing_cols = [c for c in model.feature_names_in_ if c not in eval_data.columns]
            
            # if there missing cols filling them with 0s
            for c in missing_cols:
                eval_data[c] = 0.0
            
            eval_data = eval_data[model.feature_names_in_]

        eval_data = eval_data.astype(float)
        
        # explainer based on model type
        explainer = "tree" if any(k in name.lower() for k in ["forest", "tree", "boost"]) else ("linear" if any(k in name.lower() for k in ["linear", "logistic"]) else "kernel")
        
        # target class 1 for binary fraud classification
        images = explain_model(x_test=eval_data, model=model, name=name, explainer_type=explainer, ver=version, target_class=1)
        
        # if we didn't get any output from explain_model function
        if not images:
            raise HTTPException(500, "SHAP plot generation failed.")
        
        # for frontend printing those plots
        fig, axes = plt.subplots(len(images), 1, figsize=(12, 6 * len(images)))
        if len(images) == 1: 
            axes = [axes]
            
        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis('off')
            
        plt.tight_layout()
        
        img_buf = io.BytesIO()
        
        plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        img_buf.seek(0)
        
        return img_buf

    # calling the sync_explain function as sync function
    buffer = await to_thread.run_sync(sync_explain)
    
    return StreamingResponse(buffer, media_type="image/png")


# endpoints for evaluating
# technically this is unncessary as it is already calculated in main.py but if user wants those evaluation metrcs plots
# they can visit this endpoint
# AND same as sync_explain, the sync_eval could be its own file
@app.get("/evaluate", responses={200: {"content": {"image/png": {}}}})
async def evaluate_models(b: BackgroundTasks, _=Depends(auth)):
    try:
        model_inst = verify_and_get_model()
        ver = model_inst[1]
    except HTTPException as e:
        return handle_train_err(e, b)

    def sync_eval():
        metrics_path = PROJECT_ROOT / f"models/ev_metrics/v{ver}/best_models_metrics_{ver}.joblib"
        if not metrics_path.exists():
            raise HTTPException(404, detail="Evaluation metrics file not found. Please train models first.")
            
        best_models = joblib.load(metrics_path)
        images = eval_plots(best_models, ver=ver)

        if not images:
            raise HTTPException(500, detail="Evaluation plot generation failed.")

        # for plotting those metrics
        fig, axes = plt.subplots(len(images), 1, figsize=(10, 5 * len(images)))
        if len(images) == 1:
            axes = [axes]
        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis('off')
        
        plt.tight_layout()

        img_buf = io.BytesIO()
        
        plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        img_buf.seek(0)
        
        return img_buf

    buffer = await to_thread.run_sync(sync_eval)
    return StreamingResponse(buffer, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT, reload=True)