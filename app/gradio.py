import io, json, httpx, gradio as gr
from PIL import Image
from src.v1.config import API_URL

def api_call(endpoint: str, method: str = "GET", token: str = "", json_data: dict = None, raw_response: bool = False):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        res = httpx.request(method, f"{API_URL}{endpoint}", json=json_data, headers=headers, timeout=30.0)
        if raw_response:
            return Image.open(io.BytesIO(res.content)) if res.status_code == 200 else None
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def handle_auth(action, user, pwd):
    res = api_call(f"/{action.lower()}", "POST", json_data={"username": user, "password": pwd})
    token = res.get("access_token", "") if action == "Login" and isinstance(res, dict) else ""
    return token, res

def handle_single_inference(action, token, time_val, amount_val, *v_vals):
    data_dict = {
        "Time": float(time_val),
        "Amount": float(amount_val)
    }
    for i, val in enumerate(v_vals, start=1):
        data_dict[f"V{i}"] = float(val)

    payload = {"data": data_dict}
    
    if action == "Predict":
        return api_call("/predict", "POST", token=token, json_data=payload), None
    
    img = api_call("/explain", "POST", token=token, json_data=payload, raw_response=True)
    return {"status": "image_generated" if img else "failed"}, img

def handle_batch_inference(token, json_str):
    try:
        data_list = json.loads(json_str)
        payload = {"data": data_list} if isinstance(data_list, list) else data_list
        return api_call("/predict/batch", "POST", token=token, json_data=payload)
    except Exception as e:
        return {"error": f"Invalid JSON format: {str(e)}"}

with gr.Blocks(title="Credit Card Fraud Detection Platform") as demo:
    auth_token = gr.State("")

    gr.Markdown("# Credit Card Fraud Detection ML Control Center")
    
    with gr.Tab("Inference & Authentication"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Auth & System")
                user_in = gr.Textbox(label="Username")
                pass_in = gr.Textbox(label="Password", type="password")
                
                with gr.Row():
                    btn_login = gr.Button("Login", variant="primary")
                    btn_reg = gr.Button("Register")
                
                with gr.Row():
                    btn_status = gr.Button("/status")
                    btn_health = gr.Button("/health")
                    
                token_disp = gr.Textbox(label="Active Bearer Token", interactive=False, max_lines=1)
                sys_out = gr.JSON(label="System Response")

            with gr.Column(scale=2):
                gr.Markdown("### Single Transaction Inference & Explanation")
                with gr.Row():
                    time_in = gr.Number(label="Time", value=61290.0)
                    amount_in = gr.Number(label="Amount", value=11.5)

                gr.Markdown("#### Principal Components (V1 - V28)")
                v_inputs = []
                # Create collapsible or multi-row inputs for V1 through V28
                sample_v_defaults = [
                    1.228821,-0.063408,0.274145,0.647465,-0.048135,0.372073,-0.224231,0.079939,0.640759,-0.273054,
                    -1.252728,0.465079,0.400502,-0.292842,-0.101774,-0.399836,0.034336,-0.783550,0.141345,-0.096566,
                    -0.129554,-0.083779,-0.151661,-0.700372,0.598550,0.491409,0.002989,0.001782
                ]
                
                with gr.Row():
                    for i in range(1, 15):
                        v_inputs.append(gr.Number(label=f"V{i}", value=sample_v_defaults[i-1]))
                with gr.Row():
                    for i in range(15, 29):
                        v_inputs.append(gr.Number(label=f"V{i}", value=sample_v_defaults[i-1]))

                with gr.Row():
                    btn_pred = gr.Button("Predict (/predict)", variant="primary")
                    btn_exp = gr.Button("Explain (/explain)")

                pred_out = gr.JSON(label="Prediction Result")
                img_out = gr.Image(label="SHAP Explanation Plot", type="pil")

    with gr.Tab("Model Metrics & Evaluation"):
        gr.Markdown("### Model Performance Analysis (/evaluate)")
        btn_eval = gr.Button("Fetch Evaluation Plots", variant="primary")
        eval_img_out = gr.Image(label="Performance Charts (Bar Chart, ROC, PR Curves)", type="pil")

    with gr.Tab("Batch Operations & Monitoring"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Batch Processing (/predict/batch)")
                batch_json = gr.TextArea(
                    label="Batch JSON Input",
                    value='[\n  {\n    "Time": 61290.0, "V1": 1.2288211502, "V2": -0.0634077165, "V3": 0.2741451422, "V4": 0.6474650218,\n    "V5": -0.0481345612, "V6": 0.3720730286, "V7": -0.2242305874, "V8": 0.0799390492, "V9": 0.6407588171,\n    "V10": -0.2730537022, "V11": -1.2527279388, "V12": 0.4650787707, "V13": 0.4005021153, "V14": -0.2928418606,\n    "V15": -0.101774016, "V16": -0.3998358978, "V17": 0.0343356568, "V18": -0.7835502549, "V19": 0.1413449004,\n    "V20": -0.0965659024, "V21": -0.1295544481, "V22": -0.0837793282, "V23": -0.1516614739, "V24": -0.7003715973,\n    "V25": 0.5985501645, "V26": 0.4914090706, "V27": 0.0029892597, "V28": 0.0017822861, "Amount": 11.5\n  }\n]',
                    lines=8
                )
                btn_batch = gr.Button("Run Batch Prediction")
                batch_out = gr.JSON(label="Batch Results")

            with gr.Column():
                gr.Markdown("### Ground Truth & Production Monitoring")
                with gr.Row():
                    req_id = gr.Textbox(label="Request ID")
                    class_val = gr.Radio([0, 1], value=0, label="Actual Class (0: Legitimate, 1: Fraud)")
                btn_label = gr.Button("Submit Ground Truth (/label)")
                label_out = gr.JSON(label="Label Response")

                gr.Markdown("#### Data Drift & Retraining Check")
                btn_monitor = gr.Button("Run Production Monitoring (/monitor)", variant="primary")
                mon_out = gr.JSON(label="Monitoring Diagnostics")

    btn_login.click(lambda u, p: handle_auth("Login", u, p), inputs=[user_in, pass_in], outputs=[auth_token, sys_out])
    btn_reg.click(lambda u, p: handle_auth("Register", u, p), inputs=[user_in, pass_in], outputs=[auth_token, sys_out])
    auth_token.change(lambda t: t, inputs=[auth_token], outputs=[token_disp])

    btn_status.click(lambda: api_call("/status"), outputs=[sys_out])
    btn_health.click(lambda t: api_call("/health", token=t), inputs=[auth_token], outputs=[sys_out])

    inf_inputs = [auth_token, time_in, amount_in] + v_inputs
    btn_pred.click(lambda *args: handle_single_inference("Predict", *args), inputs=inf_inputs, outputs=[pred_out, img_out])
    btn_exp.click(lambda *args: handle_single_inference("Explain", *args), inputs=inf_inputs, outputs=[pred_out, img_out])

    btn_eval.click(lambda t: api_call("/evaluate", "GET", token=t, raw_response=True), inputs=[auth_token], outputs=[eval_img_out])
    btn_batch.click(handle_batch_inference, inputs=[auth_token, batch_json], outputs=[batch_out])
    btn_label.click(lambda t, r, c: api_call("/label", "POST", token=t, json_data={"request_id": r, "class": int(c)}), inputs=[auth_token, req_id, class_val], outputs=[label_out])
    btn_monitor.click(lambda t: api_call("/monitor", "POST", token=t), inputs=[auth_token], outputs=[mon_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())