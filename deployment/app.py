from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware 
import joblib
import pandas as pd
import os 

app = FastAPI()

# Add CORS middleware (critical for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Get the absolute path to the deployment directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths for templates and static files
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Flexible model path (works in both local and production)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(BASE_DIR), "artifacts", "churn_deployment_model.joblib"))

# Load model with error handling
try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

def get_action_suggestion(prob):
    if prob >= 0.70:
        return "High Risk – Immediate retention action required. Offer personalized discounts, loyalty rewards, or special plans."
    elif prob >= 0.40:
        return "Medium Risk – Monitor closely. Provide proactive customer support and engagement offers."
    else:
        return "Low Risk – No immediate action needed. Continue normal engagement."

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")  # Health check endpoint
async def health_check():
    if model is None:
        return JSONResponse({"status": "unhealthy", "model_loaded": False}, status_code=500)
    return JSONResponse({
        "status": "healthy", 
        "model_loaded": True,
        "model_path": MODEL_PATH
    })

@app.post("/predict")
async def predict(
    tenure_months: int = Form(...),
    contract_type: str = Form(...),
    monthly_charges: float = Form(...),
    payment_method: str = Form(...),
    support_ticket_count: int = Form(...),
    avg_call_minutes: float = Form(...),
    avg_data_usage_gb: float = Form(...)
):
    # Check if model loaded successfully
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    
    # ---------------- VALIDATION RULES ----------------
    if tenure_months < 1 or tenure_months > 75:
        raise HTTPException(status_code=400, detail="Invalid tenure value. Must be between 1-75 months")
    
    if monthly_charges < 19 or monthly_charges > 119:
        raise HTTPException(status_code=400, detail="Monthly charges must be between $19 and $119")
    
    if support_ticket_count < 0 or support_ticket_count > 7:
        raise HTTPException(status_code=400, detail="Support tickets must be between 0-7")
    
    if avg_call_minutes < 0 or avg_call_minutes > 275:
        raise HTTPException(status_code=400, detail="Call minutes must be between 0-275")
    
    if avg_data_usage_gb < 0 or avg_data_usage_gb > 30:
        raise HTTPException(status_code=400, detail="Data usage must be between 0-30 GB")

    valid_contracts = ["month-to-month", "one year", "two year"]
    if contract_type not in valid_contracts:
        raise HTTPException(status_code=400, detail="Invalid contract type")

    valid_payments = ["electronic check", "credit card", "bank transfer", "mailed check"]
    if payment_method not in valid_payments:
        raise HTTPException(status_code=400, detail="Invalid payment method")

    # Prepare input data
    input_data = {
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "monthly_charges": monthly_charges,
        "payment_method": payment_method,
        "support_ticket_count": support_ticket_count,
        "avg_call_minutes": avg_call_minutes,
        "avg_data_usage_gb": avg_data_usage_gb
    }

    df = pd.DataFrame([input_data])
    
    try:
        prob = float(model.predict_proba(df)[:, 1][0])
        
        if prob >= 0.70:
            risk = "HIGH"
        elif prob >= 0.40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        suggestion = get_action_suggestion(prob)

        return JSONResponse({
            "probability": round(prob * 100, 2),
            "risk": risk,
            "suggestion": suggestion
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# NEW: For local testing only
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)