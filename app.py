from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from rag_engine import generate_answer
from claim_model import predict_claim

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
def chat(query: str = Form(...)):
    answer = generate_answer(query)
    return {"response": answer}

@app.post("/claim-check")
def claim_check(
    amount: float = Form(...),
    years: int = Form(...),
    network: int = Form(...)
):
    result = predict_claim(amount, years, network)
    return {"prediction": result}
