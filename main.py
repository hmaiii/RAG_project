from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_module import build_graph

# Create FastAPI application
app = FastAPI()

# Enable CORS to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict this if needed, e.g., ["http://localhost:3000"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the answer graph from rag_module.py
graph = build_graph()

# Define input model expected from frontend
class QuestionRequest(BaseModel):
    question: str

# Define API key dependency
API_KEY = "my-secret-api-key"  # Replace this with a secure key from env or config
API_KEY_NAME = "X-API-Key"

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# Define endpoint to receive question and return answer
@app.post("/ask")
def ask_question(data: QuestionRequest, api_key: str = Depends(verify_api_key)):
    response = graph.invoke({"question": data.question})
    return {"answer": response["answer"]}

# Basic route to test if API is running
@app.get("/")
def root():
    return {"message": "API is running"}
