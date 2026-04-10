from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize the app
# Use root_path="" if using a custom domain or HTTP API $default stage
app = FastAPI(title="Lambda FastAPI", root_path="")

# Add CORS so your frontend doesn't complain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "online", "message": "FastAPI on Lambda is working!"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}


# This is a helpful helper to check if the Lambda environment is healthy
@app.get("/health")
def health_check():
    return {"status": "healthy"}
