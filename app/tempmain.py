from fastapi import FastAPI

app = FastAPI(
    root_path="",  # Forces the app to treat the domain root as the base
    docs_url="/docs",
    openapi_url="/openapi.json",
)


@app.get("/")
def read_root():
    return {"Hello": "World"}
