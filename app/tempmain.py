from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html

app = FastAPI(root_path="")


@app.get("/", include_in_schema=False)
def read_root():
    return {"status": "ok"}


# Overriding the docs route to use a public CDN
# This bypasses potential local file/dependency issues
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",  # Should match your openapi_url
        title="API Docs",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    from fastapi.openapi.utils import get_openapi

    return get_openapi(title="My API", version="1.0.0", routes=app.routes)
