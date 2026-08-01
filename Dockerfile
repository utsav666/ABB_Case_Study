# FROM public.ecr.aws/lambda/python:3.12

# COPY requirements-lambda.txt ./

# RUN python -m pip install --upgrade pip \
#     && pip install --no-cache-dir --only-binary=:all: \
#         -r requirements-lambda.txt

# COPY app/ app/
# COPY config/ config/
# COPY sample_data/ sample_data/

# CMD ["app.lambda_handler.handler"]

FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY config/ ./config/
COPY sample_data/ ./sample_data/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
