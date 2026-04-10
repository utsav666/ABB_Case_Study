FROM public.ecr.aws/lambda/python:3.11

COPY requirements-lambda.txt ./

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --only-binary=:all: \
        -r requirements-lambda.txt

COPY app ./app
COPY config ./config
COPY sample_data ./sample_data

CMD ["app.lambda_handler.handler"]
