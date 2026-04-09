FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY app ./app
COPY config ./config
COPY sample_data ./sample_data

CMD ["app.lambda_handler.handler"]
