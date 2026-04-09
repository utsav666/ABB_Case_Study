FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt ./

RUN if command -v dnf >/dev/null 2>&1; then \
        dnf install -y gcc gcc-c++ make && dnf clean all; \
    else \
        yum install -y gcc gcc-c++ make && yum clean all; \
    fi \
    && python -m pip install --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY app ./app
COPY config ./config
COPY sample_data ./sample_data

CMD ["app.lambda_handler.handler"]
