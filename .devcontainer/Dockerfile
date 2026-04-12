FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    openssh-client \
    curl \
    wget \
    vim \
    nano \
    postgresql-client \
    sqlite3 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash vscode && \
    mkdir -p /workspace && \
    chown -R vscode:vscode /workspace

# Set working directory
WORKDIR /workspace

# Install Python development tools
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
    black \
    flake8 \
    pylint \
    pytest \
    pytest-cov \
    ipython \
    jupyter \
    jupyterlab

# Install dependencies - requirements file will be installed in post-create.sh
# This allows flexibility with the build context

# Install essential packages for the project
RUN pip install \
    langchain \
    langchain-openai \
    langchain-community \
    streamlit \
    pandas \
    numpy \
    sqlalchemy \
    sqlglot \
    python-dotenv \
    openai \
    pydantic \
    tiktoken \
    wikipedia

# Switch to non-root user
USER vscode

# Configure SSH
RUN mkdir -p ~/.ssh && \
    chmod 700 ~/.ssh && \
    ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true && \
    chmod 644 ~/.ssh/known_hosts 2>/dev/null || true && \
    git config --global --add safe.directory /workspace

# Initialize shell
RUN echo 'export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> ~/.bashrc

EXPOSE 8501 8502 8503 5000

CMD ["/bin/bash"]
