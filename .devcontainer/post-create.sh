#!/bin/bash
set -e

echo "🚀 Setting up SQL Agent Dev Container..."

# Navigate to workspace
cd /workspace

# Setup SSH key
echo "🔐 Setting up SSH key..."
SSH_KEY="$HOME/.ssh/id_ed25519-gharista"

if [ -f "$SSH_KEY" ]; then
    # Start SSH agent if not running
    eval "$(ssh-agent -s)" > /dev/null 2>&1
    # Add the SSH key to the agent
    ssh-add "$SSH_KEY" 2>/dev/null || true
    echo "✅ SSH key configured: $SSH_KEY"
else
    echo "⚠️  SSH key not found at: $SSH_KEY"
    echo "   Please ensure your SSH key exists on your host machine at:"
    echo "   ~/.ssh/id_ed25519-gharista"
fi

# Make SSH agent persistent in bash sessions
if ! grep -q "SSH_AUTH_SOCK" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'SSHEOF'

# Start SSH agent if not running
if [ -z "$SSH_AUTH_SOCK" ]; then
    eval "$(ssh-agent -s)" > /dev/null 2>&1
    ssh-add ~/.ssh/id_ed25519-gharista 2>/dev/null || true
fi
SSHEOF
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate || true

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements if available
if [ -f "./requirements.txt" ]; then
    echo "📥 Installing dependencies from requirements.txt..."
    pip install -r ./requirements.txt 2>&1 | tail -5
fi

# Install essential packages
echo "📥 Installing essential packages..."
pip install \
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
    wikipedia \
    jupyter \
    jupyterlab \
    black \
    flake8 \
    pylint \
    pytest \
    ipython > /dev/null 2>&1

# Create necessary directories
echo "📁 Setting up directories..."
mkdir -p ./app/config
mkdir -p ./logs

# Check for .env file
if [ ! -f "./app/config/.env" ]; then
    echo "⚠️  Warning: .env file not found at ./app/config/.env"
    echo "📝 Creating template .env file..."
    cat > ./app/config/.env << 'EOF'
# OpenAI Configuration
OPENAI_MODEL=gpt-4

# Database Configuration
DATABASE_URL=sqlite:///./sql_agent.db

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
EOF
    echo "✅ Template .env file created. Please update with your actual API key."
fi

echo ""
echo "✅ Dev container setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Update your .env file with your OpenAI API key at: ./app/config/.env"
echo "2. Run Streamlit app: streamlit run ./app/brick_chat.py"
echo "3. Access at: http://localhost:8501"
echo ""
echo "🎯 Useful commands:"
echo "   - python -m pytest                    # Run tests"
echo "   - black .                             # Format code"
echo "   - flake8 .                            # Lint code"
echo "   - jupyter lab                         # Start Jupyter Lab"
echo ""
