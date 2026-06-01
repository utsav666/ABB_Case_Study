# Dev Container README

This project is configured with a development container for easy setup and consistent development environment.

## Prerequisites

- **Docker** - [Install Docker Desktop](https://www.docker.com/products/docker-desktop)
- **VS Code** - [Install VS Code](https://code.visualstudio.com/)
- **Remote - Containers Extension** - Install from VS Code Extensions Marketplace (search: "Remote - Containers")

## Quick Start

### Option 1: Using VS Code (Recommended)

1. **Open the project in VS Code:**
   ```bash
   code /path/to/ABB_Case_Study
   ```

2. **Reopen in Container:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Search for "Remote-Containers: Reopen in Container"
   - Select and wait for the container to build

3. **Access the application:**
   - Streamlit app will be available at `http://localhost:8501`

### Option 2: Using Command Line

1. **Build and start the container:**
   ```bash
   cd /path/to/ABB_Case_Study
   docker-compose -f .devcontainer/docker-compose.yml up -d
   ```

2. **Connect to the container:**
   ```bash
   docker-compose -f .devcontainer/docker-compose.yml exec dev bash
   ```

3. **Run the Streamlit app:**
   ```bash
   cd /workspace
   streamlit run Agent-Case/SQL-Agent/app/brick_chat.py
   ```

## Configuration

### Environment Variables

Create/update `Agent-Case/SQL-Agent/app/config/.env` with:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
DATABASE_URL=sqlite:///./sql_agent.db
DEBUG=True
LOG_LEVEL=INFO
```

### Port Mappings

- **8501-8503** - Streamlit Applications
- **5000** - Flask/Debug Server
- **5432** - PostgreSQL (if using db profile)

## Running the Application

### Within the Dev Container

```bash
# From inside the container (/workspace directory):

# Run Streamlit app
streamlit run Agent-Case/SQL-Agent/app/brick_chat.py

# Run tests (if available)
pytest

# Start Jupyter Lab
jupyter lab

# Format code with Black
black .

# Lint code
flake8 .
```

### Using Optional Database Services

To include database services (PostgreSQL/SQLite), use the `db` profile:

```bash
docker-compose -f .devcontainer/docker-compose.yml --profile db up -d
```

## Development Workflow

1. **Code Editing:**
   - All changes to files in your local workspace are reflected in the container in real-time
   - VS Code extensions run inside the container for better integration

2. **Running Code:**
   - Execute scripts and applications directly from VS Code terminal
   - Integrated debugging with Python extensions

3. **Git Operations:**
   - Git is installed and configured in the container
   - Use VS Code's Source Control panel or terminal commands

## Troubleshooting

### Container won't start
```bash
# Check Docker daemon is running
docker ps

# View container logs
docker-compose -f .devcontainer/docker-compose.yml logs dev

# Rebuild container
docker-compose -f .devcontainer/docker-compose.yml build --no-cache
```

### Permission issues
```bash
# Ensure proper ownership
docker-compose -f .devcontainer/docker-compose.yml exec dev sudo chown -R vscode /workspace
```

### Python dependencies not installed
```bash
# Install from requirements
docker-compose -f .devcontainer/docker-compose.yml exec dev pip install -r Agent-Case/SQL-Agent/requirements.txt
```

### Port already in use
```bash
# Change port mapping in docker-compose.yml or kill existing process:
lsof -i :8501
kill -9 <PID>
```

## Useful VS Code Shortcuts

- `Ctrl+` ` (backtick)` - Open integrated terminal
- `Ctrl+Shift+P` - Command palette
- `F5` - Start debugging
- `Ctrl+K Ctrl+O` - Open folder in container

## Additional Resources

- [VS Code Dev Containers Documentation](https://code.visualstudio.com/docs/remote/containers)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)

## Stopping the Container

### In VS Code:
- Press `Ctrl+Shift+P`
- Search "Remote-Containers: Reopen Locally"

### In Terminal:
```bash
docker-compose -f .devcontainer/docker-compose.yml down
```

## Cleaning Up

```bash
# Remove containers
docker-compose -f .devcontainer/docker-compose.yml down

# Remove images (optional)
docker rmi $(docker images -q)

# Remove volumes (optional, use with caution)
docker volume prune
```
