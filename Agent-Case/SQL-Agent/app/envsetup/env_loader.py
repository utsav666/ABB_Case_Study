# core/env_loader.py
from dotenv import load_dotenv
from pathlib import Path
import os

_env_loaded = False  # simple flag to avoid loading multiple times

def load_env():
    """
    Load environment variables from the central .env file
    no matter which module calls it.
    """
    global _env_loaded
    if _env_loaded:
        return  # prevent redundant loading
    
    # Find project root (assuming this file lives in project/core/)
    root_dir = Path(__file__).resolve().parents[1]
    env_path = root_dir / "config" / ".env"
    
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at: {env_path}")
    
    load_dotenv(dotenv_path=env_path)
    _env_loaded = True
    print(f"[env_loader] Loaded environment from {env_path}")
load_env()