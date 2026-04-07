"""Utility to create DB tables from SQLAlchemy models.

Run this after setting DATABASE_URL in environment (or in app/config/.env).
"""
import os
from backend_utils.db import engine, Base


def main():
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("Done.")


if __name__ == "__main__":
    main()
