"""Create a test user in the database.
Usage:
  Set DATABASE_URL env var (or app/config/.env), then run:
    python create_test_user.py your-email@example.com yourpassword
"""
import sys
from backend_utils.db import SessionLocal
from backend_utils import crud


def main():
    if len(sys.argv) < 3:
        print("Usage: python create_test_user.py email password")
        sys.exit(1)
    email = sys.argv[1]
    password = sys.argv[2]
    db = SessionLocal()
    try:
        existing = crud.get_user_by_email(db, email)
        if existing:
            print("User already exists:", existing.email)
            return
        user = crud.create_user(db, email, password)
        print("Created user:", user.email, user.id)
    finally:
        db.close()


if __name__ == '__main__':
    main()
