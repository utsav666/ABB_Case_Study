import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class DataLoader:
    @staticmethod
    def load_data():
        # Get the path to the schema file relative to this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up from app/sampledata to app, then to SQL-Agent, then to sample_data
        schema_path = os.path.join(
            current_dir, "..", "..", "sample_data", "schema_preserve.json"
        )
        schema_path = os.path.abspath(schema_path)

        with open(schema_path, "r") as table_structure:
            json_tab_strcuture = json.load(table_structure)
        return json_tab_strcuture


class DBLoader:
    @staticmethod
    def db_loader():
        # Load database credentials from environment variables
        db_creds = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT", 5432)),
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }

        # Check if all required credentials are provided
        missing = [k for k, v in db_creds.items() if v is None]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return {"data_base_detail": db_creds}


##### Unite Test 3 #########
# if __name__ == "__main__":
#     res= DataLoader.load_data()
#     print(res)
