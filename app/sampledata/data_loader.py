import os

from app.utils.schema_config import SchemaConfigManager


class DataLoader:
    @staticmethod
    def load_data():
        """Return schema data enriched with any UI-saved column configuration."""
        return SchemaConfigManager.get_schema_with_column_configuration()


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
