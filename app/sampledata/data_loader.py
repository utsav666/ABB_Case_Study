import os

# NOTE: DataLoader was removed. It read the bundled sample_data/schema_preserve.json
# fixture, which described a movie database and silently answered questions asked
# against real connections. Schema now comes ONLY from settings.schema_graphs.


class DBLoader:
    @staticmethod
    def db_loader():
        # Load database credentials from environment variables
        # Support both prefixed (SETTINGS_DB_*) and non-prefixed (DB_*) variable names
        db_creds = {
            "host": os.getenv("DB_HOST") or os.getenv("SETTINGS_DB_HOST"),
            "port": int(os.getenv("DB_PORT") or os.getenv("SETTINGS_DB_PORT", 5432)),
            "database": os.getenv("DB_NAME") or os.getenv("SETTINGS_DB_NAME"),
            "user": os.getenv("DB_USER") or os.getenv("SETTINGS_DB_USER"),
            "password": os.getenv("DB_PASSWORD") or os.getenv("SETTINGS_DB_PASSWORD"),
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
