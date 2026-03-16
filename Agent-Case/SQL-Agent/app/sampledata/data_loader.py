import json
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DataLoader:
    def load_data():
        # Get the path to the schema file relative to this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up from app/sampledata to app, then to SQL-Agent, then to sample_data
        schema_path = os.path.join(current_dir, '..', '..', 'sample_data', 'schema_preserve.json')
        schema_path = os.path.abspath(schema_path)
        
        with open(schema_path, 'r') as table_structure:
            json_tab_strcuture = json.load(table_structure) 
        return json_tab_strcuture
class DBLoader():
    def db_loader():
        # Get the path to the schema file relative to this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up from app/sampledata to app, then to SQL-Agent, then to sample_data
        db_path = os.path.join(current_dir, '..', '..', 'sample_data', 'database_cred.json')
        db_path = os.path.abspath(db_path)
        
        with open(db_path, 'r') as db_structure:
            json_db_strcuture = json.load(db_structure) 
        return json_db_strcuture

##### Unite Test 3 #########    
# if __name__ == "__main__":
#     res= DataLoader.load_data()
#     print(res)