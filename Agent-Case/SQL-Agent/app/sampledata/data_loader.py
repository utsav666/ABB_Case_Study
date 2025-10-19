import json
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DataLoader:
    def load_data():
        with open('SQL-Agent/sample_data/schema_preserve.json','r') as table_structure :
            json_tab_strcuture = json.load(table_structure) 
        return json_tab_strcuture
##### Unite Test 3 #########    
# if __name__ == "__main__":
#     res= DataLoader.load_data()
#     print(res)