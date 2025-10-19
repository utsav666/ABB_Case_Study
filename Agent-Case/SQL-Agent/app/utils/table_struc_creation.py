import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from sampledata.data_loader import DataLoader
class TabStrucCreation:
    def __init__(self,table_list):
        self.table_list = table_list
    def struc_creation(self):
        # with open(self.data_path) as f:
        #     data = json.load(f)
        data = DataLoader.load_data()
        tab_struc = ''
        for val in self.table_list:
            tab_struc = tab_struc+'\n'+str({val:data['tables'][val]})
        return tab_struc

####### unit test ##########
# if __name__ == '__main__':
#     tab_struc_gen=TabStrucCreation(['Movie', 'Person']) 
#     res = tab_struc_gen.struc_creation()
#     print(res)
    