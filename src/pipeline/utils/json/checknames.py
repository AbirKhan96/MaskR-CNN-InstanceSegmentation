import os
import json


files = os.listdir("./")
for fname in files[3:]:
    if fname.split(".")[-1] == 'json':
        with open(fname, "r") as read_file:
            dic = json.load(read_file)
            json_name = dic['imagePath']
            fname = fname.split(".")[0]
            json_name = json_name.split(".")[0]
            #print(f'actual: {fname} \t json:{json_name}')
            
            if json_name != fname:
                print(f'actual: {fname} \t json:{json_name}')