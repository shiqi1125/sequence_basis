import pickle
import os

cache_name = '../Data/ZM/RNA-Seq_map.pkl'
if os.path.isfile(cache_name):
    print("Deserializing data..")
    with open(cache_name, 'rb') as f:
        data = pickle.load(f)
        print("Deserialization done.")
    f.close()

print(len(data))
print(len(data["2"]))