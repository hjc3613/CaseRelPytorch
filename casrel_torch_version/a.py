import pymongo
from pprint import pprint as print
local = pymongo.MongoClient('127.0.0.1:27017')
local_db = local.get_database('jhdata')
local_collection = local_db.get_collection('ruyuanjilu')
client = pymongo.MongoClient('mongodb://read:read@192.168.8.22:20000' )
db = client.get_database('data_statistics')
ruyuanjilu = db.get_collection('ruyuanjilu')

with open(r'C:\Users\hujunchao\Desktop\ids.txt') as f:
    remote_ids = f.readlines()
    remote_ids = [i.strip() for i in remote_ids]

local_ids = [i['_id'] for i in local_collection.find()]

to_get_ids = set(remote_ids) - set(local_ids)
to_get_ids = list(to_get_ids)[:500000]
i = 0
for id_ in to_get_ids:
    one = ruyuanjilu.find_one({'_id':id_})
    local_collection.update_one({'_id':one.pop('_id')}, update={'$set':one}, upsert=True)
    i+=1
    print(i)
    pass
