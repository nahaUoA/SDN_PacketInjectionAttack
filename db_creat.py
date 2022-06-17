
from couchdb.client import Server, Document
from collections import OrderedDict
import copy
import os
import time



#Create connection object for CouchDB connection 
server = Server('http://admin:Celmdpqosx1!@localhost:5984')
#server = Server('http://localhost:5984')

print (server)

# creat/connect to "new_database" database
try:
    dblogger = server.create('common_logger')
except Exception:
    dblogger = server['common_logger']

# print all available database(s) of CouchDB
for db in server:
    print( db)
print("\n")

dblogger['ctrl_status'] = dict(core_ctrl = False, edge_ctrl = False)
dblogger['route_status'] = dict(route_info = False, route_no = 0)



