
from couchdb.client import Server, Document
from collections import OrderedDict
import copy
import os
import time



#Create connection object for CouchDB connection 
server = Server('http://admin:Celmdpqosx1!@localhost:5984')
#server = Server('http://localhost:5984')

print (server)
print ("\n")

# connect to "new_database" database
dblogger = server['common_logger']

# print all available database(s) of CouchDB
for db in server:
    print (db)
print ("\n")


del server['common_logger']

