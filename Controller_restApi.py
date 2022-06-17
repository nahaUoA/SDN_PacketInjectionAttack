from couchdb.client import Server, Document
from collections import OrderedDict
import copy
import os
import time
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
        


#Create connection object for CouchDB connection 
server = Server('http://admin:Celmdpqosx1!@localhost:5984')
#server = Server('http://localhost:5984')

dblogger = server['common_logger']
print (dblogger)


class TodoSimple(Resource):

    def __init__(self, *args, **kwargs):
        super(TodoSimple, self).__init__(*args, **kwargs)
        self.route_id = 0

    def get(self, route_id):
        self.route_id = route_id
        doc_DB = dblogger['route_status']
        doc_DB['route_info'] = True 
        doc_DB['route_no'] = self.route_id 
        dblogger.save(doc_DB)
        return {'route_id': self.route_id}

    def put(self, route_id):
        self.route_id = route_id
        doc_DB = dblogger['route_status']
        doc_DB['route_info'] = True 
        doc_DB['route_no'] = self.route_id 
        dblogger.save(doc_DB)
        return {'route_id': self.route_id}

    def post(self, route_id):
        self.route_id = route_id
        doc_DB = dblogger['route_status']
        doc_DB['route_info'] = True 
        doc_DB['route_no'] = self.route_id 
        dblogger.save(doc_DB)
        return {'route_id': self.route_id}
        

api.add_resource(TodoSimple, '/route_id/<string:route_id>')

if __name__ == '__main__':
    app.run(debug=True)
