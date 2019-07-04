import pyrebase
import json
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # please create .env for firebase config

FIREBASE_API_KEY = os.getenv('FIREBASE_API_KEY')
FIREBASE_AUTH_DOMAIN = os.getenv('FIREBASE_AUTH_DOMAIN')
FIREBASE_DATABASE_URL = os.getenv('FIREBASE_DATABASE_URL')
FIREBASE_STORAGE_BUCKET = os.getenv('FIREBASE_STORAGE_BUCKET')

config = {
    "apiKey": FIREBASE_API_KEY,
    "authDomain": FIREBASE_AUTH_DOMAIN,
    "databaseURL": FIREBASE_DATABASE_URL,
    "storageBucket": FIREBASE_STORAGE_BUCKET
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()


dataset_file = open('../data/adult_dataset.txt')
dataset = json.load(dataset_file)
db.child('adult_dataset').set(dataset)


instances_file = open('../data/adult_instances.txt')
data = json.load(instances_file)

e0 = []
e1 = []
for e in data:
    if (e['instance']['subject'][-1] == 0):
        e0.append(e)
    else:
        e1.append(e)
e0 = e0[:len(e1)]

mid = 8

train_e0 = e0[:mid]
train_e1 = e1[:mid]
test_e0 = e0[mid:]
test_e1 = e1[mid:]
train = train_e0 + train_e1
test = test_e0 + test_e1


db.child('adult_dataset').child('train').set(train)
db.child('adult_dataset').child('test').set(test)