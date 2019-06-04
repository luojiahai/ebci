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
train = data[:len(data)//2]
test = data[len(data)//2:]

db.child('adult_dataset').child('train').set(train)
db.child('adult_dataset').child('test').set(test)