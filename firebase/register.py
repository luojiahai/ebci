import pyrebase
import json
import os
from dotenv import load_dotenv, find_dotenv
import random
import string

def generate_password(length=12):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

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
auth = firebase.auth()

f = open('./users.txt','w')

for i in range(100):
    email = "test{:03d}@lawkhg.com".format(i)
    password = generate_password()
    f.write(email + ',' + password + '\n')
    auth.create_user_with_email_and_password(email, password)

f.close()