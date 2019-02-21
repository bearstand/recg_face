#!/usr/bin/env python3
import sqlite3
import pickle

def load_data():
    try:
        with open('data.pkl', 'rb') as input:
            encodings = pickle.load(input)
            names = pickle.load(input)
        return encodings, names
    except FileNotFoundError:
        return [],[]

encodings, names = load_data()

conn = sqlite3.connect('faces.db')

c= conn.cursor()

c.execute(" delete from newFaces")

pdata=pickle.dumps(encodings[2], pickle.HIGHEST_PROTOCOL)
c.execute(" insert into newFaces(encoding,imageFileId) values ( ?, ?)", (sqlite3.Binary(pdata), 2))

pdata=pickle.dumps(encodings[1], pickle.HIGHEST_PROTOCOL)
c.execute(" insert into newFaces(encoding,imageFileId) values ( ?, ?)", (sqlite3.Binary(pdata), 1))
print(encodings)
conn.commit()
conn.close()
