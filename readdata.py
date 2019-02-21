#!/usr/bin/env python3
import sqlite3
import pickle


conn = sqlite3.connect('faces.db', detect_types=sqlite3.PARSE_DECLTYPES)

c= conn.cursor()
encodings=[]

c.execute("select * from newFaces")
rows=c.fetchall()
for i in rows:
    data=pickle.loads(i[0])
    encodings.append(data)

print(len(encodings))
print(encodings)

conn.commit()
conn.close()
