#!/usr/bin/env python3
import sqlite3
from PIL import Image


conn = sqlite3.connect('faces.db')

c= conn.cursor()
encodings=[]
c.execute("select MAX(pid) from persons")
row=c.fetchone()
if ( row[0] == None ):
    maxpid=0
else:
    maxpid=row[0]

print(maxpid)

c.execute("select * from newFaces")
rows=c.fetchall()
for i in rows:
    filename="pictures/"+str(i[1])+".jpg"
    Image.open(filename).show()
    
    text = input("If you want to save it give it a name:")
    if ( len(text) > 0):
        maxpid=maxpid+1
        c.execute("insert into knownfaces( pid, encoding, imageFileId, firstseen, lastseen) values( ?,?,?,?,?)", ( maxpid, i[0], i[1], i[2], i[2] ))
        c.execute("insert into persons values( ?,?,?,?,?)", ( maxpid, text,  i[2], i[2], 1 ))

c.execute("delete from newFaces")

conn.commit()
conn.close()
