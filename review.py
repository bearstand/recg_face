#!/usr/bin/env python3
import sqlite3
import os
from PIL import Image


conn = sqlite3.connect('.recdata/faces.db')

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
    src_filename=".recdata/newfaces/"+str(i[1])+".jpg"
    dst_filename=".recdata/knownfaces/"+str(i[1])+".jpg"
    Image.open(src_filename).show()
    
    text = input("If you want to save it give it a name:")
    if ( len(text) > 0):
        maxpid=maxpid+1
        c.execute("insert into knownfaces( pid, encoding, imageFileId, firstseen, lastseen) values( ?,?,?,?,?)", ( maxpid, i[0], i[1], i[2], i[2] ))
        c.execute("insert into persons values( ?,?,?,?,?)", ( maxpid, text,  i[2], i[2], 1 ))
        os.rename(src_filename, dst_filename)
    else:
        os.remove(src_filename)

c.execute("delete from newFaces")

conn.commit()
conn.close()
