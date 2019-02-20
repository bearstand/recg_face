#!/usr/bin/env python3
import sqlite3
import pickle

conn = sqlite3.connect('faces.db')

c= conn.cursor()
c.execute("drop table if exists newFaces")
c.execute("create table newFaces ( encoding blob,  imageFileId integer, firstseen integer )")

c.execute("drop table if exists knownFaces")
c.execute("create table knownFaces ( pid integer, encoding blob,  imageFileId integer, firstseen integer, lastseen integer)")

c.execute("drop table if exists persons")
c.execute("create table persons ( pid integer, name text,  firstseen integer, lastseen integer, counter integer)")

c.execute("drop table if exists config")
c.execute("create table config ( imageFileId integer)")
c.execute("insert into config values (0)")

conn.commit()
conn.close()
