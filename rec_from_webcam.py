#!/usr/bin/env python3
import face_recognition
import cv2
import numpy as np
from PIL import Image
import pickle
import os
import sqlite3

conn = sqlite3.connect('.recdata/faces.db')
c= conn.cursor()
c.execute("select imageFileId from config")
max_file_id=c.fetchone()[0]

def middle_point(loc ):
    (t, r, b, l) = loc
    return ( (t+b)/2, (r+l)/2)

def distance_of_points( p1, p2 ):
    (x1,y1) = p1
    (x2,y2) = p2
    return ( (x1-x2)**2 + (y1-y2)**2 )

def distance_of_locations( loc1, loc2 ):
    return( distance_of_points( middle_point(loc1), middle_point(loc2)))
    
MAX_DISTANCE=500

class UnknownFaces:
    def __init__(self):
        self.images=[]
        self.locations=[]
        self.encodings=[]


    def add( self, image, location, encoding ):
        self.images.append(image)
        self.locations.append(location)
        self.encodings.append(encoding)
        #print("new unknown face inserted:", len(self.locations))
        return True

    def remove( self, location ):
        lenth=len(self.locations)
        i=0
        while i<lenth:
            if ( distance_of_locations( self.locations[i], location ) < MAX_DISTANCE ): 
                #print("unknown face removed:", len(self.locations))
                del self.locations[i]
                del self.images[i]
                lenth=lenth-1
            i=i+1

    def count_of_same_position(self, location ):
        count=0
        for l in self.locations:
            if ( distance_of_locations( l, location ) < MAX_DISTANCE ):
                count+=1
        return count

    def get_the_best_image(self, location):
        indexes=[]
        encodings=[]
        for i in range(len(self.locations)):
            if ( distance_of_locations( self.locations[i], location ) < MAX_DISTANCE ):
                indexes.append(i)
                encodings.append(self.encodings[i])

        av=np.average(encodings)
        min_dist=999999
        min_pointer=0
        for i in range(len(encodings)):
            dist=np.linalg.norm(av-encodings[i])
            if dist < min_dist:
                min_pointer=i
                min_dist=dist
        return self.images[indexes[min_pointer]], self.encodings[indexes[min_pointer]] 

    def clear():
        self.images=[]
        self.locations=[]

def is_small( location):
    (top, right, bottom, left) = location
    if ( bottom-top < 30 or right-left < 30 ):
        return True
    else:
        return False

def draw_rectangle( frame, location, name ):
    (top, right, bottom, left) =location 
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    if ( name == "new" ):
        color=(0, 0, 255)
    else:
        color=(255, 0, 0)

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom + 35), (right, bottom), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom + 29), font, 1.0, (255, 255, 255), 1)
    return

def blurValue(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b= cv2.Laplacian(gray, cv2.CV_64F).var()
    return b

def show_image( frame ):
    Image.fromarray(frame).show()
    return

def init_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def init_dirs():
    init_dir(".recdata/newfaces")
    init_dir(".recdata/knownfaces")

        
def load_data():
    encodings=[]
    names=[]
    c.execute("select knownFaces.encoding, persons.name from knownFaces, persons where knownFaces.pid=persons.pid")
    rows=c.fetchall()
    for r in rows:
        encodings.append(pickle.loads(r[0]))
        names.append(r[1])
    return encodings, names

def save_data( encodings, names,files ):
    with open('data.pkl', 'wb') as output:
        pickle.dump(encodings, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(names, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(files, output, pickle.HIGHEST_PROTOCOL)
    return

def get_face_image( frame, location):
    (top, right, bottom, left) =location 
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    image=frame[top:bottom, left:right]
    rgb_image = image[:, :, ::-1]
    return rgb_image

    
def get_better_face_encoding( image ):
    encodings = face_recognition.face_encodings(image, known_face_locations=None, num_jitters=5)
    return encodings[0]

def save_new_faces( image, encoding):
        global max_file_id
        pdata=pickle.dumps(encoding, pickle.HIGHEST_PROTOCOL)
        max_file_id+=1
        Image.fromarray(image).save(".recdata/newfaces/"+str(max_file_id)+".jpg")
        c.execute(" insert into newFaces values ( ?, ?, strftime('%s','now'))", (sqlite3.Binary(pdata), max_file_id ))
        c.execute("update config set imageFileId=?", ( max_file_id, ))
        conn.commit()

init_dirs()
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

known_face_encodings, known_face_names = load_data( )

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
unknown_faces=UnknownFaces()
average_blur=500

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        #remove_small_faces( face_locations )
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for f in range(len(face_encodings)):
            face_encoding = face_encodings[f]
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "new"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                unknown_faces.remove(face_locations[f])
            elif not is_small(face_locations[f]):
                image=get_face_image( frame, face_locations[f]) 
                b=blurValue(image)
                average_blur=0.9*average_blur+0.1*b
                #print("average_blur:", average_blur, " b:", b)
                if ( b  > average_blur ):
                    unknown_faces.add(image,face_locations[f],face_encoding )
                    if( unknown_faces.count_of_same_position(face_locations[f]) >=5 ):
                        image2, encoding=unknown_faces.get_the_best_image(face_locations[f])
                        save_new_face(image2, encoding) 
                        unknown_faces.remove(face_locations[f])
                        known_face_encodings.append(encoding)
                        known_face_names.append(name)
                        print("add one new face")

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for location, name in zip(face_locations, face_names):
        draw_rectangle( frame, location, name )

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    key=cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        conn.commit()
        conn.close()
        break
    elif key & 0xFF == ord('l'):
        print("data reloaded")
        known_face_encodings, known_face_names = load_data()


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
