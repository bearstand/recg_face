#!/usr/bin/env python3
import face_recognition
import cv2
import numpy as np
from PIL import Image
import pickle

def remove_small_faces( face_locations ):
    l=len(face_locations)
    i=0
    while i<l:
        (top, right, bottom, left) = face_locations[i]
        if ( bottom-top < 50 or right-left < 50 ):
            face_locations.remove(face_locations[i])
            l=l-1
        i+=1
    return

def draw_rectangle( frame, location,name ):
    (top, right, bottom, left) =location 
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    if ( name == "Unknown" ):
        color=(0, 0, 255)
    else:
        color=(255, 0, 0)
    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom + 35), (right, bottom), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom + 29), font, 1.0, (255, 255, 255), 1)
    return

def blurValue(frame, location): 
    (top, right, bottom, left) =location 
    image=frame[top:bottom, left:right]
    #Image.fromarray(image).show()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b= cv2.Laplacian(gray, cv2.CV_64F).var()

    return b

def show_image( frame ):
    Image.fromarray(frame).show()
    return

def save_image( image, filename ):
    Image.fromarray(image).save("pictures/"+filename+".jpg")
    return

def load_data():
    with open('data.pkl', 'rb') as input:
        encodings = pickle.load(input)
        names = pickle.load(input)
    return encodings, names

def save_data( encodings, names ):
    with open('data.pkl', 'wb') as output:
        pickle.dump(encodings, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(names, output, pickle.HIGHEST_PROTOCOL)
    return
    
def get_better_face_encoding( frame, location):
    (top, right, bottom, left) =location 
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    image=frame[top:bottom, left:right]
    rgb_frame = image[:, :, ::-1]
    new_location=(0,right-left,bottom-top,0)
    encodings = face_recognition.face_encodings(rgb_frame, [new_location], num_jitters=5)
    return encodings[0], rgb_frame

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

#known_face_encodings, known_face_names = load_data( )

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

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
        remove_small_faces( face_locations )
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for f in range(len(face_encodings)):
            face_encoding = face_encodings[f]
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                draw_rectangle( frame, face_locations[f], "unknown" )
                cv2.imshow('Video', frame)
                b=blurValue(rgb_small_frame, face_locations[f])
                print("blur:", b)
                if ( b  > 1000 ):
                    encoding, image=get_better_face_encoding( frame, face_locations[f])
                    show_image(image)
                    text = input("what's your name?")
                    if ( len(text) > 0 ):
                            known_face_encodings.append(encoding)
                            known_face_names.append(text)
                            save_image(image, text+"_"+str(len(known_face_encodings))) 

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for location, name in zip(face_locations, face_names):
        draw_rectangle( frame, location, name )

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        save_data( known_face_encodings, known_face_names )
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
