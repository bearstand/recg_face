#!/usr/bin/env python3
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
print("len of encodings:", len(encodings))
print("len of names:", len(names))
print("encodings[0]", encodings[0])
print("len of encodings[0]", len(encodings[0]))
