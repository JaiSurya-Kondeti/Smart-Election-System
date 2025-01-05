from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

# Speak function
def speak(text):
    speaker = Dispatch('SAPI.SpVoice')
    speaker.Speak(text)

# Initialize video capture and face detection
video = cv2.VideoCapture(0)  # Use 0 for the primary camera
face_det = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure data directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Load face data and labels
with open('data/names.pkl', 'rb') as file:
    LABELS = pickle.load(file)
    

with open('data/faces_data.pkl', 'rb') as file:
    FACES = pickle.load(file)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
imgBg = cv2.imread("Vote_hero.jpg")

# Column names for CSV
COL_NAMES= ['NAME', 'VOTE', 'DATE', 'TIME']
csv_file = "Votes.csv"

# Check if the CSV file exists and create it if not
if not os.path.isfile(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(COL_NAMES)  # Write header

# Main loop
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_det.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)  # Predict the label
        t = time.time()

        # Get current date and time
        date = datetime.fromtimestamp(t).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(t).strftime("%H:%M:%S")
        exist=os.path.isfile('Votes'+'.csv')
        # Prepare vote data
        vote = [output[0], "VOTED", date, timestamp]

        # Write to CSV file
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(vote)

        # Speak vote confirmation
        speak(f"Vote recorded for {output[0]}")

        # Draw rectangle and label on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Overlay the frame on the background image
    #imgBg[370:370 + 480, 225:225 + 640] = frame

    # Display the frame
    cv2.imshow('frame', imgBg)

    # Exit on 'q' key press
    k = cv2.waitKey(1)
    def check_if_exists(value):

        try:
            with open("Votes.csv",'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row and row[0] == value:
                        return True
        except FileNotFoundError:
            print("File not found or unable to Open the CSV File.")
        return False
    voter_exist=check_if_exists(output[0])
    if voter_exist:
        speak("YOU HAVE ALREADY VOTED")
        break
    if k==ord('1'):
        speak('YOUR VOTE HAS BEEN RECOREDED')
        time.sleep(5)
        if exist:
            with open("votes" + ".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                attendance=[output[0],"SACET",date,timestamp]
                writer.writerow(attendance)
            csvfile.close() 
        else:
             with open("votes" + ".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                attendance=[output[0],"SACET",date,timestamp]
                writer.writerow(attendance)
                csvfile.close() 
        speak("THANK YOT FOR YOUR VOTE")
    if voter_exist:
        speak("YOU HAVE ALREADY VOTED")
    if k==ord('2'):
        speak('YOUR VOTE HAS BEEN RECOREDED')
        time.sleep(5)  
        if exist:
            with open("votes" + ".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                attendance=[output[0],"BEC",date,timestamp]
                writer.writerow(attendance)
            csvfile.close() 
        else:
             with open("votes" + ".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                attendance=[output[0],"BEC",date,timestamp]
                writer.writerow(attendance)
                csvfile.close() 
        speak("THANK YOT FOR YOUR VOTE")




# Release resources
video.release()
cv2.destroyAllWindows()
