import cv2
import pickle
import numpy as np
import os

# Create directory for storing data if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')

# Start video capture
video = cv2.VideoCapture(0)  # Change to 0 if 1 doesn't work for your camera
face_det = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces_data = []

# Input Aadhar number
name = input("Enter Your Aadhar Number(Unique):")

# Settings for frame capture
framesTotal = 51
captureAfterFrame = 2

frame_count = 0  # Counter for the frames

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_det.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around detected faces and capture data
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        
        # Append data if conditions are met
        if len(faces_data) <= framesTotal and frame_count % captureAfterFrame == 0:
            faces_data.append(resized_img)
        
        # Increment frame counter
        frame_count += 1

        # Display the count of captured frames
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,225),1)
    # Show the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed or the required frames are captured
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= framesTotal:
        break

# Release resources
video.release()
cv2.destroyAllWindows()

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape((framesTotal, -1))
print(faces_data)


if 'names.pkl' not in os.listdir('data/'):
    names=[name]*framesTotal
    with open("data/names.pkl", "wb") as file:
        pickle.dumps(faces_data, file)

else:
    with open("data/names.pkl", "rb") as file:
        names=pickle.load(file)
    names=[name]*framesTotal
    with open("data/names.pkl", "wb") as file:
        pickle.dump(names,file)



if 'faces_data.pkl' not in os.listdir('data/'):
    with open("data/faces_data.pkl", "wb") as file:
        pickle.dump(faces_data,file)

else:
        with open("data/faces_data.pkl", "rb") as file:
            faces=pickle.load(file)
        faces=np.append(faces,faces_data,axis=0)
        with open("data/faces_data.pkl", "wb") as file:
            pickle.dump(faces,file)


# # Save data to file
# with open(f"data/{name}.pkl", "wb") as file:
#     pickle.dump(faces_data, file)

# print(f"Data saved for Aadhar Number: {name}")
