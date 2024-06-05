import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pyautogui

# Directory containing the images for attendance
path = 'imageAttendance'
images = []
classNames = []

# Load images and their corresponding class names
myList = os.listdir(path)
print(f"Images found: {myList}")
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"Error loading image: {cl}")
print(f"Class names: {classNames}")


def findEncodings(images):
    """Encodes a list of images."""
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError as e:
            print(f"Error encoding image: {e}")
    return encodeList


def markAttendance(name):
    """Marks the attendance of the identified person."""
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            time_now = datetime.now()
            tString = time_now.strftime('%H:%M:%S')
            dString = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tString},{dString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
# Max PROP CAP (WORKS :) )
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Frame Size
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Frame size

actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f'Actual frame size: {actual_width}x{actual_height}')  # This depends on the MP of your camera...


# Fail safe trigger function
def is_failsafe_triggered():
    """Checks if the failsafe condition is met (mouse cursor at the top-right corner)."""
    screen_width, screen_height = pyautogui.size()
    cursor_x, cursor_y = pyautogui.position()
    return cursor_x >= screen_width - 1 and cursor_y <= 1


# More Variables
processedNames = set()
frame_count = 0
process_frame_interval = 60  # Captures every 60th frame

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    # New Capturing Method
    frame_count += 1
    if frame_count % process_frame_interval != 0:

        # Video Capture
        cv2.imshow('Webcam', img)
        cv2.flip(img, 1)
        if cv2.waitKey(10) == 13 or is_failsafe_triggered():
            print("Fail Safe Init!...")
            break
        continue

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            if name not in processedNames:
                print(f"Match found: {name}")
                processedNames.add(name)
                y1, x2, y2, x1 = [v * 4 for v in faceLoc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                markAttendance(name)

    # Re-Configured to Return Key ( Enter Key )
    if cv2.waitKey(10) == 13 or is_failsafe_triggered():
        print("Exiting...")
        break

# OLD CAPTURING METHOD
#
# imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#
#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
#
# for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#     matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#     faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#
#         matchIndex = np.argmin(faceDis)
#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             print(f"Match found: {name}")
#
#             y1, x2, y2, x1 = [v * 4 for v in faceLoc]
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#             markAttendance(name)
#
#     cv2.imshow('Webcam', img)
#
#     if cv2.waitKey(10) == 13 or is_failsafe_triggered():
#         print("Fail Safe Init!...")
#         break
#
cap.release()
cv2.destroyAllWindows()
