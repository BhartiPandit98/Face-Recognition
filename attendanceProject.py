import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#Create a list of all images in the folder rather loading invidually
path = 'imagesAttendance'
images = []
className = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])
print(className)

#funciton for computing encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeimg)
    return encodeList

#Marking the atttendance with time and name
# def markAttendance(name):
#     with open('attendance.csv', 'r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtstring = now.strftime("%H:%M:%S")
#             f.writelines(f'\n{name}, {dtstring}')


encodeListKnown = findEncodings(images)

print('Encoding Complete')

#We have to find matches between our encodings. Webcam will be used for new images to compare encodings with
#Initialize the webcam

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read() #get many images framewise from webcam
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) #to resize the images
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceLocframe = face_recognition.face_locations(imgS)
    encodeCurframe = face_recognition.face_encodings(imgS, faceLocframe)

    for encodeFace, faceLoc in zip(encodeCurframe, faceLocframe):#find the distances between know and new faces
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis) #gives the index of the image from the known images for which distance in min

        #Display the face name along with rectangle around it
        if matches[matchIndex]:
            name = className[matchIndex].upper()
            #print(name)
            y1, x2, y2, x1  = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x2+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            #markAttendance(name) #to mark attendance in the excel sheet

    #Show the image that matches
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
