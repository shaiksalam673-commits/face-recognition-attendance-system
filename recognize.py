import cv2
import csv
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

names = {1: "Salam"}  # add your names here

def mark_attendance(name):
    with open('attendance.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        time_now = datetime.now().strftime('%H:%M:%S')
        writer.writerow([name, time_now])

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if confidence < 70:
            name = names.get(id, "Unknown")
            mark_attendance(name)
        else:
            name = "Unknown"

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Recognition', img)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
