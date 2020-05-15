from __future__ import print_function
import cv2 as cv
import argparse

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('../opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
eyes_cascade = cv.CascadeClassifier('../opencv/data/haarcascades/haarcascade_eye.xml')

#img = cv.imread('people.jpg')
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    print('faces: ',len(faces))

    for (x,y,w,h) in faces:
        # center = (x + w//2, y + h//2)
        # img = cv.ellipse(img, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        cv.rectangle(frame,(x,y), (x+w, y+h), (0,0,255),2)
        eye_gray = gray[y:y+h, x:x+w]
        eye_color = frame[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(eye_gray)
        for(ex,ey,ew,eh) in eyes:
                eye_center = (x + ex + ew//2, y + ey + eh//2)
                radius = int(round((ew + eh)*0.25))
                frame = cv.circle(frame, eye_center, radius, (9,255, 0 ), 4)
            #cv.rectangle(eye_color, (ex,ey), (ex+ew, ey+eh), (0,255,0),2)

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destryAllWindows()
