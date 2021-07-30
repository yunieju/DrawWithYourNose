import cv2
import numpy as np
import time
import os
import MouthOpenTrackingModule
import face_recognition

drawColor = (255, 0, 255)
brushThickness = 15
eraserThickness = 100

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# default start point
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)
mouthDetector = MouthOpenTrackingModule.MouthDetector()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    # Find all the faces and face encodings in the frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(frame)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings,
                                                                         face_landmarks_list):
        # Display text for mouth open / close
        ret_mouth_open = mouthDetector.is_mouth_open(face_landmarks)

        # Find a nose tip
        nose = face_landmarks['nose_tip']
        noseX = [x[0] for x in nose]
        noseY = [x[1] for x in nose]
        x1, y1 = int(sum(noseX) / len(noseX)), int(sum(noseY) / len(noseY))

        # Selection Mode / Stop Drawing
        if ret_mouth_open is True:
            text = 'Drawing Stopped'
            # Restart from the last point
            xp, yp = x1, y1

        # Drawing Mode
        else:
            text = 'Drawing Mode'
            cv2.circle(frame, (x1, y1), 5, drawColor, 10)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(frame, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
        cv2.putText(frame, text, (580, 700), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # blend frame and drawing image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, imgCanvas)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()