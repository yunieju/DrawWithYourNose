import cv2
import numpy as np
# 1. Refactor this into main and findNose methods

class NoseDetector():
    #make it return the min, max of the x, y of nose box
    def findNose(self, img, nose_cascade):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in nose_rects:
            print(x,y,w,h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            break
        return img

    def findNosePosition(self, img, nose_cascade):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in nose_rects:
            print(x, y, w, h)
            cv2.rectangle(img, (x + w // 2, y + h // 2), (x + w // 2, y + h//2), (0, 255, 0), 3)
            break
        return img

def main():
    nose_cascade = cv2.CascadeClassifier('./haarcascade_mcs_nose.xml')
    if nose_cascade.empty():
        raise IOError('Unable to load the nose cascade classifier xml file')

    cap = cv2.VideoCapture(0)
    ds_factor = 0.5
    detector = NoseDetector()
    while True:
        ret, img = cap.read()
        img = cv2.resize(img, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        img = detector.findNosePosition(img, nose_cascade)

        cv2.imshow('Nose Detector', img)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
