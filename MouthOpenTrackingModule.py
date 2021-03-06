# should have mouthDetect
import math
import face_recognition
import cv2

class MouthDetector():
    def get_lip_height(self, lip):
        sum = 0
        for i in [2, 3, 4]:
            # distance between two near points up and down
            distance = math.sqrt((lip[i][0] - lip[12 - i][0]) ** 2 +
                                 (lip[i][1] - lip[12 - i][1]) ** 2)
            sum += distance
        return sum / 3

    def get_mouth_height(self, top_lip, bottom_lip):
        sum = 0
        for i in [8, 9, 10]:
            # distance between two near points up and down
            distance = math.sqrt((top_lip[i][0] - bottom_lip[18 - i][0]) ** 2 +
                                 (top_lip[i][1] - bottom_lip[18 - i][1]) ** 2)
            sum += distance
        return sum / 3

    def check_mouth_open(self, top_lip, bottom_lip):
        top_lip_height = self.get_lip_height(top_lip)
        bottom_lip_height = self.get_lip_height(bottom_lip)
        mouth_height = self.get_mouth_height(top_lip, bottom_lip)

        # if mouth is open more than lip height * ratio, return true.
        ratio = 0.5
        if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
            return True
        else:
            return False

    def is_mouth_open(self, face_landmarks):
        top_lip = face_landmarks['top_lip']
        bottom_lip = face_landmarks['bottom_lip']

        top_lip_height = self.get_lip_height(top_lip)
        bottom_lip_height = self.get_lip_height(bottom_lip)
        mouth_height = self.get_mouth_height(top_lip, bottom_lip)

        # if mouth is open more than lip height * ratio, return true.
        ratio = 0.5
        print('top_lip_height: %.2f, bottom_lip_height: %.2f, mouth_height: %.2f, min*ratio: %.2f'
              % (top_lip_height, bottom_lip_height, mouth_height, min(top_lip_height, bottom_lip_height) * ratio))

        if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
            return True
        else:
            return False