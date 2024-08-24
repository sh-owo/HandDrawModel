import cv2
from cvzone.HandTrackingModule import HandDetector
import math

class HandDetect:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.org = (0, 185)
        self.color = (0, 0, 255)
        self.thickness = 2
        self.hand_detector = HandDetector(maxHands=1)

    def track(self, ):
        camera = cv2.VideoCapture(0)

        while True:
            correct, img = camera.read()
            img = cv2.flip(img, 1)
            if not correct:
                break
            img_origin = img.copy()

            hands, img = self.hand_detector.findHands(img)
            current_state = "None"

            if hands and self.hand_detector.fingersUp(hands[0])[1] == 1:
                current_state = "Hand Detected"
                lmList = hands[0]['lmList']
                x, y = lmList[8][0], lmList[8][1]
                cv2.circle(img_origin, (x, y), 10, (0, 255, 0), cv2.FILLED)


            img_origin = cv2.putText(img_origin, current_state, self.org, self.font, self.fontScale,
                                     self.color, self.thickness, cv2.LINE_AA)
            cv2.imshow("main", img_origin)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        camera.release()
        cv2.destroyAllWindows()
        return False

if __name__ == "__main__":
    hand = HandDetect()
    while True:
        if not hand.track():
            break