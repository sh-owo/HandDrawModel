import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import torch
import CNN as cnn
import math

width: int
height: int
draw_table: np.ndarray
current_state: str = "None"

class HandDetect:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.org = (0, 185)
        self.color = (0, 0, 255)
        self.thickness = 2
        self.hand_detector = HandDetector(maxHands=1)
        self.model_path = 'model.pth'

        # Load or create model
        if os.path.exists(self.model_path):
            self.model = cnn.Net()
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
        else:
            cnn.train()
            self.model = cnn.Net()
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()

    def track(self):
        global width, height, draw_table, current_state

        # Initialize camera
        camera = cv2.VideoCapture(0)

        # Check if the camera opened successfully
        if not camera.isOpened():
            print("Error: Could not open camera.")
            return False

        # Get first frame to initialize draw_table
        correct, img = camera.read()
        if correct:
            height, width, _ = img.shape
            draw_table = np.zeros((height, width), dtype=np.uint8)

        while True:
            correct, img = camera.read()
            img = cv2.flip(img, 1)
            if not correct:
                break
            img_origin = img.copy()

            hands, img = self.hand_detector.findHands(img)


            if hands and self.hand_detector.fingersUp(hands[0])[1] == 1 and self.hand_detector.fingersUp(hands[0])[0] == 0:
                lmList = hands[0]['lmList']
                x, y = lmList[8][0], lmList[8][1]
                cv2.circle(img_origin, (x, y), 35, (0, 255, 0), cv2.FILLED)
                for i in range(-30, 30):
                    for j in range(-30, 30):
                        if math.sqrt(i ** 2 + j ** 2) < 30:
                            draw_table[y + i, x + j] = 255

            # Combine the draw_table with the original image
            img_origin[draw_table == 255] = (255, 255, 255)

            input_tensor = preprocess_draw_table(draw_table)
            with torch.no_grad():
                output = self.model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                current_state = f'Prediction: {predicted.item()}'

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # elif key == ord('f'):

            elif key == ord('r'):
                draw_table.fill(0)
                current_state = "None"

            img_origin = cv2.putText(img_origin, current_state, self.org, self.font, self.fontScale,
                                     self.color, self.thickness, cv2.LINE_AA)
            cv2.imshow("main", img_origin)

        camera.release()
        cv2.destroyAllWindows()

        return False  # Ensure the loop continues

def preprocess_draw_table(draw_table):
    coords = cv2.findNonZero(draw_table)
    if coords is None:
        return torch.zeros((1, 1, 28, 28), dtype=torch.float32)  # Return a blank tensor if no drawing is detected

    x, y, w, h = cv2.boundingRect(coords)
    cropped_table = draw_table[y:y + h, x:x + w]
    resized_table = cv2.resize(cropped_table, (28, 28))
    input_tensor = torch.tensor(resized_table, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    input_tensor = (input_tensor - 0.5) / 0.5
    return input_tensor

if __name__ == "__main__":
    hand = HandDetect()
    while True:
        if not hand.track():
            break
