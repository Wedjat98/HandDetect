import cv2
import mediapipe as mp
import time

mpHand = mp.solutions.hands
cap = cv2.VideoCapture(1)
# poses = mpPose.Pose(model_complexity=0)
hand = mpHand.Hands(model_complexity=0)
mpDraw = mp.solutions.drawing_utils
# handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=12)
# handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)
pTime = 0
cTime = 0

while 1:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hand.process(imgRGB)
        if result.multi_hand_landmarks:

            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

            # print(result.multi_hand_landmarks)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
