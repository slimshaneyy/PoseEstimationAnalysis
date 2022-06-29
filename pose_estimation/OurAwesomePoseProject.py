import cv2
import time
import PoseModule as pm
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('PoseVideos/poseestimation5.MOV')
pTime = 0
detector = pm.poseDetector()

left_knee_angle_over_time = []


def calculate_angle(a, b, c):
    a = np.array(a)  # Proximal
    b = np.array(b)  # Middle
    c = np.array(c)  # Distal

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle
    print(angle)
    left_knee_angle_over_time.append(angle)


while cap.isOpened():
    if cv2.waitKey(1) == ord('q'):
        break
    if not cap.isOpened():
        break
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    right_elbow = lmList[14]
    left_elbow = lmList[13]
    right_shoulder = lmList[12]
    left_shoulder = lmList[11]
    right_hip = lmList[24]
    left_hip = lmList[23]
    left_knee = lmList[25]
    right_knee = lmList[26]
    right_ankle = lmList[28]
    left_ankle = lmList[27]
    right_foot_index = lmList[32]
    left_foot_index = lmList[31]
    calculate_angle(left_hip, left_knee, left_ankle)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

plt.plot(left_knee_angle_over_time)
plt.xlabel('Time')
plt.ylabel('Knee Angle: Degrees')
plt.title('Knee Angle Over Time')
plt.show()