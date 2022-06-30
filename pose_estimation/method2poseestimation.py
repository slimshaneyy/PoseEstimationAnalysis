import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyinputplus as pyip

# body_parts = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_wrist',
#                               'left_elbow', 'left_shoulder', 'right_hip', 'right_knee',
#                               'right_ankle', 'right_foot', 'left_hip', 'left_knee', 'left_ankle',
#                               'left_foot']
# print("Welcome to the joint angle analyser\n")
# print("Please choose the proximal joint ")
# first_joint = pyip.inputMenu(body_parts)
# print("Please choose the middle joint ")
# second_joint = pyip.inputMenu(body_parts)
# print("Please choose the distal joint ")
# third_joint = pyip.inputMenu(body_parts)
# print(first_joint)
angle_over_time = []



def calculate_angle(a, b, c):
    a = np.array(a)  # Proximal
    b = np.array(b)  # Middle
    c = np.array(c)  # Distal

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    angle_over_time.append(angle)

    return angle



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#  VIDEO FEED
cap = cv2.VideoCapture('PoseVideos/poseestimation7.MOV')
# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        #  Recolor Image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    
        # Make Detection
        results = pose.process(image)
    
        #  Recolor Image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #  Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get Coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            # Calculate Angle
            angle = int(calculate_angle(right_hip, right_knee, right_ankle))

            # Visualize Angle
            image = cv2.putText(image, "Knee Angle: " + str(angle), (800, 950),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


        except:
            pass
    
        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    angle_max = int(max(angle_over_time))
    angle_min = int(min(angle_over_time))
    print(f'Max angle: {angle_max} degrees')
    print(f'Min angle: {angle_min} degrees')
    plt.plot(angle_over_time)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    plt.ylabel('Angle: Degrees')
    plt.title('Angle Throughout Movement')
    plt.show()

