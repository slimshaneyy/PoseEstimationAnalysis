import numpy as np

left_hip = [533, 934]
left_knee = [515, 1277]
left_ankle = [566, 1623]



def calculate_angle(a, b, c):
    a = np.array(a)  # Proximal
    b = np.array(b)  # Middle
    c = np.array(c)  # Distal

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    print(angle)




calculate_angle(left_hip, left_knee, left_ankle)

