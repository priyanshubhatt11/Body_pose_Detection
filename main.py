import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture("D:\Vs Code\Project\Body Pose\Video\\vid1.mp4")
#cap  = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    img = cv2.resize(img,(600, 400))

    results = pose.process(img)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
    mp_draw.DrawingSpec((0,0,0), 3, 3), 
    mp_draw.DrawingSpec((255,0,255), 3, 3))
    cv2.imshow("Pose Estimation", img) 

    h, w, c = img.shape
    opImg = np.zeros([h, w, c])
    opImg.fill(255)
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    mp_draw.DrawingSpec((0,0,0), 3, 3), 
    mp_draw.DrawingSpec((255,0,255), 3, 3))
    cv2.imshow("Pose", opImg)

    print(results.pose_landmarks)
   
    cv2.waitKey(3)