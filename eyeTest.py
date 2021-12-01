import cv2
import numpy as np
import face_recognition
import time

def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

Kernal = np.ones((3, 3), np.uint8)

detectorPaths = {
	"face": "haarcascade_frontalface_default.xml",
	# "eyes": "haarcascade_eye.xml",
}
detectors = {}
for (name, path) in detectorPaths.items():
	detectors[name] = cv2.CascadeClassifier(path)

pTime = time.time()
img = cv2.imread('./Pictures/fake.jpg')
img = cv2.resize(img, (0, 0), fx=1, fy=1)
# print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray,5,1,1)
faces = detectors['face'].detectMultiScale(gray, 1.3, 5)

left_eyes = []
right_eyes = []
count = 0
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    face_landmarks_list = face_recognition.face_landmarks(roi_color)[0]

    start_left = face_landmarks_list['left_eye'][0]
    end_left = face_landmarks_list['left_eye'][3]
    width_left = int((end_left[0] - start_left[0])/2)
    center_left = (int((start_left[0] + end_left[0])/2), int((start_left[1] + end_left[1])/2))
    roi_color = cv2.circle(roi_color, center_left, 0, (255, 0, 0), 10)
    cv2.rectangle(roi_color, (center_left[0]-width_left, center_left[1]-width_left), (center_left[0]+width_left, center_left[1]+width_left), (255, 0, 0), 2)
    
    start_right = face_landmarks_list['right_eye'][0]
    end_right = face_landmarks_list['right_eye'][3]
    width_right = int((end_right[0] - start_right[0])/2)
    center_right = (int((start_right[0] + end_right[0])/2), int((start_right[1] + end_right[1])/2))
    roi_color = cv2.circle(roi_color, center_right, 0, (255, 0, 0), 10)
    cv2.rectangle(roi_color, (center_right[0]-width_right, center_right[1]-width_right), (center_right[0]+width_right, center_right[1]+width_right), (255, 0, 0), 2)

    roi_left = roi_gray[center_left[1]-width_left : center_left[1]+width_left, 
                        center_left[0]-width_left : center_left[0]+width_left]
    blur_left = cv2.GaussianBlur(roi_left,(7,7),0)
    # _, binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, binary_left = cv2.threshold(blur_left, 60, 255, cv2.THRESH_BINARY)
    # binary = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    left_eyes.append(binary_left.tobytes())
    cv2.imshow('left eye', binary_left)

    roi_right = roi_gray[center_right[1]-width_right : center_right[1]+width_right, 
                        center_right[0]-width_right : center_right[0]+width_right]
    blur_right = cv2.GaussianBlur(roi_right,(5,5),0)
    # _, binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, binary_right = cv2.threshold(blur_right, 60, 255, cv2.THRESH_BINARY)
    # binary_right = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    right_eyes.append(binary_right.tobytes())
    cTime = time.time()
    print('Detect time:', cTime-pTime)
    # cv2.imshow('right eye', binary_right)
    # cv2.waitKey()
    
    # else:
    #     pass
# print(hamming_distance(left_eyes[0], right_eyes[0]))

cv2.imshow('face', img)
cv2.waitKey()