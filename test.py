import pickle
import face_recognition
import cv2
import time

# pTime = time.time()
# imgInp = face_recognition.load_image_file('./Pictures/joji.jpg')
# imgInp = cv2.cvtColor(imgInp, cv2.COLOR_BGR2RGB)
# encodeInp = face_recognition.face_encodings(imgInp)[0]
# cTime = time.time()

# print(cTime-pTime)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

pTime = time.time()
img = cv2.imread('.\\Pictures\\joji.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    crop_img = img[y:y+h, x:x+w].copy()
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    encoded = face_recognition.face_encodings(crop_img)[0]
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cTime = time.time()

# print(cTime-pTime)

cv2.imshow('img', img)
# cv2.imshow('cropped img', crop_img)
cv2.waitKey()