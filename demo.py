import cv2
import face_recognition

imgInp = face_recognition.load_image_file('./Pictures/joji.jpg')
imgInp = cv2.cvtColor(imgInp, cv2.COLOR_BGR2RGB)

imgTrue = face_recognition.load_image_file('./Pictures/pinkguy.jpg')
imgTrue = cv2.cvtColor(imgTrue, cv2.COLOR_BGR2RGB)

imgFalse = face_recognition.load_image_file('./Pictures/max.jpg')
imgFalse = cv2.cvtColor(imgFalse, cv2.COLOR_BGR2RGB)

faceLocInp = face_recognition.face_locations(imgInp)[0]
encodeInp = face_recognition.face_encodings(imgInp)[0]
cv2.rectangle(imgInp, (faceLocInp[3], faceLocInp[0]), (faceLocInp[1], faceLocInp[2]), (255, 0, 0), 2)

faceLocTrue = face_recognition.face_locations(imgTrue)[0]
encodeTrue = face_recognition.face_encodings(imgTrue)[0]
cv2.rectangle(imgTrue, (faceLocTrue[3], faceLocTrue[0]), (faceLocTrue[1], faceLocTrue[2]), (255, 0, 0), 2)

faceLocFalse = face_recognition.face_locations(imgFalse)[0]
encodeFalse = face_recognition.face_encodings(imgFalse)[0]
cv2.rectangle(imgFalse, (faceLocFalse[3], faceLocFalse[0]), (faceLocFalse[1], faceLocFalse[2]), (255, 0, 0), 2)

results = face_recognition.compare_faces([encodeTrue, encodeFalse], encodeInp)
distance = face_recognition.face_distance([encodeTrue, encodeFalse], encodeInp)
print(results, distance)

cv2.imshow('Joji', imgInp)
cv2.imshow('Pink Guy', imgTrue)
cv2.imshow('Maxmoefoe', imgFalse)
cv2.waitKey(0)