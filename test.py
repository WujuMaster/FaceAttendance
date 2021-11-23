import pickle
import face_recognition
import cv2
import time
import numpy as np

from scipy.spatial import distance as dist

# pTime = time.time()
# imgInp = face_recognition.load_image_file('./Pictures/joji.jpg')
# imgInp = cv2.cvtColor(imgInp, cv2.COLOR_BGR2RGB)
# encodeInp = face_recognition.face_encodings(imgInp)[0]
# cTime = time.time()

# print(cTime-pTime)


# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# pTime = time.time()
# img = cv2.imread('.\\Pictures\\joji.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# for (x, y, w, h) in faces:
#     crop_img = img[y:y+h, x:x+w].copy()
#     crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
#     encoded = face_recognition.face_encodings(crop_img)[0]
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# cTime = time.time()

# # print(cTime-pTime)

# cv2.imshow('img', img)
# # cv2.imshow('cropped img', crop_img)
# cv2.waitKey()
detectorPaths = {
	"face": "haarcascade_frontalface_default.xml",
	# "eyes": "haarcascade_eye.xml",
}
detectors = {}
for (name, path) in detectorPaths.items():
	detectors[name] = cv2.CascadeClassifier(path)

Kernal = np.ones((3, 3), np.uint8)

left_eyes = []
right_eyes = []

# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import model_from_json
# json_file = open('antispoofing_model.json','r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load antispoofing model weights 
# model.load_weights('antispoofing_model.h5')
# print("Model loaded from disk")


def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2
COUNTER = 0
TOTAL = 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
pTime = 0
while 1:
    ret, img = cap.read()
    rate = 3
    small_img = cv2.resize(img, (0, 0), fx=round(1/rate, 2), fy=round(1/rate, 2))
    norm_img = np.zeros((small_img.shape[0], small_img.shape[1]))
    small_img = cv2.normalize(small_img, norm_img, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,5,1,1)
    faces = detectors['face'].detectMultiScale(gray, 1.3, 5)
    try:
        for (x,y,w,h) in faces:
            
            # roi_color = img[y*rate-5:(y+h)*rate+5, x*rate-5:(x+w)*rate+5]
            # resized_face = cv2.resize(roi_color,(160,160))
            # resized_face = resized_face.astype(float) / 255.0
            # resized_face = img_to_array(resized_face)
            # resized_face = np.expand_dims(resized_face, axis=0)
            # preds = model.predict(resized_face)[0]
            # # print(preds)
            # if preds> 0.5:
            #     label = 'spoof'
            #     cv2.putText(img, label, (x*rate,y*rate - 10),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            #     cv2.rectangle(img, (x*rate,y*rate),((x+w)*rate,(y+h)*rate),
            #         (0, 0, 255), 2)
            # else:
            #     label = 'real'
            #     cv2.putText(img, label, (x*rate,y*rate - 10),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            #     cv2.rectangle(img, (x*rate,y*rate),((x+w)*rate,(y+h)*rate),
            #     (0, 255, 0), 2)

            # if preds > 0.5:
            #     label = 'spoof'
            #     color = (0,0,255)
            # else:
            #     label = 'real'
            #     color = (0, 255, 0)
            # cv2.putText(img, label, (x*rate,y*rate - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            #     0.5, color, 2)
            # cv2.rectangle(img, (x*rate,y*rate),((x+w)*rate,(y+h)*rate),
            #     color, 2)

            # roi_gray = gray[y:y+h, x:x+w]
            face = img[y*rate:(y+h)*rate, x*rate:(x+w)*rate].copy()
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_landmarks_list = face_recognition.face_landmarks(face)[0]
            left_eye = face_landmarks_list['left_eye']
            right_eye = face_landmarks_list['right_eye']
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                print("eye_closed")
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0
                print("reset")
            cv2.putText(img, "Blinks: {}".format(TOTAL), (20, 30),
			    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "EAR: {:.2f}".format(ear), (300, 30),
			    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            cv2.rectangle(img,(x*rate,y*rate),((x+w)*rate,(y+h)*rate),(255,0,0),2)
            # eyes = detectors['eyes'].detectMultiScale(roi_gray, 1.3, 5)
            # face_center = (x + int(0.5*w), y + int(0.5*h))

            # if (len(eyes)>=2):
            #     for (ex,ey,ew,eh) in eyes:
            #         eye_center = (ex+int(0.5*ew), ey+int(0.5*eh))
            #         roi_color = cv2.circle(roi_color, eye_center, 0, (0, 255, 0), 10)
            #         # eye_start = (ex,ey+int(0.25*eh))
            #         # eye_end = (ex+ew,ey+int(0.75*eh))
            #         eye_start = (eye_center[0] - 20, eye_center[1] - 10)
            #         eye_end = (eye_center[0] + 20, eye_center[1] + 10)
            #         cv2.rectangle(roi_color, eye_start, eye_end, (0,255,0), 2)
            #         eyes_roi = roi_gray[ey: ey+eh, ex:ex + ew]
            #         _, binary = cv2.threshold(eyes_roi, 60, 255, cv2.THRESH_BINARY_INV)
            #         # opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, Kernal)
            #         # dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, Kernal)
            #         if (x+ex) < face_center[0]:
            #             # left_eyes.append((ex,ey,ew,eh))
            #             print(binary.shape)
            #         else:
            #             # right_eyes.append((ex,ey,ew,eh))
            #             print(len(binary))
            # else:
            #     pass 
                    
        cTime = time.time()
        fps = int(1/(cTime-pTime))
        pTime = cTime
        cv2.putText(img, "Fps:" + str(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        cv2.imshow('detecting', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass

cap.release()
cv2.destroyAllWindows()