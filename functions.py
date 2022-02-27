import cv2
import face_recognition
import pickle
import numpy as np
import time
from scipy.spatial import distance as dist
from datetime import datetime

font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def save_face(picpath, savefile, name='unknown'):
    print('Saving...')
    input = face_recognition.load_image_file(picpath)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    encoded = face_recognition.face_encodings(input)[0]
    
    data = {name:encoded}

    with open(savefile, 'ab') as fp:
        pickle.dump(data, fp)   

def save_to_db(pic, db_file):
    answer = input('Do you want to save new data? (Y/n) ').lower()
    if answer==('y') or answer==('yes'):
        save_face(pic, db_file, name=input('Enter name to save: '))
        return True
    else:
        return False

def verify_face(picpath, savefile):
    if isinstance(picpath, np.ndarray):
        input = cv2.cvtColor(picpath, cv2.COLOR_BGR2RGB)
    else:
        input = face_recognition.load_image_file(picpath)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

    encoded = face_recognition.face_encodings(input)[0]

    names = []
    features = []
    with open(savefile, 'rb') as fr:
        try:
            while True:
                dict = pickle.load(fr)
                for key, value in dict.items():
                    names.append(key)
                    features.append(value)
        except EOFError:
            pass

    id = ''
    min = 1
    for i in range(len(names)):
        result = face_recognition.compare_faces([features[i]], encoded, tolerance=0.4)
        distance = face_recognition.face_distance([features[i]], encoded)
        if result[0]==True:
            if min > distance and distance<=0.5:
                min = distance
                id = names[i] 
    return id

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            if name != 'unknown':
                now = datetime.now()
                dtString = now.strftime('%d/%m/%Y %H:%M:%S')
                f.writelines(f'\n{name}, {dtString}')

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2

def cam_capture(url, savefile):
    vid = cv2.VideoCapture(url, cv2.CAP_DSHOW)
    pTime = 0
    COUNTER = 0
    TOTAL = 0
    while True:
        success , frame = vid.read()
        if not success:
            break
        else:
            rate = 3
            small_frame = cv2.resize(frame, (0, 0), fx=round(1/rate, 2), fy=round(1/rate, 2))
            # Normalize image to fix brightness
            norm_img = np.zeros((small_frame.shape[0], small_frame.shape[1]))
            small_frame = cv2.normalize(small_frame, norm_img, 0, 255, cv2.NORM_MINMAX)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            # rgb_small_frame = small_frame[:, :, ::-1]
            gray = cv2.cvtColor(rgb_small_frame, cv2.COLOR_RGB2GRAY)
            locations = ()
            name = ''
            try:
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    crop_img = rgb_small_frame[y:y+h, x:x+w].copy()
                    face = frame[y*rate-10:(y+h)*rate+10, x*rate-10:(x+w)*rate+10].copy()
                    face_landmarks_list = face_recognition.face_landmarks(face)[0]
                    left_eye = face_landmarks_list['left_eye']
                    right_eye = face_landmarks_list['right_eye']
                    leftEAR = eye_aspect_ratio(left_eye)
                    rightEAR = eye_aspect_ratio(right_eye)
                    ear = (leftEAR + rightEAR) / 2.0
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                        # print("eye_closed")
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1
                        COUNTER = 0
                        # print("reset")
                    # cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 40),
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    locations = (x, y, w, h)

                name = verify_face(crop_img, savefile)

                if len(name)<1:
                    name = 'unknown'
                
                #start_point is top left, end_point is below right
                if len(locations) > 0:
                    markAttendance(name)
                    X, Y, W, H = x*rate, y*rate, w*rate, h*rate
                    if TOTAL > 0:
                        color = (0, 255, 0)
                        cv2.putText(frame, name, (X, Y-5), font, 0.75, color, 2)
                        cv2.rectangle(frame, (X,Y), (X+W,Y+H), color, 2)
                    else:
                        cv2.putText(frame, "Blink to detect face...", (90, 20), font, 0.5, (255,100,0), 2)  
                else:
                    TOTAL = 0
                    pass

            except:
                pass

            cTime = time.time()
            fps = int(1/(cTime-pTime))
            pTime = cTime
            cv2.putText(frame, "Fps:" + str(fps), (10, 20), font, 0.7, (0,0,0), 2)
            
            # cv2.imshow('detecting', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # vid.release()
    # cv2.destroyAllWindows()