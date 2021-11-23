import cv2
import face_recognition
import pickle
import numpy as np
import time

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

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
    # locations = face_recognition.face_locations(input)[0]

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
        result = face_recognition.compare_faces([features[i]], encoded)
        distance = face_recognition.face_distance([features[i]], encoded)
        if result[0]==True:
            if min > distance and distance<=0.5:
                min = distance
                id = names[i] 
    return id

def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

Kernal = np.ones((3, 3), np.uint8)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def cam_capture(savefile):
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pTime = 0
    left_eyes = []
    right_eyes = []
    while True:
        _ , frame = vid.read()
        rate = 3
        small_frame = cv2.resize(frame, (0, 0), fx=round(1/rate, 2), fy=round(1/rate, 2))

        # Normalize image to fix brightness
        norm_img = np.zeros((small_frame.shape[0], small_frame.shape[1]))
        small_frame = cv2.normalize(small_frame, norm_img, 0, 255, cv2.NORM_MINMAX)

        rgb_small_frame = small_frame[:, :, ::-1]

        gray = cv2.cvtColor(rgb_small_frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        try:
            for (x, y, w, h) in faces:
                crop_img = rgb_small_frame[y:y+h, x:x+w].copy()

                if len(left_eyes)>10:
                    left_eyes.clear()
                    right_eyes.clear()
                # roi_color = rgb_small_frame[y:y+h, x:x+w]
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (100, 100))
                face_landmarks_list = face_recognition.face_landmarks(roi_gray)[0]

                start = face_landmarks_list['left_eye'][0]
                end = face_landmarks_list['left_eye'][3]
                width = int((end[0] - start[0])/2)
                center = (int((start[0] + end[0])/2), int((start[1] + end[1])/2))
                # cv2.rectangle(frame, ((x + center[0]-width)*rate, (y + center[1]-width)*rate), ((x + center[0]+width)*rate, (y + center[1]+width)*rate), (255, 0, 0), 2)
                roi = roi_gray[center[1]-width : center[1]+width, center[0]-width : center[0]+width]
                _, binary = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY)
                opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, Kernal)
                dilate_left = cv2.morphologyEx(opening, cv2.MORPH_DILATE, Kernal)
                left_eyes.append(dilate_left.tobytes())

                start = face_landmarks_list['right_eye'][0]
                end = face_landmarks_list['right_eye'][3]
                width = int((end[0] - start[0])/2)
                center = (int((start[0] + end[0])/2), int((start[1] + end[1])/2))
                # cv2.rectangle(frame, ((x + center[0]-width)*rate, (y + center[1]-width)*rate), ((x + center[0]+width)*rate, (y + center[1]+width)*rate), (255, 0, 0), 2)
                roi = roi_gray[center[1]-width : center[1]+width, center[0]-width : center[0]+width]
                _, binary = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY)
                opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, Kernal)
                dilate_right = cv2.morphologyEx(opening, cv2.MORPH_DILATE, Kernal)
                right_eyes.append(dilate_right.tobytes())

                locations = (x, y, w, h)
            name = verify_face(crop_img, savefile)
            left_score = 0
            right_score = 0
            if len(left_eyes)==10:
                for i in range(len(left_eyes)):
                    for j in range(i, len(left_eyes)):
                        left_score = left_score + hamming_distance(left_eyes[i], left_eyes[j])
                        right_score = right_score + hamming_distance(right_eyes[i], right_eyes[j])
            
            liveness_score = (left_score + right_score)/20
            liveness = ''
            if liveness_score>0:
                print(liveness_score)
            if liveness_score >= 50:
                liveness = 'Real'
            elif liveness_score > 0 and liveness_score < 50:
                liveness = 'Fake'

        except:
            name = ''
            locations = ()
            
        if len(name)<1:
            name = 'unknown'

        #start_point is top left, end_point is below right
        if len(locations) > 0:
            cv2.rectangle(frame, (x*rate, y*rate), ((x+w)*rate, (y+h)*rate), (255, 0, 0), 2)
            cv2.putText(frame, name, (x*rate, y*rate), font, fontScale, fontColor, lineType)
            cv2.putText(frame, liveness, ((x+w-10)*rate, y*rate), font, fontScale, fontColor, lineType)

        cTime = time.time()
        fps = int(1/(cTime-pTime))
        pTime = cTime
        cv2.putText(frame, "Fps:" + str(fps), (10, 20), font, fontScale*0.7, (0,0,0), lineType)

        cv2.imshow('detecting', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()