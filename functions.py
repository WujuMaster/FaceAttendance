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

# def cam_capture(savefile):
#     vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     pTime = 0
#     while True:
#         _ , frame = vid.read()
#         rate = 3
#         small_frame = cv2.resize(frame, (0, 0), fx=round(1/rate, 2), fy=round(1/rate, 2))
#         # Normalize image to fix brightness
#         norm_img = np.zeros((small_frame.shape[0], small_frame.shape[1]))
#         small_frame = cv2.normalize(small_frame, norm_img, 0, 255, cv2.NORM_MINMAX)
#         rgb_small_frame = small_frame[:, :, ::-1]     
#         try:
#             name, locations = verify_face(rgb_small_frame, savefile)
#         except:
#             name = ''
#             locations = ()          
#         if len(name)<1:
#             name = 'unknown'
#         #start_point is top left, end_point is below right
#         if len(locations) > 0:
#             cv2.rectangle(frame, (locations[3]*rate, locations[0]*rate), (locations[1]*rate, locations[2]*rate), (255, 0, 0), 2)
#             cv2.putText(frame, name, (locations[3]*rate, locations[0]*rate), font, fontScale, fontColor, lineType)
#         cTime = time.time()
#         fps = int(1/(cTime-pTime))
#         pTime = cTime
#         cv2.putText(frame, "Fps:" + str(fps), (10, 20), font, fontScale*0.7, (0,0,0), lineType)
#         cv2.imshow('detecting', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     vid.release()
#     cv2.destroyAllWindows()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def cam_capture(savefile):
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pTime = 0
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
                locations = (x, y, w, h)
            # name, locations = verify_face(rgb_small_frame, savefile)
            name = verify_face(crop_img, savefile)
        except:
            name = ''
            locations = ()
            
        if len(name)<1:
            name = 'unknown'

        #start_point is top left, end_point is below right
        if len(locations) > 0:
            cv2.rectangle(frame, (x*rate, y*rate), ((x+w)*rate, (y+h)*rate), (255, 0, 0), 2)
            print(w, h)
            cv2.putText(frame, name, (x*rate, y*rate), font, fontScale, fontColor, lineType)

        cTime = time.time()
        fps = int(1/(cTime-pTime))
        pTime = cTime
        cv2.putText(frame, "Fps:" + str(fps), (10, 20), font, fontScale*0.7, (0,0,0), lineType)

        cv2.imshow('detecting', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()