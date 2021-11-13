import cv2
import face_recognition
import pandas as pd
import os
import numpy as np
import time

def check_savefile(savefile):
    if not os.path.isfile(savefile):
        df = pd.DataFrame(list())
        df.to_csv(savefile)

def save_face(picpath, savefile, name='unknown'):
    input = face_recognition.load_image_file(picpath)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

    faceLoc = face_recognition.face_locations(input)[0]
    encoded = face_recognition.face_encodings(input)[0]

    start_point = (faceLoc[3], faceLoc[0])
    end_point = (faceLoc[1], faceLoc[2])

    data = pd.DataFrame([[name, start_point, end_point, encoded]])
    
    check_savefile(savefile)
    
    df = pd.read_csv(savefile)
    if len(df.axes[0]) < 1:
        data.to_csv(savefile, index=False, header=['Name', 'Start_point', 'End_point', 'Features'])
    else:
        data.to_csv(savefile, mode='a', index=False, header=None)        
    print('Saved!')

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
    locations = face_recognition.face_locations(input)[0]
    # print(type(locations))

    df = pd.read_csv(savefile)
    features = df.iloc[:, 3].values
    names = df.iloc[:, 0].values

    id = ''
    min = 5
    for i in range(len(df.axes[0])):
        features[i] = features[i][1:-1]
        face = np.fromstring(features[i], sep=' ')
        result = face_recognition.compare_faces([face], encoded)
        distance = face_recognition.face_distance([face], encoded)
        if result[0]==True:
            if min > distance and distance<=0.5:
                min = distance
                id = names[i] 
    return id, locations

def cam_capture(savefile):
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pTime = 0
    while True:
        _ , frame = vid.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33)
        # print(frame.shape)
        rgb_small_frame = small_frame[:, :, ::-1]
        try:
            name, locations = verify_face(rgb_small_frame, savefile)
            # name, locations = verify_face(frame, savefile)
        except:
            name = ''
            locations = ()
            
        if len(name)<1:
            name = 'unknown'

        #start_point is top left, end_point is below right
        if len(locations) > 0:
            # cv2.rectangle(frame, (locations[3], locations[0]), (locations[1], locations[2]), (255, 0, 0), 2)
            cv2.rectangle(frame, (locations[3]*3, locations[0]*3), (locations[1]*3, locations[2]*3), (255, 0, 0), 2)
            
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (locations[3]*3, locations[0]*3)
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2
            cv2.putText(frame, name, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        cTime = time.time()
        fps = int(1/(cTime-pTime))
        pTime = cTime
        cv2.putText(frame, "Fps:" + str(fps), (10, 20), font, fontScale*0.7, (0,0,0), lineType)

        cv2.imshow('detecting', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
