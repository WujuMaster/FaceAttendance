import os
from functions import cam_capture
from functions import save_to_db
from flask import Flask,render_template,Response
import cv2

db_file = './facedb.pkl'
pictures = './Pictures/'
# user_pass = '1jfiegbquuxla:admin2020@'
# cam_url = 'rtsp://' + user_pass + '192.168.100.11/H264?ch=1&subtype=0'
cam_url = 'http://1jfiegbquuxla:admin2020@192.168.100.11:8090/video'

app = Flask(__name__, template_folder="template")
# camera=cv2.VideoCapture(0)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/attendance')
def attendance():
    return render_template('stream.html')

@app.route('/attendance/video')
def video():
    return Response(cam_capture(cam_url, db_file),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True, port=80)


# menu = {}
# menu['1'] = "Create new faceID" 
# menu['2'] = "Log in with Camera"
# menu['3'] = "Exit"
# while True: 
#     options = sorted(menu)
#     for entry in options: 
#         print(entry, menu[entry])

#     selection = input("Option: ") 
#     if selection =='1': 
#         pic = input('Select picture: ')
#         pic = os.path.join(pictures, pic + '.jpg')
#         res = save_to_db(pic, db_file)
#         if res:
#             print('Saved successfully!')
#         else:
#             print('Cancelled!')
#     elif selection == '2':
#         cam_capture(db_file)
#     elif selection == '3': 
#         break
#     else:
#         print("Unknown Option Selected!")
#     print('======')