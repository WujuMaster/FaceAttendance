import os
from functions import cam_capture
# from functions import verify_face
from functions import save_to_db

db_file = './FaceDatabase.csv'
pictures = './Pictures/'

menu = {}
menu['1'] = "Create new faceID" 
menu['2'] = "Log in with Camera"
menu['3'] = "Exit"
while True: 
    options = sorted(menu)
    for entry in options: 
        print(entry, menu[entry])

    selection = input("Option: ") 
    if selection =='1': 
        pic = input('Select picture: ')
        pic = os.path.join(pictures, pic + '.jpg')
        res = save_to_db(pic, db_file)
        if res:
            print('Saved successfully!')
        else:
            print('Cancelled!')
    elif selection == '2':
        cam_capture(db_file)
    elif selection == '3': 
        break
    else:
        print("Unknown Option Selected!")
    print('======')
