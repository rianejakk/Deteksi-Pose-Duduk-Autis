from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
from keras.models import load_model
import mediapipe as mp
import numpy as np
import pandas as pd
import math

cam_on = False
cap = None

path_saved_model = "model_au_improved.h5"
threshold = 0.70

torso_size_multiplier = 2.5
n_landmarks = 33
n_dimensions = 3
landmark_names = [
    'nose',
    'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky_1', 'right_pinky_1',
    'left_index_1', 'right_index_1',
    'left_thumb_2', 'right_thumb_2',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]
class_names = [
    'Correct', 'Incorrect'
]

##############

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

col_names = []
for i in range(n_landmarks):
    name = mp_pose.PoseLandmark(i).name
    name_x = name + '_X'
    name_y = name + '_Y'
    name_z = name + '_Z'
    name_v = name + '_V'
    col_names.append(name_x)
    col_names.append(name_y)
    col_names.append(name_z)
    col_names.append(name_v)

# Load saved model
model = load_model(path_saved_model, compile=True)

# GUI
app = Tk()
app.title("Dibisa")
app.geometry("900x700")
bg_img = PhotoImage(file="bg.png")
bg_label = Label(app, image=bg_img)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
app.bind('<Escape>', lambda e: app.quit())

label_widget = Label(app)
label_widget.place(anchor="s", x=450, y=500)

def start_vid():
    global cam_on, cap
    cam_on = True
    cap = cv2.VideoCapture(0)

    def update_frame():
        global cam_on, cap
        success, img = cap.read()
        if not success:
            print('No Video feed')
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        if result.pose_landmarks:
            lm_list = []
            for landmarks in result.pose_landmarks.landmark:
                max_distance = 0
                lm_list.append(landmarks)
            center_x = (lm_list[landmark_names.index('right_hip')].x +
                        lm_list[landmark_names.index('left_hip')].x)*0.5
            center_y = (lm_list[landmark_names.index('right_hip')].y +
                        lm_list[landmark_names.index('left_hip')].y)*0.5

            shoulders_x = (lm_list[landmark_names.index('right_shoulder')].x +
                            lm_list[landmark_names.index('left_shoulder')].x)*0.5
            shoulders_y = (lm_list[landmark_names.index('right_shoulder')].y +
                            lm_list[landmark_names.index('left_shoulder')].y)*0.5

            for lm in lm_list:
                distance = math.sqrt((lm.x - center_x) **
                                        2 + (lm.y - center_y)**2)
                if(distance > max_distance):
                    max_distance = distance
            torso_size = math.sqrt((shoulders_x - center_x) **
                                    2 + (shoulders_y - center_y)**2)
            max_distance = max(torso_size*torso_size_multiplier, max_distance)

            pre_lm = list(np.array([[(landmark.x-center_x)/max_distance, (landmark.y-center_y)/max_distance,
                                        landmark.z/max_distance, landmark.visibility] for landmark in lm_list]).flatten())
            data = pd.DataFrame([pre_lm], columns=col_names)
            predict = model.predict(data)[0]

            # Lebih besar dari threshold, tampilkan hasil prediksi
            if max(predict) > threshold :
                pose_class = class_names[predict.argmax()]
                print('predictions: ', predict)
                print('predicted Pose Class: ', pose_class)
            else:
                # Jika tidak, tampilkan 'Unknown Pose'
                pose_class = 'Unknown Pose'
                print('[INFO] Predictions is below given Confidence!!')

            # Show Result
            img = cv2.putText(
                img, f'{pose_class}',
                (40, 50), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2
            )

        # Apparently it is necessary to convert the image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        captured_image = Image.fromarray(img)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        label_widget.photo_image = photo_image
        label_widget.configure(image=photo_image)

        if cam_on:
            label_widget.after(1, update_frame)

    update_frame()

    if cv2.waitKey(1) == 27:
        return
cv2.destroyAllWindows()

def stop_vid():
    global cam_on
    cam_on = False
    
    if cap:
        cap.release()
        label_widget.config(image="")

opn_cam = Button(app, text="Open Camera", command=start_vid)
opn_cam.place(anchor="n", x=400, y=550)

close_cam = Button(app, text="Close Camera", command=stop_vid)
close_cam.place(anchor="n", x=500, y=550)

app.mainloop()