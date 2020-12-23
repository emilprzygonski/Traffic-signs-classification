# load modules
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2 as cv
import numpy as np
from keras.models import load_model

# load pretrained model
model = load_model('traffic.h5')

# dictionary class -> sign name
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Vehicles > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

# needed functions 
def classify(file_path):
    global label_packed
    image = cv.cvtColor(cv.imread(file_path), cv.COLOR_RGB2BGR)
    image = cv.resize(image, (35,35))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict(image)[0]
    cls = np.argmax(pred)
    sign = classes[cls]
    label.configure(foreground='#A41010', text=f'Predicted sign: {sign}')
    label.pack(side=TOP)

def show_classify_button(file_path):
    classify_b=Button(top,text='Classify image',command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#A41010', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.6,rely=0.8)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        sign_image.pack(side=TOP, pady=30)

        label.configure(text='')
        show_classify_button(file_path)
        upload_button.place(relx=0.2, rely=0.8)
    except:
        pass


top=tk.Tk()
top.geometry('700x400')
top.title('Traffic Signs Classification')
top.configure(background='white')

label=Label(top,background='white', font=('arial',15,'bold'))
sign_image = Label(top, background='white')

upload_button=Button(top,text='Load image...',command=upload_image,padx=10,pady=5)
upload_button.configure(background='#A41010', foreground='white',font=('Calibri Light',10,'bold'))
upload_button.pack(side=BOTTOM,pady=50)

sign_image.place(relx=0.5,rely=0.5)
label.place(relx=0.1,rely=0.3)

heading = Label(top, text='Guess traffic sign',pady=20, font=('Calibri Light',40,'bold'))
heading.configure(background='white',foreground='#A41010')
heading.pack()
top.mainloop()
