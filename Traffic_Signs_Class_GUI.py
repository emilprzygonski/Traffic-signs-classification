import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2 as cv
import numpy as np
from keras.models import load_model

model = load_model('traffic.h5')

# dictionary class -> sign name
classNames = {0: 'Speed limit (20km/h)',
              1: 'Speed limit (30km/h)',
              2: 'Speed limit (50km/h)',
              3: 'Speed limit (60km/h)',
              4: 'Speed limit (70km/h)',
              5: 'Speed limit (80km/h)',
              6: 'End of speed limit (80km/h)',
              7: 'Speed limit (100km/h)',
              8: 'Speed limit (120km/h)',
              9: 'No passing',
              10: 'No passing veh over 3.5 tons',
              11: 'Right-of-way at intersection',
              12: 'Priority road',
              13: 'Yield',
              14: 'Stop',
              15: 'No vehicles',
              16: 'Vehicles > 3.5 tons prohibited',
              17: 'No entry',
              18: 'General caution',
              19: 'Dangerous curve left',
              20: 'Dangerous curve right',
              21: 'Double curve',
              22: 'Bumpy road',
              23: 'Slippery road',
              24: 'Road narrows on the right',
              25: 'Road work',
              26: 'Traffic signals',
              27: 'Pedestrians',
              28: 'Children crossing',
              29: 'Bicycles crossing',
              30: 'Beware of ice/snow',
              31: 'Wild animals crossing',
              32: 'End speed + passing limits',
              33: 'Turn right ahead',
              34: 'Turn left ahead',
              35: 'Ahead only',
              36: 'Go straight or right',
              37: 'Go straight or left',
              38: 'Keep right',
              39: 'Keep left',
              40: 'Roundabout mandatory',
              41: 'End of no passing',
              42: 'End no passing veh > 3.5 tons'}


def classify(file_path):
    # global label_packed
    rgb_img = cv.cvtColor(cv.imread(file_path), cv.COLOR_RGB2BGR)
    resized_img = cv.resize(rgb_img, (35, 35))
    array_img = np.expand_dims(resized_img, axis=0)
    batch_img = np.array(array_img / 255.0)
    predictions = model.predict(batch_img)[0]
    predicted_value = np.argmax(predictions)
    predicted_sign = classNames[predicted_value]
    label.configure(foreground='#A41010', text=f'Predicted sign: {predicted_sign}')
    label.pack(side=TOP)


def show_classify_button(file_path):
    classify_button = Button(top, text='Classify image', command=lambda: classify(file_path), padx=10, pady=5)
    classify_button.configure(background='#A41010', foreground='white', font=('arial', 10, 'bold'))
    classify_button.place(relx=0.6, rely=0.8)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded_img = Image.open(file_path)
        uploaded_img.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        image = ImageTk.PhotoImage(uploaded_img)

        sign_image.configure(image=image)
        sign_image.image = image
        sign_image.pack(side=TOP, pady=30)

        label.configure(text='')
        show_classify_button(file_path)
        upload_button.place(relx=0.2, rely=0.8)
    except:
        pass


top = tk.Tk()
top.geometry('700x400')
top.title('Traffic Signs Classification')
top.configure(background='white')

label = Label(top, background='white', font=('arial', 15, 'bold'))
sign_image = Label(top, background='white')

upload_button = Button(top, text='Load image...', command=upload_image, padx=10, pady=5)
upload_button.configure(background='#A41010', foreground='white', font=('Calibri Light', 10, 'bold'))
upload_button.pack(side=BOTTOM, pady=50)

sign_image.place(relx=0.5, rely=0.5)
label.place(relx=0.1, rely=0.3)

heading = Label(top, text='Guess traffic sign', pady=20, font=('Calibri Light', 40, 'bold'))
heading.configure(background='white', foreground='#A41010')
heading.pack()
top.mainloop()
