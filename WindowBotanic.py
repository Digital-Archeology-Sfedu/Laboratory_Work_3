from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
import time
import tkinter as tk
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
window_width = 360
window_height = 300

root = tk.Tk()
root.title('Botanic Predictor')
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
#root.geometry('360x240+0+0')
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)

root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

file = None
line = pd.DataFrame()
def open_file():
    file_path = askopenfile(mode='r', filetypes=[('TXT files', '*txt'), ('CSV Files', '*csv')])
    if file_path is not None:
        myTXT = tk.Label(
            root,
            text = file_path.name
            )
        myTXT.config(anchor=CENTER)
        myTXT.place(x = 50, y = 0)
        preprocessing(file_path)

def kmeans_predictor():
    kmean = joblib.load('models/kmeans_model.sav')
    predict = kmean.predict(line)
    result_output(predict, 'KMeans')

def rf_predictor():
    rf = joblib.load('models/random_forest_model.sav')
    predict = rf.predict(line)
    result_output(predict, 'Random Forest')
def gb_predictor():
    gb = joblib.load('models/gradient_boosting_model.sav')
    predict = gb.predict(line)
    result_output(predict, 'Gradient Boost')
def knn_predictor():
    knn = joblib.load('models/k_neighbors_model.sav')
    predict = knn.predict(line)
    result_output(predict, 'KNeighbors')

def result_output(result, text_predictor):
    type_leaf = ''
    print(result)
    if result[0] == 1:
        type_leaf = 'Клен полевой'
    elif result[0] == 10:
        type_leaf = 'Клен ясенелистный'
    elif result[0] == 11:
        type_leaf = 'Клен серебристый'
    output = tk.Label(
        root,
        text= text_predictor + ', predict: ' + type_leaf

    )
    output.place(x=50, y=220)



def preprocessing(file):
    global line
    secv = pd.read_csv(file.name, sep=",", header=None)


    pca = joblib.load('models/pca_model.sav')
    line = pca.transform(secv)


dlbtn = Button(
    root,
    text ='Choose File ',
    command = lambda:open_file()
    )
dlbtn.place(x = 110, y = 20, height=30, width=150)
name_file = tk.Label(
        root,
        text = 'File path:'
        )
name_file.place(x = 0, y = 0)

result = tk.Label(
        root,
        text = 'Result: '
        )
result.place(x = 0, y = 220)

rfbutton = tk.Button(text="Random Forest Predict", padx="3", pady="3", command = lambda:rf_predictor())
rfbutton.place(x = 110, y = 60, height=30, width=150)
kmbutton = tk.Button(text="Kmeans Predict", padx="3", pady="3", command = lambda:kmeans_predictor())
kmbutton.place(x = 110, y = 100, height=30, width=150)
gbbutton = tk.Button(text="Gradient Boosting Predict", padx="3", pady="3", command = lambda:gb_predictor())
gbbutton.place(x = 110, y = 140, height=30, width=150)
knnbutton = tk.Button(text="KNeighbors Predict", padx="3", pady="3", command = lambda:knn_predictor())
knnbutton.place(x = 110, y = 180, height=30, width=150)
root.mainloop()