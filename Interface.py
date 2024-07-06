import sys
import tkinter as tk
#NUMPY AND PANDAS FOR DATA STORAGE
import pandas as pd
import numpy as np
#REGULAR EXPRESSION (re) USED FOR REMOVAL OF PARENTHESIS AND EXCLAMATION MARKS IN DATA
import re
#Natural Language Toolkit (NLTK) used for stopword removal and lemmatization
import nltk
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemm=WordNetLemmatizer()
import matplotlib.pyplot as plt
from sklearn import metrics
#OS used to remove the warning in the terminal
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
#Tensorflow used for model creation and traning
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

#JSON used to store tokenizer values
import json

    

def load_asset(path):
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    assets = os.path.join(base, "assets")
    return os.path.join(assets, path)

window = tk.Tk()
window.geometry("1377x768")
window.configure(bg="#ffffff")
window.title("Untitled")

canvas = tk.Canvas(
    window,
    bg = "#ffffff",
    width = 1377,
    height = 768,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x=0, y=0)

class TkForge_Text(tk.Text):
    def __init__(self, master=None, placeholder="Enter text", placeholder_fg='grey', **kwargs):
        super().__init__(master, **kwargs)
        
        self.p, self.p_fg, self.fg = placeholder, placeholder_fg, self.cget("fg")
        self.putp()
        self.bind("<FocusIn>", self.toggle)
        self.bind("<FocusOut>", self.toggle)

    def putp(self):
        self.delete('1.0', tk.END)
        self.insert('1.0', self.p)
        self.config(fg=self.p_fg)
        self.p_a = True

    def toggle(self, event):
        if self.p_a:
            self.delete('1.0', tk.END)
            self.config(fg=self.fg)
            self.p_a = False
        elif self.get('1.0', tk.END).replace(' ', '').replace('\n', '') == '': self.putp()

    def get(self, i1='1.0', i2=tk.END): return '' if self.p_a else super().get(i1, i2)

    def is_placeholder(self, b):
        self.p_a = b
        self.config(fg=self.p_fg if b == True else self.fg)

    def get_placeholder(self): return self.p

canvas.create_rectangle(0, 0, 1366, 768, fill='#d9d9d9', outline="")
canvas.create_rectangle(0, 0, 272, 768, fill='#212524', outline="")
canvas.create_rectangle(273, 0, 1377, 768, fill='#404040', outline="")

button_1_image = tk.PhotoImage(file=load_asset("1.png"))
button_1 = tk.Button(
    image=button_1_image,
    relief="flat",
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 has been pressed!")
)
button_1.place(x=0, y=0, width=272, height=51)


canvas.create_line(0, 51, 272, 51, fill="#000000", width=1.0)


button_2_image = tk.PhotoImage(file=load_asset("2.png"))
def manual():
    cords=[369,457,819, 492]
    canvas.coords(green,cords)
    textarea_1.place(x=369, y=215, width=900, height=200)
    canvas.delete(txt)
    button_6.configure(image=button_6_image)
    button_6.configure(command=analyze)
button_2 = tk.Button(
    image=button_2_image,
    relief="flat",
    borderwidth=0,
    highlightthickness=0,
    command=manual
)
button_2.place(x=0, y=51, width=272, height=49)

button_3_image = tk.PhotoImage(file=load_asset("3.png"))
df=pd.read_csv(r"Data\test.csv",usecols=["review"],dtype={"review":str})
reviews=np.array(df.review)
x_test=np.array(reviews[300001:399999])
j=1
button_6_auto_image = tk.PhotoImage(file=load_asset("9.png"))
def auto_analyze():
    global model, tokenizer, j
    text=x_test[j]
    test_sequence= tokenizer.texts_to_sequences([text])
    test_pad=pad_sequences(test_sequence, maxlen=30)
    prediction = model.predict(test_pad)
    j=j+1
    len=1269-369
    glen=len*prediction[0]
    dim=369+glen
    cords=canvas.coords(green)
    cords[2]=int(dim)
    canvas.coords(green,cords)
    button_6.configure(image=button_6_auto_image)
    button_6.configure(command=next)
def auto():
    global j,txt
    cords=[369,457,819, 492]
    canvas.coords(green,cords)
    textarea_1.place_forget()
    txt = canvas.create_text(
    369,
    215,
    text=x_test[j],
    width=900,
    anchor="nw",
    fill="#ffffff",
    font=("Inter",13)
    )
    button_6.configure(command=auto_analyze)
def next():
    global j,txt
    cords=[369,457,819, 492]
    canvas.coords(green,cords)
    canvas.itemconfig(txt,text=x_test[j])
    button_6.configure(image=button_6_image)
    button_6.configure(command=auto_analyze)
button_3 = tk.Button(
    image=button_3_image,
    relief="flat",
    borderwidth=0,
    highlightthickness=0,
    command=auto
)
button_3.place(x=0, y=100, width=272, height=49)

canvas.create_line(272, 0, 272, 768, fill="#000000", width=1.0)

textarea_1 = TkForge_Text(
    bd=0,
    bg="#404040",
    fg="#ffffff",
    placeholder="Enter Your Text Here",
    insertbackground="#ffffff",
    highlightthickness=0,
    font=(15)
)
textarea_1.place(x=369, y=215, width=900, height=200)


canvas.create_rectangle(369, 457, 1269, 492, fill='#ff0000', outline="")
green=canvas.create_rectangle(369,457,819, 492, fill='#14ff00', outline="")
canvas.create_line(819,447,819,502,fill="#ffffff")
#canvas.create_rectangle(819, 457, 1270, 492, fill='#ff0000', outline="")

def analyze():
    global model, tokenizer
    text=textarea_1.get(1.0,"end-1c")
    test_sequence= tokenizer.texts_to_sequences([text])
    test_pad=pad_sequences(test_sequence, maxlen=30)
    prediction = model.predict(test_pad)
    len=1269-369
    glen=len*prediction[0]
    dim=369+glen
    cords=canvas.coords(green)
    cords[2]=int(dim)
    canvas.coords(green,cords)
button_6_image = tk.PhotoImage(file=load_asset("6.png"))
button_6 = tk.Button(
    image=button_6_image,
    relief="flat",
    borderwidth=0,
    highlightthickness=0,
    command=analyze
)
button_6.place(x=703, y=529, width=228, height=52)


model=Sequential()
tokenizer=Tokenizer()
"""def warn():
    canvas.tag_raise(txt)
    canvas.tag_raise(txt)
    canvas.itemconfig(txt,text="Model Already Loaded!")
    time.sleep(10)
    canvas.itemconfig(txt,text="..")"""

def load_mod():
    global model,tokenizer
    model=load_model("my_model1.keras")
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    button_7.config(image=button_7_image2)
    button_7.config(command=None)

button_7_image = tk.PhotoImage(file=load_asset("7.png"))
button_7_image2=tk.PhotoImage(file=load_asset("8.png"))
button_7 = tk.Button(
    image=button_7_image,
    relief="flat",
    borderwidth=0,
    highlightthickness=0,
    command=load_mod,
    background="#414141"
)
#txt=canvas.create_text(1165,78,anchor="nw",text=".",fill="#ffffff")
button_7.place(x=1118, y=20, width=210, height=50)

canvas.create_rectangle(358, 212, 1269, 415, fill='#404040', outline="#d9d9d9", width="1.0")

window.resizable(False, False)
window.mainloop()
