import tkinter as tk
from tkinter import filedialog, Text, ttk
import pandas as pd
import numpy as np
import model as md

import matplotlib.pyplot as plt
root = tk.Tk()
root.title("Datasets")
dataset = []


def exitFrame():
    root.destroy()

def addCsv():

    fileName = filedialog.askopenfilename(initialdir="~/DMW/Project", title="Select Train csv", filetypes=(("CSV Files", "*.csv"),))
    if(fileName not in dataset):
        dataset.append(fileName)
        if(len(dataset) == 2):
            exitFrame()


canvas = tk.Canvas(root, height=100, width=700, bg="#263D42")
canvas.pack()
button_frame = tk.Frame(root, bg="white")
button_frame.place(relwidth=0.5, relheight=0.5, relx=0.1, rely=0.1)
button_frame.pack(fill=tk.X, side=tk.BOTTOM)


train_button = tk.Button(button_frame, text='Train.csv', command=addCsv)
test_button = tk.Button(button_frame, text='Test.csv', command=addCsv)
if(len(dataset) == 2):
    exitFrame()
button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)

train_button.grid(row=1, column=0, sticky=tk.W + tk.E)
test_button.grid(row=1, column=1, sticky=tk.W + tk.E)

root.mainloop()


root2 = tk.Tk()
root2.configure(background="#263D42")

"""
root2.title("Training")
canvas = tk.Canvas(root2, height=100, width=700, bg="#263D42")

canvas.pack()
"""
df_train,df_test = md.files_input(dataset)
md.visualization(df_train,df_test)
df_all = md.preprocessing(df_train,df_test)

df_all,len_train = md.sentiment_analysis(df_all)

sub_preds, df_test = md.modelML(df_all,len_train)
df_test = md.dict_sentiment_model(df_test,sub_preds)

"""
write remaining code here
"""

"""
canvas.create_text(350, 50, fill="white", font="Times 20 bold",
                   text="Your Model is Ready")
canvas.delete("all")"""

root2.title("Predictions")

# canvas = tk.Canvas(root2, height=700, width=700, bg="#263D42")

# canvas.pack()
tp = ttk.Label(root2, text="                 ", background="#263D42")
tp.grid(row=0, column=2, sticky=tk.W)

Condition = ttk.Label(root2, text="Condition", background="#263D42")
Condition.grid(row=2, column=0, sticky=tk.W)
Condition_var = tk.StringVar()
Condition_entrybox = ttk.Entry(root2, width=16, textvariable=Condition_var)
Condition_entrybox.grid(row=2, column=1)

# converting into dataframe
DB = pd.DataFrame()


def action():
    global DB
    DF = pd.DataFrame(columns=['uniqueID', 'drugName', 'condition', 'rating', 'date', 'usefulCount'])
    UNIQUEID = UniqueID_var.get()
    DF.loc[0, 'uniqueID'] = UNIQUEID
    DRUGNAME = DrugName_var.get()
    DF.loc[0, 'drugName'] = DRUGNAME
    CONDITION = Condition_var.get()
    DF.loc[0, 'condition'] = CONDITION
    RATING = Rating_var.get()
    DF.loc[0, 'rating'] = RATING
    DATE = Date_var.get()
    DF.loc[0, 'date'] = DATE
    USEFULCOUNT = UseFulCount_var.get()
    DF.loc[0, 'usefulCount'] = USEFULCOUNT
    DB = DF
    DB["uniqueID"] = pd.to_numeric(DB["uniqueID"])
    DB["rating"] = pd.to_numeric(DB["rating"])
    DB["usefulCount"] = pd.to_numeric(DB["usefulCount"])


def predict():
	md.recommend(Condition)
    # action()
    """
    predict on DB
    """
    # print(DB)
    ans = tk.Tk()
    ans.configure(background="#263D42")
    classification = ttk.Label(ans, text="classification", background="#263D42")
    classification.grid(row=5, column=0, sticky=tk.W)
    ans.mainloop()


submit_button = ttk.Button(root2, text="Submit", command=predict)

submit_button.grid(row=6, column=0, sticky=tk.W + tk.E)

root2.mainloop()
