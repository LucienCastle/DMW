import tkinter as tk
from tkinter import filedialog, Text, ttk
import pandas as pd
import numpy as np
import model as md
import tkinter.font as tkFont

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

fontStyle = tkFont.Font(family="Lucida Grande", size=20)

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

root2.geometry("700x700")

df_train,df_test = md.files_input(dataset)
md.visualization(df_train,df_test)
df_all = md.preprocessing(df_train,df_test)

df_all,len_train = md.sentiment_analysis(df_all)

sub_preds, df_test = md.modelML(df_all,len_train)
df_test = md.dict_sentiment_model(df_test,sub_preds)

root2.title("Predictions")
fontStyle = tkFont.Font(family="Lucida Grande", size=20)
fontsize = fontStyle['size']

top_frame = tk.Frame(root2, bg='#263D42', height = 200)
top_frame.pack(fill=tk.X,side='top')

some_frame = tk.Frame(root2, bg='#263D42')
some_frame.pack(fill = tk.X,side='top')

left_frame = tk.Frame(some_frame,bg='#263D42',width = 200)
left_frame.pack(side=tk.LEFT)
condition_frame = tk.Frame(some_frame)
condition_frame.pack(side=tk.LEFT)
Condition = ttk.Label(condition_frame, text="Condition", background="#263D42",font = fontStyle, foreground = '#FFFFFF')
Condition.grid(row=10, column=0, sticky=tk.W)
entry_frame = tk.Frame(some_frame)
entry_frame.pack(side=tk.LEFT) 
Condition_var = tk.StringVar()
Condition_entrybox = ttk.Entry(entry_frame, width=16, textvariable=Condition_var, font = fontStyle)
Condition_entrybox.grid(row=10, column=10, sticky=tk.E)
rigt_frame = tk.Frame(some_frame,bg='#263D42',width = 200)
rigt_frame.pack(side=tk.RIGHT)

mid_frame = tk.Frame(root2,bg='#263D42',height=100)
mid_frame.pack(fill=tk.X, side = 'top')

def predict():
	CONDITION = Condition_var.get()
	best_drug = md.recommend(CONDITION,df_test)
	
	ans = tk.Tk()
	ans.configure(background="#263D42")
	ans.geometry("700x350")
	fontStyle = tkFont.Font(family="Lucida Grande", size=20)
	fontsize = fontStyle['size']
	ans_frame = tk.Frame(ans, bg = '#263D42')
	ans_frame.pack(fill = tk.BOTH)

	if best_drug and not best_drug.empty():
		eff_drug = best_drug
	else:
		eff_drug = 'No such ailment exist/No drug for your ailment. Reenter your ailment'

	classification = ttk.Label(ans_frame, text=eff_drug, font = fontStyle,background="#263D42", foreground = '#FFFFFF')
	classification.grid(row=5, column=0, sticky=tk.W)	
	ans.mainloop()

some_frame2 = tk.Frame(root2,height=50, width=50)
some_frame2.pack()
submit_button = ttk.Button(some_frame2, text="Submit", command=predict)

submit_button.grid(row=16, column=0, sticky=tk.W + tk.E)

root2.mainloop()
