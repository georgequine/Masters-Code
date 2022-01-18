print("Booting System")

# import necessary modules
from tkinter import * 
from tkinter import ttk
from PIL import Image, ImageTk
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tkinter import filedialog
import os
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pickle

print("Modules Sucessfully Imported")

#Window Customisation
root = Tk()
root.title("Random Forest Application")
root.geometry("400x400") # set starting size of window
root.maxsize(400, 400) # width x height
root.config(bg="#6FAFE7") # set background color of root window
 
#Top Header
Heading = Label(root, text="Random Forest Application", bg="#2176C1", fg='white', relief=RAISED)
Heading.pack(ipady=5, fill='x')
Heading.config(font=("Font", 20)) # change font and size of label

#Initialise variables 
list = ["Clockwise", "Anti-Clockwise", "Left Swipe", "Right Swipe", "Select"]
fulldata = pd.read_csv('/Users/Owner/Documents/myModel/full_dataset.csv')
print("Fulldata Loaded")
light1status = 0
global model
# load it again
with open('/Users/Owner/Documents/myModel/rf_classifier.pkl', 'rb') as fid:
	model = pickle.load(fid)
print("Model Installed")
print(model)
global features
global x
global y 
global principalComponents
global pca
pca = PCA(n_components=0.99)
features = [str(i) for i in range(1001,13001)]
x = fulldata.loc[:, features].values
y = fulldata.loc[:,['label']].values
principalComponents = pca.fit_transform(x)
print("PCA Complete")
degrees = 0
slider_pos = 0
gesture_number = 0

print("System Ready\n")

#Switch to control functions depending on detection (Not in use yet)
def switch(detection):

	global gesture	
	global degrees
	global slider_pos

	if detection == "Clockwise":
		gesture = "Clockwise"
		degrees += 25
		rotate_image(degrees)


	elif detection == "Anti-Clockwise":
		gesture = "Anti-Clockwise"
		degrees -= 25
		rotate_image(degrees)


	elif detection == "Left Swipe":
		gesture = "Left Swipe"
		slider_pos = -75
		slider(slider_pos)

	elif detection == "Right Swipe":
		gesture = "Right Swipe"
		slider_pos = 75
		slider(slider_pos)

	elif detection == "Double Hand":
		gesture = "Select"
		light1()
	

#Timer at top of window. Once timer reaches 0, user can perform gesture (Not in use yet)
def go():
	# call countdown first time    
	countdown(5)
	# root.after(0, countdown, 5)

def countdown(count):
	# change text in label        
	clock['text'] = count

	if count > 0:
		# call countdown again after 1000ms (1s)
		root.after(1000, countdown, count-1)
	else:
		print("Timer Finished")

#Clock Label
clock = Label(root, text = 'Click begin to start timer')
clock.pack(ipady=5, fill='x', anchor = 'e')

#Button to begin timer
begin = Button(root, text='Begin', width=25, bg = "white", command=go)
begin.pack()

def open_file():
	global fulldata
	import re
	
	print("1. Select File")
	rep = filedialog.askopenfilenames()
	tic = time.perf_counter() #Starting timer for 1
	rep = re.sub("[()]","", str(rep))
	#print("Filepath:",rep)
	#rep = rep.replace("/", "\\")
	rep = rep.replace("'", "")
	rep = rep.replace(",", "")
	path=os.path.dirname(rep)
	foldername = os.path.basename(path)
	test_csv = pd.read_csv(str(rep), header=0, skip_blank_lines=True)
	toc = time.perf_counter() #Finishing timer for 1
	print(f"Reading file took {toc - tic:0.4f} seconds")

	print("2. Tranforming the csv")
	tic2 = time.perf_counter() #Starting timer for 2
	Df = test_csv
	# dropping first row as it shows unit
	Df.drop([Df.index[0]], inplace=True)
	Df = Df[1000:13000]
	# converting everything from string to float
	Df = Df.applymap(lambda x : float(x))
	Df['EWM_mean'] = Df.iloc[:,1].ewm(span=200,adjust=True).mean()
	Df.drop(['Time', 'Channel A'], inplace=True, axis=1)
	nw_df = Df.T
	toc2 = time.perf_counter() #Finishing timer for 2
	print(f"Transforming the csv took {toc2 - tic2:0.4f} seconds")

	print("3. Sorting data with labels and features  ")
	tic3 = time.perf_counter() #Starting timer for 3
	#fulldata = pd.read_csv('/content/drive/MyDrive/full_dataset.csv')
	#features = [str(i) for i in range(1001,13001)]
	features1 = [i for i in range(1001,13001)]
	#x = fulldata.loc[:, features].values
	#y = fulldata.loc[:,['label']].values
	X = nw_df.loc[:, features1].values
	toc3 = time.perf_counter() #Finishing timer for 3
	print(f"Sorting the the csv took {toc3 - tic3:0.4f} seconds")

	print("4. Applying PCA")
	tic4 = time.perf_counter() #Starting timer for 4
	#principalComponents = pca.fit_transform(x)
	newdata_transformed = pca.transform(X)
	pc = pca.explained_variance_ratio_.cumsum()
	x1 = StandardScaler().fit(principalComponents)
	X1 = x1.transform(newdata_transformed)
	newdf = pd.DataFrame(data = X1, columns = [f'pc_stdscaled_{i}' for i in range(len(pc))])
	toc4 = time.perf_counter() #Finishing timer for 4
	print(f"Applying PCA took {toc4 - tic4:0.4f} seconds")

	print(X1)

	print("5. Model evaluate")
	tic5 = time.perf_counter() #Starting timer for 5
	pred = model.predict(np.array(X1))
	gesture_list = ["Anti-Clockwise", "Clockwise","Double Hand", "Left Swipe", "Right Swipe"]
	# print(gesture_list[pred[0]])
	prediction = gesture_list[pred[0]]
	update(prediction)
	detection = prediction
	switch(detection)
	toc5 = time.perf_counter() #Finishing timer for 5

	print(f"Applying PCA took {toc5 - tic5:0.4f} seconds")

	print("")
	print(f"Total time {toc5 - tic:0.4f} seconds")
	print()
	print("Folder Name: ",foldername)
	print("Prediction: ",prediction)
	print("Detection: ",detection)

	if (prediction[0].lower() == foldername[0]):
		print("PASSED: Correct Detection")
	else:
		print("FAILED: Incorrect Detection")


#Button to open file
file = Button(root, text='Open File', width=25, bg = "white", command=open_file)
file.pack()

#Dial knob for rotating once clockwise/anticlockwise has been detected
image = Image.open("dial_knob2.png")
width, height = image.size
image.thumbnail((width/5, height/5))
photoimage = ImageTk.PhotoImage(image)
image_label = Label(root, image=photoimage, bg="#6FAFE7", relief=FLAT)
image_label.image = photoimage
image_label.pack(padx=10, pady=10)

#To rotate file
def rotate_image(degrees):
	new_image = image.rotate(-int(float(degrees)))
	photoimage = ImageTk.PhotoImage(new_image)
	image_label.image = photoimage #Prevent garbage collection
	image_label.config(image = photoimage)
	
def slider(slider_pos):
	current = w2.get()
	print(current)
	move = current + slider_pos
	w2.set(move)


#Slider once swipe left/right has been detected
w2 = ttk.Scale(root, from_=0, to=360, orient=HORIZONTAL, length=300)
w2.pack(padx=10, pady=10)
w2.set(180)


#Change colour of select button
def light1():
	global light1status, strongboxit
	if light1status == 0:
		button.config(bg = "red")
		light1status = 1

	else:
		light1status = 0
		button.config(bg = "green")


#Select button for double hand detection 
button = Button(root, text='Select', width=25, bg = "green", command=light1)
button.pack(padx=10, pady=10)

#Update text depending on detetion
def update(detection):
	global gesture_number
	gesture_number += 1
	Heading.config(text = str(gesture_number) + '.' + detection)

#Display text of gesture
Heading = Label(root, text="hello", bg="#2176C1", fg='white', relief=RAISED)
Heading.pack(side="bottom",ipady=5, fill='x')
Heading.config(font=("Font", 20)) # change font and size of label


root.mainloop()