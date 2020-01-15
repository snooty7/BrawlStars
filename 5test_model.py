from __future__ import print_function
import numpy as np
from grabscreen import grab_screen
import cv2
import time
import pywinauto
from directkeys import PressKey,ReleaseKey, W, A, S, D
from getkeys import key_check
from collections import deque, Counter
import random
from statistics import mode,mean
from motion import motion_detection
#############################################################
# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.linear_model import LogisticRegression
import os
import json
import pickle
########################################################################

GAME_WIDTH = 800
GAME_HEIGHT = 600

how_far_remove = 800
rs = (20,15)
log_len = 25

motion_req = 800
motion_log = deque(maxlen=log_len)

WIDTH = 299
HEIGHT = 299
LR = 1e-3
EPOCHS = 10

choices = deque([], maxlen=5)
hl_hist = 250
choice_hist = deque([], maxlen=hl_hist)
##########################################################
# load the user configs
with open('conf.json') as f:    
	config = json.load(f)

# config variables
model_name 		= config["model"]
weights 		= config["weights"]
include_top 	= config["include_top"]
train_path 		= config["train_path"]
test_path 		= config["test_path"]
features_path 	= config["features_path"]
labels_path 	= config["labels_path"]
test_size 		= config["test_size"]
results 		= config["results"]
model_path 		= config["model_path"]
seed 			= config["seed"]
classifier_path = config["classifier_path"]

# load the trained logistic regression classifier
print ("[INFO] loading the classifier...")
classifier = pickle.load(open(classifier_path, 'rb'))

#########################################################################

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

t_time = 0.25

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    pywinauto.mouse.click(button='left', coords=(776, 491))

def left():
    if random.randrange(0,3) == 1:
        PressKey(W)
        pywinauto.mouse.click(button='left', coords=(776, 491))
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    pywinauto.mouse.click(button='left', coords=(776, 491))
    #ReleaseKey(S)

def right():
    if random.randrange(0,3) == 1:
        PressKey(W)
        pywinauto.mouse.click(button='left', coords=(776, 491))
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    pywinauto.mouse.click(button='left', coords=(776, 491))
    
def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    pywinauto.mouse.click(button='left', coords=(776, 491))


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    pywinauto.mouse.click(button='left', coords=(776, 491))
    
    
def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    pywinauto.mouse.click(button='left', coords=(776, 491))

    
def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    pywinauto.mouse.click(button='left', coords=(776, 491))

    
def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    pywinauto.mouse.click(button='left', coords=(776, 491))

def no_keys():

    if random.randrange(0,3) == 1:
        PressKey(W)
        pywinauto.mouse.click(button='left', coords=(776, 491))
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    pywinauto.mouse.click(button='left', coords=(776, 491))
    
############################################################
base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
model = Model(input=base_model.input, output=base_model.layers[-1].output)
image_size = (299, 299)

train_labels = os.listdir(train_path)

# get all the test images paths
test_images = os.listdir(test_path)
#######################################################################################
#model = googlenet(WIDTH, HEIGHT, 3, LR, output=9)
MODEL_NAME = ''
#model.load(MODEL_NAME)

print('We have loaded a previous model!!!!')

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    mode_choice = 0

    screen = grab_screen(region=(0,40,GAME_WIDTH,GAME_HEIGHT+40))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(screen, (WIDTH,HEIGHT))

    t_minus = prev
    t_now = prev
    t_plus = prev

    while(True):
        
        if not paused:
            screen = grab_screen(region=(0,40,GAME_WIDTH,GAME_HEIGHT+40))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            last_time = time.time()
            screen = cv2.resize(screen, (WIDTH,HEIGHT))

            delta_count_last = motion_detection(t_minus, t_now, t_plus)

            t_minus = t_now
            t_now = t_plus
            t_plus = screen
            t_plus = cv2.blur(t_plus,(4,4))

            #prediction = model.predict([screen.reshape(WIDTH,HEIGHT,3)])[0]
            #prediction = np.array(prediction) * np.array([4.5, 0.1, 0.1, 0.1,  1.8,   1.8, 0.5, 0.5, 0.2])

            ###############################################
            #img = image.load_img(screen, target_size=image_size)
            #x 			= image.img_to_array(img)
            x 			= np.expand_dims(screen, axis=0)
            x 			= preprocess_input(x)
            feature 	= model.predict(x)
            flat 		= feature.flatten()
            flat 		= np.expand_dims(flat, axis=0)
            preds 		= classifier.predict(flat)
            prediction 	= train_labels[preds[0]]
            ###############################################################################
            mode_choice = prediction

            if mode_choice == 'straight':
                straight()
                choice_picked = 'straight'
                
            elif mode_choice == 'reverse':
                reverse()
                choice_picked = 'reverse'
                
            elif mode_choice == 'left':
                left()
                choice_picked = 'left'
            elif mode_choice == 'right':
                right()
                choice_picked = 'right'
            elif mode_choice == 'forward+left':
                forward_left()
                choice_picked = 'forward+left'
            elif mode_choice == 'forward+right':
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 'reverse+left':
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 'reverse+right':
                reverse_right()
                choice_picked = 'reverse+right'
            elif mode_choice == 'nokeys':
                no_keys()
                choice_picked = 'nokeys'

            motion_log.append(delta_count_last)
            motion_avg = round(mean(motion_log),3)
            print('loop took {} seconds. Motion: {}. Choice: {}'.format( round(time.time()-last_time, 3) , motion_avg, choice_picked))
            
            if motion_avg < motion_req and len(motion_log) >= log_len:
                print('WERE PROBABLY STUCK FFS, initiating some evasive maneuvers.')

                # 0 = reverse straight, turn left out
                # 1 = reverse straight, turn right out
                # 2 = reverse left, turn right out
                # 3 = reverse right, turn left out

                quick_choice = random.randrange(0,4)
                
                if quick_choice == 0:
                    reverse()
                    time.sleep(random.uniform(1,2))
                    forward_left()
                    time.sleep(random.uniform(1,2))

                elif quick_choice == 1:
                    reverse()
                    time.sleep(random.uniform(1,2))
                    forward_right()
                    time.sleep(random.uniform(1,2))

                elif quick_choice == 2:
                    reverse_left()
                    time.sleep(random.uniform(1,2))
                    forward_right()
                    time.sleep(random.uniform(1,2))

                elif quick_choice == 3:
                    reverse_right()
                    time.sleep(random.uniform(1,2))
                    forward_left()
                    time.sleep(random.uniform(1,2))

                for i in range(log_len-2):
                    del motion_log[0]
    
        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       
