import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('training_data-1.npy', allow_pickle=True)

#df = pd.DataFrame(train_data)
#print(df.head())
#print(Counter(df[1].apply(str)))
i = 1
for data in train_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('test', img)
    print(choice)
    if choice == [1,0,0,0,0,0,0,0,0]:
        cv2.imwrite('C:\\Users\\Snooty7\\Desktop\\BrawlStars\\straight\\{}.jpg'.format(i),img)
        i+=1
    if choice == [0,1,0,0,0,0,0,0,0]:
        cv2.imwrite('C:\\Users\\Snooty7\\Desktop\\BrawlStars\\reverse\\{}.jpg'.format(i),img)
        i+=1
    if choice == [0,0,1,0,0,0,0,0,0]:
        cv2.imwrite('C:\\Users\\Snooty7\\Desktop\\BrawlStars\\left\\{}.jpg'.format(i),img)
        i+=1
    if choice == [0,0,0,1,0,0,0,0,0]:
        cv2.imwrite('C:\\Users\\Snooty7\\Desktop\\BrawlStars\\right\\{}.jpg'.format(i),img)
        i+=1 
    if choice == [0,0,0,0,1,0,0,0,0]:
        cv2.imwrite('C:\\Users\\Snooty7\\Desktop\\BrawlStars\\forward+left\\{}.jpg'.format(i),img)
        i+=1 
    if choice == [0,0,0,0,0,1,0,0,0]:
        cv2.imwrite('C:\\Users\\Snooty7\\Desktop\\BrawlStars\\forward+right\\{}.jpg'.format(i),img)
        i+=1   
    if choice == [0,0,0,0,0,0,1,0,0]:
        cv2.imwrite('C:\\Users\\Snooty7\\Desktop\\BrawlStars\\reverse+left\\{}.jpg'.format(i),img)
        i+=1   
    if choice == [0,0,0,0,0,0,0,1,0]:
        cv2.imwrite('C:\\Users\\Snooty7\\Desktop\\BrawlStars\\reverse+right\\{}.jpg'.format(i),img)
        i+=1    
    if choice == [0,0,0,0,0,0,0,0,1]:
        cv2.imwrite('C:\\Users\\Snooty7\\Desktop\\BrawlStars\\nokeys\\{}.jpg'.format(i),img)
        i+=1                                        
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
#print(np.shape(train_data))
#print(np.ndim(train_data))