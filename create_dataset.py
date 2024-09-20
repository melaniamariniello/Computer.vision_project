import os
#pickle is a python library commonly used to save data
#The pickle module implements binary protocols for serializing and de-serializing a Python object structure. 
#“Pickling” is the process whereby a Python object hierarchy is converted into a byte stream,
import pickle
import mediapipe as mp #MediaPipe Solutions provides a suite of libraries and tools for 
#you to quickly apply artificial intelligence (AI) and machine learning (ML) techniques in your applications
import cv2
import matplotlib.pyplot as plt  #because it'll be plotted one images for each directory in order to verify if the code works well


#Iterate in all images, extract the landmarks from each images and 
# we're going to save all this data into a file we're later going to use in order to train or classifier.

#THESE ARE THREE OBJECTS WHICH ARE GOING TO BE USEFUL TO DETECT ALL THE LANDMARK IN ORDER TO DRAW THESE LANDMARKS ON TOP OF THE IMAGES; 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#Define an object hands: an hand detector
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#Define an object hands: an hand detector
DATA_DIR = './data'

# Two variables, which will contain all the information; Lables define the classes, while data encapsulate the image data
data = []
labels = []

#Iterate in all frames
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
    #    for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]: PLOT ONLY THE FIRST IMAGE FOR EACH CLASSES    

        data_aux = []

        x_ = []
        y_ = []

        #convert the image into RGB in order input the images into media pipe
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #detects all the landmarks into this image, and iterate for all the landmarks detected
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                #VERIFY IF THE CODE WORKS ONE TIME, PLOT ONLY THE LANDMARK FOR ONE IMAGE
               #mp_drawing.draw_landmarks(img_rgb, 
                                #     hand_landmarks, 
                                #     mp_hands.HAND_CONNECTIONS, 
                                #    mp_drawing_styles.get_default_hand_landmarks_style(), 
                                #    mp_drawing_styles.get_default_hand_connections_style())
                
            #plt.figure()
            #plt.imshow(img_rgb)
            #plt.show()
#
# NOW we want to take entire landmarks and create an array for all the landmarks 
    # Take all the images, from each of them take an array with the landmarks' information detected
                for i in range(len(hand_landmarks.landmark)):
                    #to test:
                    #print(hand_landmarks.landmark[i]) for each landmark we have three value along x,y,z, which define their position
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

#save these information o data.pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()