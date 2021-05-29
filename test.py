import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import numpy as np
import cv2 as cv
import cv2

BS = 8
data = []
new_model = tf.keras.models.load_model('tumor.model')

im = cv.imread('subjects/1.png') #change to the desired image to test
image = cv.cvtColor(im, cv.COLOR_BGR2RGB)
image = cv.resize(image, (224, 224))
data.append(image)
data = np.array(data) / 255.0

predIdxs = new_model.predict(data)
prob_normal = predIdxs[0][1] * 100;
prob_cancer  = predIdxs[0][0] * 100;

print("Probabilite d'etre sain: %.2f" % prob_normal)
print("Probabilite du Cancer: %.2f" % prob_cancer)

if prob_normal > 40 : 
    image = cv.resize(image, (600, 600))
    text = "{:.2f}%".format(prob_normal)
    cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.imshow("Sain", image)
    cv.waitKey()
else: 
    image = cv.resize(image, (600, 600))
    text = "{:.2f}%".format(prob_cancer)
    cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.imshow("Malade",image)
    cv.waitKey()