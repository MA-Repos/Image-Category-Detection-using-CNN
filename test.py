import cv2
import numpy as np
from keras.models import load_model
from keras.datasets import cifar10

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

img_row, img_height, img_depth = 32,32,3
classifier = load_model('TrainedModels/cifar_simple_cnn_with_5_epoch.h5')
categ_mapping = {
        '0' : "airplane",
        '1' : "automobile",
        '2' : "bird",
        '4' : "cat",
        '5' : "deer",
        '6' : "dog",
        '7' : "frog",
        '8' : "ship",
        '9' : "truck"
}

for i in range(0,5):
    rand = np.random.randint(0,len(xTest))
    input_img = xTest[rand]
    image_window = cv2.resize(input_img, None, fx=10, fy=10, interpolation = cv2.INTER_CUBIC) 
    input_img = input_img.reshape(1,img_row, img_height, img_depth) 
    
    res = str(classifier.predict_classes(input_img, 1, verbose = 0)[0])
    
    pred = categ_mapping[res]
        
    image_text = cv2.copyMakeBorder(image_window, 0, 0, 0, image_window.shape[0]*2 ,cv2.BORDER_CONSTANT,value=[128,0,0])
    cv2.putText(image_text, str(pred), (400, 120) , cv2.FONT_HERSHEY_COMPLEX_SMALL,3, (255,255,255), 2)
    cv2.imshow("Category detection prediction", image_text)
    cv2.waitKey(0)

cv2.destroyAllWindows()