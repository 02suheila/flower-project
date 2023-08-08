#>>>>>>>>>>>>>>>>>>>>>>>>step-1:IMPORT LIBRARIES
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image

#>>>>>>>>>>>>>>>>>>>>>>>step-2:LOADING MODEL FROM JSON FILE AND TRAINED MODEL
# Load the model architecture from a JSON file
with open(r"C:\Users\Ibrahim Shaik\Documents\flower-project\modell1.json", 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the trained model weights
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(r"C:\Users\Ibrahim Shaik\Documents\flower-project\model1.h5")
print("\n\n********** Model loaded from disk **********n\n")

#>>>>>>>>>>>>>>>>>>>>>>>STEP-3:DEFINE CLASSES
# Define the labels of the classes
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower','tulip']

#>>>>>>>>>>>>>>>>>>>>>>>STEP-4:TESTING IMAGE
# Load the test image
test_image = image.load_img(r"D:\flowers\dandelion\61242541_a04395e6bc.jpg", target_size=(300, 300, 1))
plt.imshow(test_image)
print("Predicting result...\n")
test_image = np.array(test_image)

# Convert the test_image to grayscale
gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to a numpy array
test_image = image.img_to_array(gray_image)

# Add a batch dimension to the image
test_image = np.expand_dims(test_image, axis=0)

#>>>>>>>>>>>>>>>>>>>>>>>>STEP-5:PREDICTION
# Make a prediction on the test image using the loaded model
result = loaded_model.predict(test_image)
# Get the predicted class label based on the highest probability
predicted_class = class_labels[np.argmax(result)]
print("\nAnd the predicted flower is", predicted_class)
