from flask import*
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image

app=Flask(__name__)
UPLOAD='static/data'
app.config['UPLOAD']=UPLOAD
@app.route('/')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=="POST":
        file=request.files['data']
        na=file.filename
        file.save(os.path.join(app.config['UPLOAD'],file.filename))
        # Load the model architecture from a JSON file
        with open(r"modell1.json", 'r') as json_file:
            loaded_model_json = json_file.read()

        # Load the trained model weights
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(r"model1.h5")

        #>>>>>>>>>>>>>>>>>>>>>>>STEP-3:DEFINE CLASSES
        # Define the labels of the classes
        class_labels = ['daisy', 'dandelion', 'rose', 'sunflower','tulip']

        #>>>>>>>>>>>>>>>>>>>>>>>STEP-4:TESTING IMAGE
        # Load the test image
        test_path=r"static/data/"+na
        test_image = image.load_img(test_path, target_size=(300, 300))
        plt.imshow(test_image)
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
        os.remove(test_path)
        return render_template('flower-html.html',name=predicted_class)
    return render_template('flower-html.html',name='flower')

if __name__=='__main__':
    app.run(debug=True)
