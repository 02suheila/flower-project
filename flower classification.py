import os
#import matplotlib.pyploty as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

#Directory with angry images
tulip_dir= os.path.join(r"D:\flowers\tulip")
#Directory wuth happy images
sf_dir= os.path.join(r"D:\flowers\sunflower")
#Directory with neuatral images
rose_dir = os.path.join(r"D:\flowers\rose")
#Directory with sad images
dandelion_dir = os.path.join(r"D:\flowers\dandelion")
#directory with surprise images
daisye_dir = os.path.join(r"D:\flowers\daisy")

train_tulip_names = os.listdir(tulip_dir)
print("tulip:",len(train_tulip_names))

train_sf_names = os.listdir(sf_dir)
print("sunflower:",len(train_sf_names))
train_rose_names = os.listdir(rose_dir)
print("rose:",len(train_rose_names))
train_dandelion_names = os.listdir(dandelion_dir)
print("dandalion:",len(train_dandelion_names))
train_daisy_names = os.listdir(daisye_dir)
print("daisy:",len(train_daisy_names))

total=len(train_tulip_names)+len(train_sf_names)+len(train_rose_names)+len(train_dandelion_names)+len(train_daisy_names)
print("Total Flowers:",total)

batch_size = 22

#All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)

#Flow training images in batches of 128 using train_datagen
train_generator = train_datagen.flow_from_directory(r"D:\flowers",
                                                    target_size=(300,300), #All images will be resized
                                                    batch_size=batch_size,
                                                    color_mode='grayscale',#specify the classes explicitly
                                                    classes = ['daisy','dandelion','rose','sunflower','tulip',],
                                                    #since we use categorical_crossentropy loss,we need
                                                    class_mode='categorical')
target_size=(300,300)


model = tf.keras.models.Sequential([
#Note the input shape is the desired size of the image 48*48 with 3 byte color
#The first convolutioin
tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(300,300,1)),
tf.keras.layers.MaxPooling2D(2, 2),

#The second convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2, 2),

#the third convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2, 2),

#Flatten the results to feed into a dense layer
tf.keras.layers.Flatten(),
#64 neuron in the fully-connected layer
tf.keras.layers.Dense(64, activation='relu'),
#5 output neurons for 5 classes with the softmax activation
tf.keras.layers.Dense(5,activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])#RMSprop(lr=0.001),
#Total sample count
total_sample=train_generator.n
#trainiing
num_epochs = 5
model.fit_generator(train_generator,
                    steps_per_epoch=int(total_sample/batch_size),
                    epochs=num_epochs,
                    verbose=1)



#serialize model to JSON
model_json = model.to_json()
with open("modell1.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights("model1.h5")
print("SUCCESSFULLY SAVED TO DISK")




