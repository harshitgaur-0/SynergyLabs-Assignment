from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rescale=(1/255),validation_split=0.2)
train_data = image_gen.flow_from_directory(directory=r"./vehicle_classification_data_200/",batch_size=32,target_size=(25,67),shuffle=True,class_mode="categorical",subset="training")
validation_data = image_gen.flow_from_directory(r"./vehicle_classification_data_200/",batch_size=32,target_size=(25,67),shuffle=True,class_mode="categorical",subset="validation")

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten
from keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor="val_accuracy",patience = 5)

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(25,67,3),padding="same"))
model.add(Conv2D(32,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="same"))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(3,activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit_generator(train_data,epochs=10,validation_data=validation_data,callbacks=[callback])

