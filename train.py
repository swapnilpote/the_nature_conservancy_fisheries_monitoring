from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

batch_size = 16

trn_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
val_datagen = ImageDataGenerator(rescale = 1./255)


trn_generator = trn_datagen.flow_from_directory('data/train', target_size = (150, 150), batch_size = batch_size, class_mode = 'categorical')
val_generator = val_datagen.flow_from_directory('data/valid', target_size = (150, 150), batch_size = batch_size, class_mode = 'categorical')


model.fit_generator(trn_generator, steps_per_epoch = trn_generator.n / batch_size, epochs = 5, validation_data = val_generator,
        validation_steps = val_generator.n/ batch_size)

model.save_weights('model.h5')