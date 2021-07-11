from tensorflow.keras.applications import vgg19
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.
y_train = to_categorical(y_train)
x_test = x_test / 255.
y_test = to_categorical(y_test)

model = vgg19.VGG19(weights=None, input_shape=(32, 32, 3), classes=10)
model.summary()
model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['acc'])
callbacks = [EarlyStopping(min_delta=0.002, patience=4), ReduceLROnPlateau(patience=1)]
model.fit(x_train, y_train, 128, 1000, validation_data=(x_test, y_test), callbacks=callbacks)

model.save('teacher.h5', include_optimizer=False)
