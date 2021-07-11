from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
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

teacher_model = load_model('teacher.h5')


def custom_loss_wrapper(curr_input_tensor):
    def distillation_loss(y_true, y_pred):
        p = 0.9
        alpha = 1.
        beta = 1.
        y_teacher = teacher_model(curr_input_tensor)
        soft_student_softmax = K.exp(y_pred / p) / K.sum(K.exp(y_pred / p))
        soft_teacher_softmax = K.exp(y_teacher / p) / K.sum(K.exp(y_teacher / p))
        return alpha * categorical_crossentropy(y_true, y_pred) + beta * categorical_crossentropy(soft_teacher_softmax, soft_student_softmax)
    return distillation_loss


def conv_block(prev_layer, num_filters, kernel_size, padding):
    u = Conv2D(num_filters, kernel_size, padding=padding)(prev_layer)
    u = BatchNormalization()(u)
    u = Activation('relu')(u)
    u = Dropout(0.2)(u)
    return u


input_tensor = Input(shape=(32, 32, 3))
x = conv_block(input_tensor, 32, 3, 'same')
x = MaxPooling2D()(x)
x = conv_block(x, 64, 3, 'same')
x = MaxPooling2D()(x)
x = conv_block(x, 128, 3, 'same')
x = MaxPooling2D()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
output_tensor = Dense(10, activation='softmax')(x)
model = Model(input_tensor, output_tensor)
model.summary()
model.compile(optimizer=Adam(lr=1e-3), loss=custom_loss_wrapper(input_tensor), metrics=['acc'])
callbacks = [EarlyStopping(min_delta=0.002, patience=4), ReduceLROnPlateau(patience=1)]
model.fit(x_train, y_train, 128, 1000, validation_data=(x_test, y_test), callbacks=callbacks)

model.save('student.h5', include_optimizer=False)

