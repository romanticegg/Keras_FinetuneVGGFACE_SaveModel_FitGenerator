# GAN - step01 - Get pre-trained VGGFACE model and fine-tune it on IJB-A target split training data
import numpy as np
from keras.utils import np_utils
import cv2
from keras.optimizers import SGD
import keras.callbacks
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras_vggface.vggface import VGGFace
import keras.callbacks
import h5py
from PIL import Image
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import setGPU
import os


img_w = 224
img_h = 224
channels = 3
classes = 332
batch_size = 32
epochs = 3
# Finetune list
Finetune_file = open("/home/zhaojian/Keras/datasets/IJB-A/IJBA_split1/train/IJBA_split1_finetune.txt", "r")
Finetune_lines = Finetune_file.readlines()
Finetune_file.close()
N_Finetune = len(Finetune_lines)
# Val list
Val_file = open("/home/zhaojian/Keras/datasets/IJB-A/IJBA_split1/train/IJBA_split1_val.txt", "r")
Val_lines = Val_file.readlines()
Val_file.close()
N_Val = len(Val_lines)

def finetune_generator(Finetune_lines, N_Finetune, batch_size, classes, img_w, img_h):
    batch_nb = -1
    epoch_nb = 1
    batch_total = len(Finetune_lines) / batch_size
    Finetune_label = np.zeros([N_Finetune, 1], dtype=np.int)
    Finetune_file_list = []
    for i in range(N_Finetune):
        Finetune_file_list.append(Finetune_lines[i].split()[0])
        Finetune_label[i] = int(Finetune_lines[i].split()[1])
    Finetune_label = np_utils.to_categorical(Finetune_label, num_classes = classes)
    while 1:
        batch_nb += 1
        if batch_nb == batch_total:
            epoch_nb += 1
            batch_nb = 0
        start_idx = batch_nb * batch_size
        Img_data = np.zeros((batch_size, img_w, img_h, 3), dtype=np.float32)
        Img_Label  = np.zeros((batch_size, classes), dtype=np.float32)
        for k in xrange(batch_size):
            img_file = Finetune_file_list[start_idx+k]
            img = image.img_to_array(image.load_img(img_file, target_size=(img_w, img_h)))
            Img_data[k, ...] = img
            Img_Label[k, ...] = Finetune_label[(start_idx + k), ...]
        Img_data = preprocess_input(Img_data)
        yield Img_data, Img_Label

def val_generator(Val_lines, N_Val, batch_size, classes, img_w, img_h):
    batch_nb = -1
    epoch_nb = 1
    batch_total = len(Val_lines) / batch_size
    Val_label = np.zeros([N_Val, 1], dtype=np.int)
    Val_file_list = []
    for i in range(N_Val):
        Val_file_list.append(Val_lines[i].split()[0])
        Val_label[i] = int(Val_lines[i].split()[1])
    Val_label = np_utils.to_categorical(Val_label, num_classes = classes)
    while 1:
        batch_nb += 1
        if batch_nb == batch_total:
            epoch_nb += 1
            batch_nb = 0
        start_idx = batch_nb * batch_size
        Img_data = np.zeros((batch_size, img_w, img_h, 3), dtype=np.float32)
        Img_Label  = np.zeros((batch_size, classes), dtype=np.float32)
        for k in xrange(batch_size):
            img_file = Val_file_list[start_idx+k]
            img = image.img_to_array(image.load_img(img_file, target_size=(img_w, img_h)))
            Img_data[k, ...] = img
            Img_Label[k, ...] = Val_label[(start_idx + k), ...]
        Img_data = preprocess_input(Img_data)
        yield Img_data, Img_Label

image_input = Input(shape=(img_w, img_h, channels))
vggface_model = VGGFace(input_tensor=image_input, include_top=False)
last_layer = vggface_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(4096, activation='relu', name='fc6')(x)
x = Dense(4096, activation='relu', name='fc7')(x)
out = Dense(classes, activation='softmax', name='fc8')(x)
keras_vggface_IJBA = Model(image_input, out)
sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
print(keras_vggface_IJBA.summary())

keras_vggface_IJBA.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

keras_vggface_IJBA_json = keras_vggface_IJBA.to_json()
with open("/home/zhaojian/Keras/Projects/GAN/models/VGGFACE/keras_vggface_IJBA.json", "w") as json_file:
    json_file.write(keras_vggface_IJBA_json)
# checkpoint
filepath="/home/zhaojian/DEEP/Keras/Projects/GAN/models/VGGFACE/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

keras_vggface_IJBA.fit_generator(finetune_generator(Finetune_lines, N_Finetune, batch_size, classes, img_w, img_h), steps_per_epoch=(N_Finetune/batch_size), epochs=epochs, verbose=1, validation_data = val_generator(Val_lines, N_Val, batch_size, classes, img_w, img_h), validation_steps = (N_Val/batch_size), callbacks=callbacks_list)

# load json and create model
#json_file = open('/home/zhaojian/DEEP/Keras/Projects/GAN/models/VGGFACE/keras_vggface_IJBA_json.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#keras_vggface_IJBA = model_from_json(loaded_model_json)
