import matplotlib.pyplot as plt 
import numpy as np 
import os 
import cv2
import random
import pickle
import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
# !pip install scikeras
from scikeras.wrappers import KerasClassifier
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import optimizers
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix

# %matplotlib inline

def plot_results(mod_history, metric, epochs):
      
      # Check out our train loss and test loss over epochs.
      train_metric = mod_history.history[metric]
      val = 'val_' + metric
      test_metric = mod_history.history[val]

      # Set figure size.
      plt.figure(figsize=(12, 8))

      # Generate line plot of training, testing loss over epochs.
      plt.plot(train_metric, label=f'Training {metric}', color='#185fad')
      plt.plot(test_metric, label=f'Testing {metric}', color='orange')

      # Set title
      plt.title(f'Training and Testing {metric} by Epoch', fontsize = 25)
      plt.xlabel('Epoch', fontsize = 18)
      plt.ylabel('Categorical Crossentropy', fontsize = 18)
      plt.xticks(range(0,epochs,5), range(0,epochs,5))
      plt.legend(fontsize = 18);



def make_predictions(mod_name, steps=20):
    preds = mod_name.predict(X_test,steps=steps)
    preds = preds.argmax(axis=-1)

    y_test_labels = np.argmax(y_test, axis=-1)

    cm = confusion_matrix(y_test_labels,preds)

    plot_confusion_matrix(cm, cm_plot_labels, normalize=True,
                          title='Face Shape Normalized')

    plt.show()



cm_plot_labels = ['Heart','Rectangle','Oval','Round', 'Square', 'Triangle']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(16,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def plot_summary_results(mod_name, mod_history, epochs):
    plot_results(mod_history, 'loss',epochs)
    plot_results(mod_history, 'accuracy', epochs)
    make_predictions(mod_name)


path = './pickle_out/rgb/'

X_train = np.asarray(pickle.load(open(path + "pickle_out_rgbX_train_rgb.pickle","rb")))
y_train = np.asarray(pickle.load(open(path + "pickle_out_rgby_train_rgb.pickle","rb")))
X_test = np.asarray(pickle.load(open(path + "pickle_out_rgbX_test_rgb.pickle","rb")))
y_test = np.asarray(pickle.load(open(path + "pickle_out_rgby_test_rgb.pickle","rb")))


print("Data Summary")
print("--------------------")
print(f"X_train shape {X_train.shape}")
print(f"y_train shape {y_train.shape}")
print("--------------------")
print(f"X_test shape {X_test.shape}")
print(f"y_test shape {y_test.shape}")


# Path for VGGFace weights

path_vggface = './SavedModels/vgg_face_weights.h5'


# Loading VGG16 as base model

# base_model = VGG16(input_shape=(224, 224, 3),  # same as our input
#                    include_top=False,  # exclude the last layer
#                    weights=path_vggface)  # use VGGFace Weights

base_model = VGG16(input_shape=(224, 224, 3),  # same as our input
                   include_top=False,  # exclude the last layer
                   weights=None)  # use VGGFace Weights


for layer in base_model.layers:
  layer.trainable = False


model_t1 = Sequential()


# Compile and Fit the model

x = layers.Flatten()(base_model.output)

x = layers.Dense(64, activation='relu')(x)  # add 1 fully connected layer, try with 512 first 
x = layers.Dropout(0.5)(x)
x = layers.Dense(6, activation='softmax')(x)  # add final layer

model_t1 = tf.keras.models.Model(base_model.input, x)



model_t1.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

model_t1.summary()



datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)


datagen.fit(X_train)


history_t1 = model_t1.fit(datagen.flow(X_train, y_train, batch_size=32), 
                          steps_per_epoch=int(len(X_train)/32), epochs=50, 
                          validation_data=(X_test, y_test))


model_path = './SavedModels/VGGFACE.keras'
tf.keras.models.save_model(
    model_t1, filepath=model_path, overwrite=True, include_optimizer=True, save_format=None)


filename = model_path + 'vgg16-face-1.keras'   # change the filename for new iterations
model_t1.save(filename)


loaded_model = tf.keras.models.load_model(filename)
mod_t1_predict = np.argmax(model_t1.predict(X_test), axis=1) 
loaded_t1_predict = np.argmax(loaded_model.predict(X_test), axis=1)

# Check the difference

print(f'Difference in predictions: Saved model vs. original model is {np.sum(loaded_t1_predict - mod_t1_predict)}\nModel was correctly saved.')



plot_summary_results(model_t1, history_t1, 50)

history_t2 = model_t1.fit(datagen.flow(X_train, y_train, batch_size=32), 
                          steps_per_epoch=int(len(X_train)/32), epochs=20, 
                          validation_data=(X_test, y_test))


filename = model_path + 'vgg16-face-2.keras'   # change the filename for new iterations
model_t1.save(filename)


loaded_model = tf.keras.models.load_model(filename)
mod_t1_predict = np.argmax(model_t1.predict(X_test), axis=1) 
loaded_t1_predict = np.argmax(loaded_model.predict(X_test), axis=1)

# Check the difference

print(f'Difference in predictions: Saved model vs. original model is {np.sum(loaded_t1_predict - mod_t1_predict)}')



plot_summary_results(model_t1, history_t2, 20)
      

history_t3 = model_t1.fit(datagen.flow(X_train, y_train, batch_size=32), 
                          steps_per_epoch=int(len(X_train)/32), epochs=10, 
                          validation_data=(X_test, y_test))