from turtledemo.__main__ import font_sizes

import fontTools.designspaceLib.statNames
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras import layers
from keras.src.engine.training_utils_v1 import standardize_class_weights
from keras .utils import image_dataset_from_directory
import os
import numpy as np
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Naziv foldera u kojem se nalaze svi odbirci klasa
data = 'data'

#Parametri koji se rucno menjaju, sluze za podesavanje rada mreze
img_size = (128, 128)
bs = 64
ep = 200
regularizer = 0.005
lr = 0.001

#Ucitavanje svih podataka iz foldera 'data'
Data = image_dataset_from_directory(data,
                                    image_size=img_size,
                                    batch_size=bs,
                                    seed=123)

#Vrste klasa
class_labels = Data.class_names

#Brojanje odbiraka svake klase
class_counts = []
num_elements = 0
for label in class_labels:
    class_path = os.path.join(data, label)
    if os.path.isdir(class_path):
        num_elements = len(os.listdir(class_path))
        class_counts.append(num_elements)

#Prikaz broja odbiraka svake klase
plt.figure(figsize=(12, 12))
plt.barh(class_labels, class_counts, color='blue')
plt.xlabel('Class Labels')
plt.ylabel('Number of Elements')
plt.title('Number of Elements in Each Class')
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

#Prikaz po jednog odbirka iz svake klase
N = 53
plt.figure(figsize=(12, 12))
for img, lab in Data.take(1):
    for i in range(N):
        plt.subplot(8, 7, i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(class_labels[lab[i]], fontsize=9.5)
        plt.axis('off')
        plt.tight_layout(pad=1.0)


#Podela Data skupa na train, valid i test skup
Xtrain = image_dataset_from_directory(data,
                                      subset='training',
                                      validation_split=0.4,
                                      image_size=img_size,
                                      batch_size=bs,
                                      seed=123)

Xval = image_dataset_from_directory(data,
                                    subset='validation',
                                    validation_split=0.4,
                                    image_size=img_size,
                                    batch_size=bs,
                                    seed=123)

Xtest = Xval.take(len(Xval) // 2)
Xval = Xval.skip(len(Xval) // 2)
print("Number of samples in train set:", len(Xtrain))
print("Number of samples in validation set:", len(Xval))
print("Number of samples in test set:", len(Xtest))



#Predprocesiranje podataka, data_augmentation
data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_size[0],
                                                           img_size[1], 3)),
        #img_size[0] -> height;
        #img_size[1] -> width;
        #3 -> number of color chanels, if its RGB, this number is 3
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.2, 0.2)
    ]
)

#broj klasa u nasem datasetu
num_classes = len(class_labels)

#Pravljenje arhitekture mreze nanosenjem konvolucionih, pooling i FC slojeva na mrezu
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=l2(regularizer)),
    layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(regularizer))
])

#Pregled arhitekture mreze
model.summary()

#Podesavanje optimizera, kriterijumske funkcije i metrike
model.compile(Adam(learning_rate=lr),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

#Podrska za sprecavanje od preobucavanja
stop_early = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

#Treniranje mreze na trening skupu
history = model.fit(Xtrain,
                    epochs=ep,
                    callbacks=[stop_early],
                    batch_size=64,
                    validation_data=Xval,
                    verbose=2)

#Prikaz preciznosti i kriterijumskih funkcija za trening i za validation skup
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()

#Prikaz tacnosti na test skupu
labelsTest = np.array([])
predTest = np.array([])
for img, lab in Xtest:
    labelsTest = np.append(labelsTest, lab)
    predTest = np.append(predTest, np.argmax(model.predict(img, verbose=0), axis=1))

print('Tacnost modela na test skupu je: ' + str(100*accuracy_score(labelsTest, predTest)) + '%')

#Prikaz matrice konfuzije na test skupu
cm = confusion_matrix(labelsTest, predTest, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(40, 40))
cmDisplay.plot(ax=ax)
plt.show()

#Prikaz tacnosti na trening skupu
labelsTrain = np.array([])
predTrain = np.array([])
for img, lab in Xtrain:
    labelsTrain = np.append(labelsTrain, lab)
    predTrain = np.append(predTrain, np.argmax(model.predict(img, verbose=0), axis=1))

print('Tacnost modela na trening skupu je: ' + str(100*accuracy_score(labelsTrain, predTrain)) + '%')

#Prikaz matrice konfuzije na trening skupu
cm = confusion_matrix(labelsTrain, predTrain, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(40, 40))
cmDisplay.plot(ax=ax)
plt.show()

#Prikaz nekoliko dobro i lose klasifikovanih odbiraka
correctly_classified_indices = np.where(predTest == labelsTest)[0]
incorrectly_classified_indices = np.where(predTest != labelsTest)[0]
print("Indeksi tacno klasifikovanih slika")
print(correctly_classified_indices)
print("Indeksi netacno klasifikovanih slika")
print(incorrectly_classified_indices)

plt.figure(figsize=(6, 6))
plt.suptitle("Prikaz nekoliko korektno klasifikovanih odbiraka")
for img, lab in Xtest.take(1):
    for i, idx in enumerate(correctly_classified_indices[:5]):
        plt.subplot(1, 5, i + 1)
        plt.imshow(img[idx].numpy().astype('uint8'))
        plt.title(class_labels[lab[idx]], fontsize=7.5)
        plt.axis('off')
plt.show()
plt.suptitle("Prikaz nekoliko nekorektno klasifikovanih odbiraka")
for img, lab in Xtest.take(1):
    for i, idx in enumerate(incorrectly_classified_indices[:5]):
        plt.subplot(1, 5, i + 1)
        plt.imshow(img[idx].numpy().astype('uint8'))
        plt.title(class_labels[lab[idx]], fontsize=7.5)
        plt.axis('off')
plt.show()
