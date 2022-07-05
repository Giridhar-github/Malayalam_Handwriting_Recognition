from os import listdir
#import matplotlib.pyplot as plt
import numpy as np
import cv2
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def loadImages(path):
    # return array of images
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
#        print(path+image)
        img = cv2.imread(path + image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img.flatten()
        loadedImages.append(img)

    return loadedImages
path = "data/aa/"
aa_imgs = loadImages(path)
aa_label = np.ones(len(aa_imgs))*0


path = "data/il/"
il_imgs = loadImages(path)
il_label =  np.ones(len(il_imgs))*1

imgs = aa_imgs.copy()
imgs.extend(il_imgs)

path = "data/in/"
in_imgs = loadImages(path)
in_label =  np.ones(len(in_imgs))*2

imgs.extend(in_imgs)

path = "data/ka/"
ka_imgs = loadImages(path)
ka_label = np.ones(len(ka_imgs))*3

imgs.extend(ka_imgs)

path = "data/la/"
la_imgs = loadImages(path)
la_label = np.ones(len(la_imgs))*4

imgs.extend(la_imgs)

path = "data/na/"
na_imgs = loadImages(path)
na_label = np.ones(len(na_imgs))*5

imgs.extend(na_imgs)

path = "data/pa/"
pa_imgs = loadImages(path)
pa_label = np.ones(len(pa_imgs))*6
imgs.extend(pa_imgs)

path = "data/ra/"
ra_imgs = loadImages(path)
ra_label = np.ones(len(ra_imgs))*7
imgs.extend(ra_imgs)

path = "data/ssa/"
ssa_imgs = loadImages(path)
ssa_label = np.ones(len(ssa_imgs))*8

imgs.extend(ssa_imgs)

path = "data/tha/"
tha_imgs = loadImages(path)
tha_label = np.ones(len(tha_imgs))*9
imgs.extend(tha_imgs)

data = np.array(imgs)
data = data.reshape([-1,28,28])
labels = np.hstack((aa_label,il_label,in_label,ka_label,
                    la_label,na_label,pa_label,
					ra_label,ssa_label,tha_label))
labels =np.uint8(labels)

# train
X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=0.33,random_state=42)
#plt.imshow(data[0], cmap=plt.get_cmap('gray'))

# flatten 28*28 images to a 784 vector for each image
X_train = X_train.reshape(X_train.shape[0], 28, 28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28,1).astype('float32')
	
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

print("Training data size",len(X_train))
print("Testing data size",len(X_test))
