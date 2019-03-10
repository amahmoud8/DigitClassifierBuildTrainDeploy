import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
from keras.layers import Input, concatenate, Concatenate



class Classifier:
    def __init__(self,debug=False):
        fashion_mnist = keras.datasets.mnist
        # fetch training and testing data
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        # expose incorrect dimensions
        self.checkDim()
        # normalize training and testing
        self.train_images = self.train_images/255.0
        self.test_images = self.test_images/255.0

        # for debugging purpose, print dimensions and view samples
        if debug:
            self.sampleImages()
        # build model
        self.model = self.buildModel(debug=debug)

    def buildModel(self,debug=False):
        # change input shape
        inputData = Input(shape=(28, 28, 1))
        conv1 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputData)
        conv1_pool = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2_1 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(conv1_pool)
        conv2_1_pool = MaxPooling2D(pool_size=(2, 2))(conv2_1)
        conv3_1 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(conv2_1_pool)

        conv2_2 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(conv1_pool)
        conv2_2_pool = MaxPooling2D(pool_size=(2, 2))(conv2_2)
        conv3_2 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(conv2_2_pool)
        # merge two convolutional layers
        merge = concatenate([conv3_1, conv3_2])
        # fully connected layers
        flat = Flatten()(merge)
        FC1000 = Dense(1000, activation='relu')(flat)
        FC500 = Dense(500, activation='relu')(FC1000)
        outputLayer = Dense(10, activation='softmax')(FC500)

        model = Model(inputs=inputData, outputs=outputLayer)

        if debug:
            model.summary()

        return model

    # train using built model
    def train(self,saved_model_name,epochs=5):
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # reshape input before fitting because network expects 4 dimensions
        self.train_images = self.train_images.reshape([-1, 28, 28, 1])

        self.model.fit(self.train_images, self.train_labels, epochs=epochs)

        self.model.save('./'+saved_model_name)

    # evaluate testing set
    def evaulate(self):
        # reshape input before evaluating because network expects 4 dimensions
        self.test_images = self.test_images.reshape([-1, 28, 28, 1])

        # evaluate on test images
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print('Test accuracy:', test_acc)

    # print dimensions and sample images
    def sampleImages(self):
        print("Dimensions of Training set images: ", self.train_images.shape)
        print("Dimensions of Training set labels: ", self.train_labels.shape)
        print("Dimensions of Testing set images: ", self.test_images.shape)
        print("Dimensions of Testing set labels: ", self.test_labels.shape)
        class_names = ['0', '1', '2', '3', '4',
                       '5', '6', '7', '8', '9']

        # view 25 classes
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[self.train_labels[i]])
        plt.show()

    # expose incorrect image dimensions
    def checkDim(self):
        assert (self.train_images.shape[1]== 28), "Dimension Error : Expect Train image width : 28"
        assert (self.train_images.shape[2] == 28), "Dimension Error : Expect Train image height : 28"
        assert (self.test_images.shape[1]== 28), "Dimension Error : Expect Test image width : 28"
        assert (self.test_images.shape[2] == 28), "Dimension Error : Expect Test image height : 28"




if __name__ == '__main__':
    clf = Classifier(debug=True)
    # model_name = 'cnnModel.h5'
    # clf.train(model_name,epochs=1)
    # clf.evaulate()