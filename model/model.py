# pylint: disable=import-error
import logging
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
# pylint: enable=import-error

logger = logging.getLogger(__name__)


class ModelVGG():

    def __init__(self):
        self.vgg16_pre_trained = VGG16(include_top=False, weights='imagenet', input_shape=(200, 200, 3))
        self.vgg16_model = self.vgg16_pre_trained.output

        self.vgg16_model = Flatten()(self.vgg16_model)
        self.vgg16_model = Dense(16384, activation='relu')(self.vgg16_model) 
        self.vgg16_model = Dense(4096, activation='relu')(self.vgg16_model) 
        self.vgg16_model = Dense(256, activation='relu')(self.vgg16_model) 
        self.pred_prob = Dense(12, activation='softmax')(self.vgg16_model)

        self.model = Model(inputs=self.vgg16_pre_trained.input, outputs=self.pred_prob)

        for layer in self.model.layers[:18]:
            layer.trainable = False

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        """ Model summary (layers architecture)
        """
        return self.model.summary()

    def train(self, train_gen, save_path, epoch):
        """Training the model
        Param
        -----
        train_gen : tf `DirectoryIterator` 
            This yielding tuples of `(x, y)` where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.

        params : dict
            A dictionary containing parameters for the model

        epoch : int
            # of epochs

        Returns
        -------
        float
            training accuracy
        """

        es = EarlyStopping(monitor="loss",patience=5,mode="min")
        self.model.fit(x=train_gen, epochs=epoch, callbacks=[es])
        self.model.save(save_path)
        train_acc = self.evaluate(train_gen)
        return train_acc

    def evaluate(self, test_gen):
        """Evaluate the model on train/val/test set
        Param
        -----
        test_gen : tf `DirectoryIterator` 
            This yielding tuples of `(x, y)` where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.

        Returns
        -------
        float
            accuracy for the respective data set
        """
        pred_probability = self.model.predict_generator(test_gen)
        pred_label = np.argmax(pred_probability,axis=1)
        accuracy = accuracy_score(test_gen.classes, pred_label)
        return accuracy



