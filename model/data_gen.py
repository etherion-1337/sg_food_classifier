# pylint: disable=import-error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# pylint: enable=import-error

class DataGenerator():
    def __init__(self):
        pass

    def get_train_data_gen(self, train_data_path):
        """Build training data generator from file path
        Param
        -----
        train_data_path : str
            path for training data file path

        Returns
        -------
        tf `DirectoryIterator`
            yielding tuples of `(x, y)`
            where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.
        """
        train_datagen = ImageDataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True)
        train_generator = train_datagen.flow_from_directory(
                            train_data_path,
                            target_size=(200, 200),
                            color_mode='rgb',
                            batch_size=32,
                            class_mode='categorical',
                            shuffle = True,
                            seed = 15)

        return train_generator

    def get_test_data_gen(self, test_data_path):
        """Build test data generator from file path
        Param
        -----
        test_data_path : str
            path for test data file path

        Returns
        -------
        tf `DirectoryIterator`
            yielding tuples of `(x, y)`
            where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.
        """
        test_datagen = ImageDataGenerator()
        test_generator = test_datagen.flow_from_directory(
                            test_data_path,
                            target_size=(200, 200),
                            color_mode='rgb',
                            batch_size=32,
                            class_mode='categorical',
                            shuffle = False)

        return test_generator

