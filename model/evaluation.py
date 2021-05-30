import numpy as np
import argparse
from tensorflow.keras.models import load_model
from data_gen import DataGenerator

def load_pretrain_model(model_path = None):
    """Load the pre-trained model.
    Param
    -----
    model_path : str
        path for the pre-trained model. Default using VGG16 (2GB)

    Return
    ------
    model : tf.keras.model
        compiled pre-trained model
    """
    if model_path is None:
        raise ValueError("No model path is given")

    model = load_model(model_path, compile = True)
    print(model.summary())

    return model

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_path", default="../weights/tensorfood.h5")
    ap.add_argument("--test_path", default="../data/train_test_split/val")
    args=vars(ap.parse_args())

    dg = DataGenerator()
    test_datagen_pl = dg.get_test_data_gen(args["test_path"])
    model = load_pretrain_model(args["model_path"])
    test_acc = model.evaluate(test_datagen_pl)
    print("Accuracy Score for test set with {} samples across {} classes is {}.".format(test_datagen_pl.__dict__["samples"], test_datagen_pl.__dict__["num_classes"],test_acc[1]))
