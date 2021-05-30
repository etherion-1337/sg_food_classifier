# pylint: disable=import-error
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# pylint: disable=import-error

# from train_generator.class_indices
model_label_dict = {
    0 : 'chilli_crab',
    1 : 'curry_puff',
    2 : 'dim_sum', 
    3 : 'ice_kacang',
    4 : 'kaya_toast',
    5 : 'nasi_ayam',
    6 : 'popiah', 
    7 : 'roti_prata',
    8 : 'sambal_stingray',
    9 : 'satay', 
    10 : 'tau_huay',
    11 : 'wanton_noodle'
}


def model_predict(model_path, img_path):
    """Predict the food given an image using pre-trained model.
    Param
    -----
    model_path : str
        path for the pre-trained model. Default using VGG16 (2GB)
    img_path : str
        path for 1 image

    Return
    ------
    pred_label : str
        one of the 12 types of food in model_label_dict
    """
    model = load_pretrain_model(model_path)
    image = load_image(img_path)
    preprocess_image = prep_image(image)
    predict_prob = predict_probability(preprocess_image, model)
    confidence = str(round(predict_prob.max()*100,2)) + "%"
    pred_label = predict_label(predict_prob)
    # print("The food is: {}, with confidence of {}".format(pred_label, confidence))

    return pred_label, confidence

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
    # print(model.summary())

    return model

def load_image(img_path = None):
    """Load one image.
    Param
    -----
    img_path : str
        path for 1 image to be predicted. expect image size (200, 200, 3), if not resize to (200, 200, 3)

    Return
    ------
    img : PIL Image
        resized image (200 ,200 ,3)
    """

    if img_path is None:
        raise ValueError("No image path is given")

    IMG_WIDTH, IMG_HEIGHT = 200, 200
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))

    return img

def prep_image(img):
    """Prepare image for predication.
    Param
    -----
    img : PIL Image
        expect image size (200, 200, 3)

    Return
    ------
    image_prep : np.array
        array of the image
    """
    image_arr = image.img_to_array(img)
    image_arr = np.expand_dims(image_arr, axis=0)
    image_prep = np.vstack([image_arr])

    return image_prep

def predict_probability(prep_image, model):
    """Predict softmaxed probability of the image (12 classes)
    Param
    -----
    prep_image : np.array
        expect prep image from raw image with size (200, 200, 3)
    model : tf.keras.model
        pre-trained model (loaded and compiled)

    Return
    ------
    pred_proba : np.array
        shape (1,12) array each contains probability of the class prediction
    """
    pred_proba = model.predict(prep_image)

    return pred_proba

def predict_label(pred_proba):
    """Predict the name of the food
    Param
    -----
    pred_prob : np.array
        shape (1,12) array each contains probability of the class prediction

    Return
    ------
    prediction : str
        name of the food (out of 12 classes)
    """
    pred_label = np.argmax(pred_proba, axis=1)
    prediction = model_label_dict[pred_label[0]]

    return prediction


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--image_path', default='../data/train_test_split/val/chilli_crab/2-jumbo.jpg')
    ap.add_argument('--model_path', default='../weights/tensorfood_small.h5')
    args=vars(ap.parse_args())

    pred_label, confidence = model_predict(args["model_path"], args["image_path"])
    print(pred_label)
    print(confidence)
