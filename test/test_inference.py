import sys
import pytest
sys.path.append("..")
from app.inference import load_pretrain_model, load_image, prep_image, predict_probability, model_predict

MODEL_PATH = "../weights/tensorfood.h5"
IMG_PATH = "./mystery_1.jpg"

@pytest.fixture()
def predict_test_image():
    pred_label, confidence = model_predict(MODEL_PATH, IMG_PATH)
    return pred_label, confidence

class TestModel():

    def test_no_model_path(self):
        with pytest.raises(ValueError) as exc_info:
            load_pretrain_model()
        error_msg = "No model path is given"
        assert exc_info.match(error_msg), "Nothing to do with empty path, something else is wrong"

class TestImage():

    def test_no_image_path(self):
        with pytest.raises(ValueError) as exc_info:
            load_image()
        error_msg = "No image path is given"
        assert exc_info.match(error_msg), "Nothing to do with empty path, something else is wrong"

class TestPredict():

    def test_ice_kacang_label(self, predict_test_image):
        pred_label, _ = predict_test_image
        expected = "ice_kacang"
        assert pred_label == expected, "Expected prediction is {}, actual is {}.".format(expected, pred_label)

    def test_ice_kacang_prob(self, predict_test_image):
        _, predicted_prob = predict_test_image
        threshold_prob = 70.0
        assert float(predicted_prob[:-1]) >= threshold_prob, "Prediction threshold is {}, actual is {}.".format(threshold_prob, predicted_prob)