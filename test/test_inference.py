import sys
import pytest
sys.path.append("..")
from app.inference import load_pretrain_model, load_image, prep_image, predict_probability


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




