# pylint: disable=import-error
import logging
from flask import Flask, render_template, request, jsonify
from waitress import serve
import os

try:
    from .inference import model_predict
except:
    from inference import model_predict

# pylint: disable=import-error

# logging setting
logging.basicConfig(filename="tf_food_app.log",
                    filemode='a', # append not overwrite
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# logger for steaming stderr
stderrLogger=logging.StreamHandler()
stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))

# save log and also display stdderr on stream
logger = logging.getLogger("tf_food")
logging.getLogger().addHandler(stderrLogger)



# Instantiate app
app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, '../weights/tensorfood_small.h5')


@app.route('/', methods=['GET'])
def index():
    return render_template('index_xav.html')

@app.route('/info', methods=['GET'])
def short_description():
    return jsonify({
        "model":"vgg_16",
        "input-size":"200x200x3",
        "num_classes":12,
        "pretrained-on":"ImageNet"
    })


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    logger.info("Predication initiated")
    img_file = request.files['file']
    logger.info("Gotten image file")
    label, conf = model_predict(MODEL_PATH, img_file)
    logger.info("Predication done")
    return jsonify({
        "food": label,
        "probability": conf
    })



if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=True, port=8000)
    # For production mode, comment the line above and uncomment below
    serve(app, host="0.0.0.0", port=8000)
