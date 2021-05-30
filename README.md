# Singapore Food Prediction

## Introduction
We try to train and deploy a computer vision model that can perform classification on food images on a webapp via docker.
This project aims to expose us to both front end and back end development.

### Architecture

This model is trained using a VGG-16 architecture with weights pretrained on the ImageNet dataset. 
This pre-trained model is then trained with 1224 pictures of 12 types of Singapore local delights.
Some the details regarding the model is shown in the table below:

| Param | Details |
| :----- | :------ | 
| Layers | 24 |
| Trainable params | 370,171,148 |
| Pre-trained on | ImageNet |
| Epoch trained | 28 (early stopped) |
| Batch size | 32 |
| Loss function | Cateogorical Cross Entropy |
| Learning rate | 0.001 |
| Training images used | 851 |
| Testing images used | 373 |
| Image dimension trained on | 200 x 200 x 3 |
| Test accuracy | 80.69% |



## Getting Started

### Prerequisites

To run the app and its associated functions, the following packages are required:

| Pacakge | Purpose | Link |
| :--- | :----| :--- |
| scikit-learn | utilities functions | [scikit-learn][1] |
| TensorFlow 2 | framework for training the model | [TensorFlow 2][2] |
| Pillow | image processing | [Pillow][3]|
| Flask | web app framework | [Flask][4] |
| waitress |  WSGI server | [waitress][5] |
| pytest | unit testing | [pytest][6] |

### Usage

For a quick start, run the following in the Terminal to see how the model perform (need to include model.h5 in production):

```
git clone
python -m src.inference --image_path your_test_image.png --model_path model.h5
```

To run the ``app.py`` locally, run the following in the Terminal and paste the local host to your browser:

```
python -m src.app
```


## Running the tests

Run the following in the Terminal at ``tests`` directory:

```
pytest test_inference.py
```


## Deployment

The model is deployed in a Docker container is hosted on AI Singapore's cluster. 



[1]: https://scikit-learn.org/stable/install.html
[2]: https://www.tensorflow.org/install
[3]: https://pypi.org/project/Pillow/2.2.1/
[4]: https://pypi.org/project/Flask/
[5]: https://pypi.org/project/waitress/
[6]: https://pypi.org/project/pytest/
