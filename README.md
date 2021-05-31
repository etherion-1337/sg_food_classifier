# Bubble the Food Connoisseur

## Introduction

We attempt to train a unicorn :unicorn:  named Bubble to perform food recognition on our Singaporean cuisine !            

Bubble is able to recognise 12 types of food (below) after feeding her. Try it out yourself :yum:               

| <!-- -->    | <!-- -->    |<!-- -->     | <!-- -->    |
|-------------|-------------|-------------|-------------|
| chilli crab | curry puff  | dim sum     | ice kacang  |
| kaya toast  | nasi ayam   | popiah      | roti prata  |
| sambal stingray | satay   | tau huay    | wanton noodle |             


### Architecture

The backend model is trained using a VGG-16 architecture with weights pretrained on the ImageNet dataset. 
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

### Repository Structure

```
.
├── app
│   ├── static                        # static object for HTML template
│   ├── templates
│   │   └── index_xav.html            # HTML script for front end
│   ├── app.py                        # Flask app
│   └── inference.py                  # inference pipeline called by Flask app
├── data
│   ├── train_set                     # pic grouped in respective folder downstream
│   │   ├── chilli_crab               # folder name = label
│   │   │   ├── image_1.jpg
│   │   │   └── ...
│   │   ├── curry_puff
│   │   │   ├── image_101.jpg
│   │   │   └── ...
│   │   └── ...
│   │   
│   └── val_set   
│       ├── chilli_crab
│       │   ├── image_810.jpg
│       │   └── ...
│       ├── curry_puff
│       │   ├── image_835.jpg
│       │   └── ...
│       └── ...
├── model                             
│   ├── data_gen.py                   # data preparation pipeline (for training)
│   ├── model.py                      # define model architecture
│   ├── evaluation.py                 # evaluation script for test set
│   └── train.py                      # training script (polyaxon or local)
├── test
│   ├── pytest.ini                    # config file for pytest
│   └── test_inference.py             # sample test case (WIP, by right should be TDD)
├── weights
│   ├── weights_1.h5
│   └── ...
├── Dockerfile                        # dockerize the app (with weights and lib)
├── environment.yml
├── requirements.txt
└── README.md
```

**Note that due to the file size limitation of Github, `weights` and `data` folders are not included, Contact author for more details** 


## Getting Started

### Prerequisites

To run the app and its associated functions, the following packages are required:                

| Pacakge | Purpose | Link |
| :--- | :----| :--- |
| scikit-learn | utilities functions | [scikit-learn][1] |
| TensorFlow 2 | framework for training the model | [TensorFlow 2][2] |
| Numpy | utilities functions | [Numpy][3] |
| Pillow | image processing | [Pillow][4]|
| Flask | web app framework | [Flask][5] |
| waitress |  WSGI server | [waitress][6] |
| pytest | unit testing | [pytest][7] |                

Two files (`environment.yml` and `requirements.txt`) are provided to if you want to restore the environment.

### Usage (After acquiring `weights` and `data`)               

1. Quick Start             

Navigate to the `app` folder and run the following in the Terminal to see how the model performs:

```
python -m inference --image_path your_test_image.jpg
```
You should be able to obtain the name of the food and the predicated probability:             
```
The food is: ice_kacang, with confidence of 99.92%
```

2. Start the App         

To run the ``app.py`` locally, run the following (while in `app` folder) in the Terminal and paste the local host/port to your browser:

```
python -m app
```

3. Unit Test            

To run the unit test, run the following in the root directory:           
```
pytest test/test_inference.py 
```

4. Training                

The training of the model is performed on the [Polyaxon][7] platform. Therefore the training script `train.py` is written in mind of this concept. To train the model locally on the machine you can first deactivate Polyaxon setting by:             

```
POLYAXON_NO_OP=1
```
or if you are using Mac:             
```
export POLYAXON_NO_OP=1
```
and then run the command below to train the model in the model directory:         
```
python -m train
```

5. Model Evaluation

Run the following command in the `model` directory:               
```
python evaluation.py
```
and you should see the model architecture followed by test metrics:           
```
Accuracy Score for test set with 373 samples across 12 classes is 0.8069705367088318.
```

## Bubble in Action ! :stuck_out_tongue:              

(Long inference time :sleeping:  because I am running on my MacBook so no GPU)

https://user-images.githubusercontent.com/46531622/120099779-f58b2680-c16f-11eb-978a-fc73291e733e.mov                      


## Deployment (WIP)        

Dockerize the App and deploy on EC2 (if got time and resource, free tier t2.nano probably wont make it :money_mouth_face:).



## TODO         
~~1. Re-organise folders~~                  
~~2. Redesign Front end~~                 
~~3. Dockerize the App~~                
4. Deploy on AWS EC2            

## Author

Zk Xav

[1]: https://scikit-learn.org/stable/install.html
[2]: https://www.tensorflow.org/install
[3]: https://numpy.org/install/
[4]: https://pypi.org/project/Pillow/2.2.1/
[5]: https://pypi.org/project/Flask/
[6]: https://pypi.org/project/waitress/
[7]: https://pypi.org/project/pytest/
[8]: https://polyaxon.com/
