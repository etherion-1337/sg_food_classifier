# pylint: disable=import-error
import logging
from polyaxon_client.tracking import Experiment, get_log_level
from model import ModelVGG
from data_gen import DataGenerator
import argparse
import numpy as np
# pylint: enable=import-error


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("polyaxon")


def run_experiment(train_path, test_path, remote):
    experiment = Experiment()
    try:
        logger.info("Starting experiment")

        experiment.set_name('vgg_16_batch32_full_new')

        DG = DataGenerator()

        # Load your data from your datapipeline
        train_datagen_pl = DG.get_train_data_gen(train_path)

        test_datagen_pl = DG.get_test_data_gen(test_path)

        # experiment.log_data_ref(data=(X_train, y_train), data_name="unique_identifier_for_data")

        # log whatever params your model is using
        # experiment.log_params()

        model_vgg = ModelVGG()
        if remote:
            save_path = experiment.get_outputs_path() + '/tensorfood.h5'
        else:
            save_path = "../weights/"
        train_acc = model_vgg.train(train_datagen_pl, save_path, epoch=100)
        test_acc = model_vgg.evaluate(test_datagen_pl)

        print("Train accuracy: %s" %train_acc)
        print("Test accuracy: %s" %test_acc)


        # Log metrics to polyaxon
        logger.info(train_acc)
        logger.info(test_acc)

        experiment.log_metrics(train_acc=train_acc)
        experiment.log_metrics(test_acc=test_acc)



        experiment.succeeded()
        logger.info("Experiment completed")
        return train_acc, test_acc

    
    except Exception as e:
        experiment.failed()
        logger.error(f"Experiment failed: {str(e)}")


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("-train","--train_path", default="../data/train_test_split/train")
    ap.add_argument("-test","--test_path", default="../data/train_test_split/val")
    args=vars(ap.parse_args())

    run_experiment(args["train_path"], args["test_path"], remote = False)

