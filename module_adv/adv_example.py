from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod
import numpy as np
import tensorflow as tf
import random

seed = 42
tf.random.set_seed(seed)
# Set the random seed for NumPy
np.random.seed(seed)
# Set the random seed for Python's built-in random module
random.seed(seed)

tf.keras.utils.set_random_seed(seed)

class ADV_example:
    def __init__(self, model, eps, x_train, y_train):
        self._model = model
        self._eps = eps
        self._x_train = x_train
        self._y_train = y_train

    def create_adv_sample(self):
        classifier = TensorFlowV2Classifier(self._model, input_shape=(self._x_train.shape[1],self._x_train.shape[2],3), nb_classes=len(self._y_train[0]), loss_object=tf.keras.losses.CategoricalCrossentropy())
        attack = FastGradientMethod(estimator=classifier, eps=self._eps)
        adversarial_samples = attack.generate(x=self._x_train)
        return adversarial_samples, self._y_train
