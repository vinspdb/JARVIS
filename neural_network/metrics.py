import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import f1_score
from vit_keras import vit
from imblearn.metrics import geometric_mean_score
import sys

l = sys.argv[1]


model = tf.keras.models.load_model("model_adv/"+l+".h5")

with open("img/" + l + "/" + l + 'test.pickle', 'rb') as handle:
        X_a_test = pickle.load(handle)
with open("img/" + l + "/" + l + 'test_label.pickle', 'rb') as handle:
        y_a_test = pickle.load(handle)

with open("img/" + l + "/" + l + '_Y_test_int.pickle', 'rb') as handle:
        Y_test_int = pickle.load(handle)

X_a_test = vit.preprocess_inputs(X_a_test)

preds_a = model.predict(X_a_test)

y_a_test = np.argmax(y_a_test, axis=1)
preds_a = np.argmax(preds_a, axis=1)

f1_score = f1_score(Y_test_int, preds_a, average='macro')
g_mean = geometric_mean_score(Y_test_int, preds_a, average="macro")

print(f1_score)
print(g_mean)