import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
from vit_keras import visualize
import sys

eventlog = sys.argv[1]
with open("../img/" + eventlog + "/" + eventlog + 'test.pickle', 'rb') as handle:
    X_a_test = pickle.load(handle)
with open("../img/" + eventlog + "/" + eventlog + '_Y_test_int.pickle', 'rb') as handle:
    y_a_test = pickle.load(handle)


model = load_model('../model_adv/'+eventlog+'.h5')
for i in range(100):# see first 100 images
    attention_map = visualize.attention_map(model=model, image=(X_a_test[i].astype('int')))
    plt.imshow(attention_map)#attention
    plt.imshow(X_a_test[i].astype('int'))#original
    plt.show()