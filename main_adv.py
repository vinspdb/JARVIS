from tensorflow.keras.models import load_model
from module_adv.adv_example import ADV_example
import pickle
from vit_keras import vit
import os
import sys

if __name__ == "__main__":
                l  = sys.argv[1]
                try:
                    os.makedirs('img_adv/'+l)
                    print('Log->', l)
                    list_acc = []
                    e = 0.001
                    with open("img/" + l + "/" + l + 'train.pickle', 'rb') as handle:
                        X_a_train = pickle.load(handle)
                    with open("img/" + l + "/" + l + 'train_label.pickle', 'rb') as handle:
                        y_a_train = pickle.load(handle)
                    X_a_train = vit.preprocess_inputs(X_a_train)
                    model = load_model('model/' + l + '.h5')
                    obj = ADV_example(model, e, X_a_train, y_a_train)
                    adv_train, adv_y_train = obj.create_adv_sample()
                    with open("img_adv/" + l + "/" + l + 'train_' + str(e) + '.pickle', 'wb') as handle:
                        pickle.dump(adv_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    with open("img_adv/" + l + "/" + l + 'train_label_' + str(e) + '.pickle', 'wb') as handle:
                        pickle.dump(adv_y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except FileExistsError:
                    pass







