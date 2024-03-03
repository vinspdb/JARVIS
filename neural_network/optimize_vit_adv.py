import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from time import perf_counter
import time
from tensorflow.keras.optimizers import Adam
import argparse
import pickle
from vit_keras import vit
import math
import vit_net
import random
import os
import pandas as pd

os.environ['PYTHONHASHSEED'] = '0'
seed = 42
tf.random.set_seed(seed)
# Set the random seed for NumPy
np.random.seed(seed)
# Set the random seed for Python's built-in random module
random.seed(seed)

tf.keras.utils.set_random_seed(seed)

space = {'num_layers': hp.choice('num_layers', [2, 3, 4]),
         'hidden_dim': hp.choice('hidden_dim', [4, 5, 6]),
         'mlp_dim': hp.choice('mlp_dim', [4, 5, 6]),
         'batch_size': hp.choice('batch_size', [6, 7, 8]),
         'learning_rate': hp.loguniform("learning_rate", np.log(0.001), np.log(0.01)),
         }

def get_image_size(num_col):
    import math
    matx = round(math.sqrt(num_col))
    if num_col>(matx*matx):
        matx = matx + 1
        padding = (matx*matx) - num_col
    else:
        padding = (matx*matx) - num_col
    return matx, padding

def fit_and_score(params):
    print(params)
    start_time = perf_counter()

    config = {}
    config["num_layers"] = params['num_layers']
    config["hidden_dim"] = params['num_layers']* (2**params['hidden_dim'])
    config["mlp_dim"] = 2 ** params['mlp_dim']
    config["num_heads"] = params['num_layers']
    config["dropout_rate"] = 0.1

    config["image_size"] = X_a_train.shape[1]

    column_log = pd.read_csv('fold/' + dataset + '.csv').shape[1] - 1
    matx, padding = get_image_size(column_log)
    config["patch_size"] = padding + column_log

    print('patch size--->', config["patch_size"])

    config["num_patches"] = int(config["image_size"] ** 2 / config["patch_size"] ** 2)
    config["num_channels"] = 3
    config["num_classes"] = len(y_a_train[0])

    model = vit_net.VisualTransformers(config)
    opt = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    h = model.fit(X_a_train, y_a_train, epochs=100, verbose=0, validation_split=0.2, callbacks=[early_stopping], batch_size=2 ** params['batch_size'])

    scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
    score = min(scores)
    print(score)

    global best_score, best_model, best_time, best_numparameters
    end_time = perf_counter()

    if best_score > score:
        best_score = score
        best_model = model
        best_numparameters = model.count_params()
        best_time = end_time - start_time
        best_model.save('model_adv/' + dataset + '.h5')

    return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(h.history['loss']), 'n_params': model.count_params(),
            'time': end_time - start_time}


if __name__ == "__main__":
            parser = argparse.ArgumentParser(description='Inception for next activity prediction.')
            parser.add_argument('-event_log', type=str, help="Event log name")
            args = parser.parse_args()
            dataset = args.event_log
            output_file = dataset
            e = 0.001
            current_time = time.strftime("%d.%m.%y-%H.%M", time.localtime())
            outfile = open(output_file+'_adv.log', 'w')

            outfile.write("Starting time: %s\n" % current_time)

            with open("img/" + dataset + "/" + dataset + 'train.pickle', 'rb') as handle:
                X_a_train = pickle.load(handle)
            with open("img/" + dataset + "/" + dataset + 'train_label.pickle', 'rb') as handle:
                y_a_train = pickle.load(handle)

            X_a_train = vit.preprocess_inputs(X_a_train)

            with open("img_adv/" + dataset + "/" + dataset + 'train_' + str(e) + '.pickle', 'rb') as handle:
                adversarial_samples = pickle.load(handle)

            with open("img_adv/" + dataset + "/" + dataset + 'train_label_' + str(e) + '.pickle', 'rb') as handle:
                y_adv = pickle.load(handle)

            X_a_train = np.append(X_a_train, adversarial_samples, axis=0)
            y_a_train = np.append(y_a_train, y_adv, axis=0)

            # model selection
            print('Starting model selection...')
            best_score = np.inf
            best_model = None
            best_time = 0
            best_numparameters = 0

            trials = Trials()
            best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=10, trials=trials,
                        rstate=np.random.RandomState(seed))
            best_params = hyperopt.space_eval(space, best)

            outfile.write("\nHyperopt trials")
            outfile.write("\ntid,loss,learning_rate,batch_size,time,n_epochs,n_params,perf_time,num_layers,hidden_dim,mlp_dim")
            for trial in trials.trials:
                outfile.write("\n%d,%f,%f,%d,%s,%d,%d,%f,%d,%d,%f" % (trial['tid'],
                                                                trial['result']['loss'],
                                                                trial['misc']['vals']['learning_rate'][0],
                                                                trial['misc']['vals']['batch_size'][0] + 6,
                                                                (trial['refresh_time'] - trial['book_time']).total_seconds(),
                                                                trial['result']['n_epochs'],
                                                                trial['result']['n_params'],
                                                                trial['result']['time'],
                                                                trial['misc']['vals']['num_layers'][0]+2,
                                                                trial['misc']['vals']['hidden_dim'][0]+4,
                                                                trial['misc']['vals']['mlp_dim'][0]+4,
                                                                ))

            outfile.write("\n\nBest parameters:")
            print(best_params, file=outfile)
            outfile.write("\nModel parameters: %d" % best_numparameters)
            outfile.write('\nBest Time taken: %f' % best_time)
            best_model.save('model_adv/'+dataset+'.h5')

            outfile.flush()

            outfile.close()
