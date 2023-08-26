import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
import tensorflow as tf
import os
import random
from Orange.data.pandas_compat import table_from_frame
import Orange

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
from pandas.api.types import is_numeric_dtype
from sklearn import preprocessing

class ReadLog:
    def __init__(self, eventlog):
        self._eventlog = eventlog
        self._list_cat_cols = []
        self._list_num_cols = []

    def build_w2v(self, prefix_list, mean_trace, name):
        w2v_model = Word2Vec(vector_size=mean_trace, seed=SEED, sg=0, min_count=1, workers=1)
        w2v_model.build_vocab(prefix_list, min_count=1)
        total_examples = w2v_model.corpus_count
        w2v_model.train(prefix_list, total_examples=total_examples, epochs=25)
        w2v_model.save(name + '_w2v_text.h5')

    def get_sequence_cat(self, sequence):
        i = 0
        list_seq = []
        list_label = []
        while i < len(sequence):
            list_temp = []
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])
                list_seq.append(list_temp)
                list_label.append(sequence.iat[i, 0][j + 1])
                j = j + 1
            i = i + 1
        return list_seq, list_label


    def get_sequence(self, sequence, max_trace, mean_trace):
        i = 0
        s = (max_trace)
        list_seq = []
        list_label = []
        while i < len(sequence):
            list_temp = []
            seq = np.zeros(s)
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])
                new_seq = np.append(seq, list_temp)
                cut = len(list_temp)
                new_seq = new_seq[cut:]
                list_seq.append(list(new_seq[-mean_trace:]))
                list_label.append(sequence.iat[i, 0][j + 1])
                j = j + 1
            i = i + 1
        return list_seq, list_label

    @staticmethod
    def get_image_size(num_col):
        import math
        matx = round(math.sqrt(num_col))
        if num_col > (matx * matx):
            matx = matx + 1
            padding = (matx * matx) - num_col
        else:
            padding = (matx * matx) - num_col
        return matx, padding
    @staticmethod
    def dec_to_bin(x):
        return format(int(x), "b")

    @staticmethod
    def pixel_conversion(bin_num):

        if len(bin_num) < 24:
            pad = 24 - len(bin_num)
            zero_pad = "0" * pad
            line = zero_pad + str(bin_num)
            n = 8
            rgb = [line[i:i + n] for i in range(0, len(line), n)]
            int_num = [int(element, 2) for element in rgb]
        else:
            n = 8
            line = str(bin_num)
            rgb = [line[i:i + n] for i in range(0, len(line), n)]
            int_num = [int(element, 2) for element in rgb]
        return int_num

    def text_to_vec_rgb(self, tweets, mean_trace, name):
        word_vec_dict = self.load_dict(name)

        list_tot = []
        for tw in tweets:
            list_grid = []
            for t in tw:
                embed_vector = word_vec_dict.get(t)
                list_temp_pix = []
                if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
                    for e in embed_vector:
                        if e > 1:
                            e = 1
                        elif e < 0:
                            e = 0
                        v = e * (2 ** 24 - 1)
                        bin_num = self.dec_to_bin(v)
                        pixel = self.pixel_conversion(bin_num)
                        list_temp_pix.append(pixel)
                else:
                    embed_vector = np.zeros((mean_trace,))
                    for e in embed_vector:
                        if e > 1:
                            e = 1
                        elif e < 0:
                            e = 0
                        v = e * (2 ** 24 - 1)
                        bin_num = self.dec_to_bin(v)
                        pixel = self.pixel_conversion(bin_num)
                        list_temp_pix.append(pixel)
                rgb = np.array(list_temp_pix)
                list_grid.append(rgb)
            list_tot.append(list_grid)
        list_tot = np.asarray(list_tot)
        return list_tot

    def load_dict(self, name):
        w2v_model = Word2Vec.load(name + '_w2v_text.h5')
        vocab = w2v_model.wv.index_to_key

        word_vec_dict = {}
        for word in vocab:
            word_vec_dict[word] = w2v_model.wv.get_vector(word)

        emb = np.array([list(item) for item in word_vec_dict.values()])

        from sklearn.preprocessing import minmax_scale
        emb_scaled = minmax_scale(emb, feature_range=(0, 1))

        i = 0
        word_vec_dict = {}
        for word in vocab:
            word_vec_dict[word] = emb_scaled[i]
            i = i + 1

        return word_vec_dict

    def make_img(self, X_act, col):
        X_rgb = []
        fake_patch = np.zeros((len(X_act[0]), X_act[0].shape[1], X_act[0].shape[1], 3))
        matx, padding = self.get_image_size(len(col))

        for i in range(len(X_act[0])):
            list_patch = [X_act[j][i] for j in range(len(col))]
            list_patch.extend([fake_patch[i]] * padding)
            grid_tot = []
            for x in range(0, len(list_patch), matx):
                grid_temp = np.hstack(list_patch[x: x + matx])
                grid_tot.append(grid_temp)
            X_rgb.append(np.vstack(grid_tot))
        return np.array(X_rgb)

    def equifreq(self, view_train, view_test, n_bin):
        sort_v = np.append(view_train, view_test)
        df = pd.DataFrame(sort_v)
        df = table_from_frame(df)
        disc = Orange.preprocess.Discretize()
        disc.method = Orange.preprocess.discretize.EqualFreq(n=n_bin)
        df = disc(df)
        df = list(df)
        df = list(map(str, df))
        view_train = df[:len(view_train)]
        view_test = df[len(view_train):]
        return view_train, view_test

    def add_time_column(self, group):
        timestamp_col = 'timestamp'
        group = group.sort_values(timestamp_col, ascending=True)
        # end_date = group[timestamp_col].iloc[-1]
        start_date = group[timestamp_col].iloc[0]

        timesincelastevent = group[timestamp_col].diff()
        timesincelastevent = timesincelastevent.fillna(pd.Timedelta(seconds=0))
        group["timesincelastevent"] = timesincelastevent.apply(
            lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds

        elapsed = group[timestamp_col] - start_date
        elapsed = elapsed.fillna(pd.Timedelta(seconds=0))
        group["timesincecasestart"] = elapsed.apply(lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds

        return group

    def extract_views(self, column, df, max_trace, mean_trace):
        df_train, df_test = self.generate_prefix_trace(df)

        df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
        df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])

        df_train = df_train.groupby('case', group_keys=False).apply(self.add_time_column)
        df_test = df_test.groupby('case', group_keys=False).apply(self.add_time_column)
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        # time to cat
        num_act = list(set(list(df_train['activity'].unique()) + list(df_test['activity'].unique())))
        num_res = list(set(list(df_train['resource'].unique()) + list(df_test['resource'].unique())))
        n_bin = (len(num_act) + len(num_res)) // 2

        list_patch_train = []
        list_patch_test = []

        for col in column:
            if is_numeric_dtype(df_train[col]):
                print('numeric col->', col)
                df_train = df_train.fillna(0)
                df_test = df_test.fillna(0)

                df_train[col], df_test[col] = self.equifreq(df_train[col],df_test[col], n_bin)
                view_train = df_train.groupby('case', sort=False).agg({col: lambda x: list(x)})
                view_test = df_test.groupby('case', sort=False).agg({col: lambda x: list(x)})
                view_train_w2v, label_train_w2v = self.get_sequence_cat(view_train)
                view_train, label_train_n = self.get_sequence(view_train, max_trace, mean_trace)
                view_test, label_test_n = self.get_sequence(view_test, max_trace, mean_trace)
            else:
                print('cat col->', col)
                if col == 'activity':
                        view_train = df_train.groupby('case', sort=False).agg({col: lambda x: list(x)})
                        view_test = df_test.groupby('case', sort=False).agg({col: lambda x: list(x)})
                        view_train_w2v, label_train_w2v = self.get_sequence_cat(view_train)
                        view_train, label_train = self.get_sequence(view_train, max_trace, mean_trace)
                        view_test, label_test = self.get_sequence(view_test, max_trace, mean_trace)


                        with open("fold/" + self._eventlog + "/" + self._eventlog + "_y_train.pk", 'wb') as handle:
                            pickle.dump(label_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        with open("fold/" + self._eventlog + "/" + self._eventlog + "_y_test.pk", 'wb') as handle:
                            pickle.dump(label_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                        df_train = df_train.fillna("UNK")
                        df_test = df_test.fillna("UNK")
                        view_train = df_train.groupby('case', sort=False).agg({col: lambda x: list(x)})
                        view_test = df_test.groupby('case', sort=False).agg({col: lambda x: list(x)})
                        view_train_w2v, l_view_train = self.get_sequence_cat(view_train)
                        view_train, l_view_train = self.get_sequence(view_train, max_trace, mean_trace)
                        view_test, l_view_test = self.get_sequence(view_test, max_trace, mean_trace)

            self.build_w2v(view_train_w2v, mean_trace, col)
            X_train_view = self.text_to_vec_rgb(view_train, mean_trace, col)
            X_test_view = self.text_to_vec_rgb(view_test, mean_trace, col)

            list_patch_train.append(X_train_view)
            list_patch_test.append(X_test_view)

        list_patch_train = np.array(list_patch_train)
        list_patch_test = np.array(list_patch_test)

        X_train = self.make_img(list_patch_train, column)
        X_test = self.make_img(list_patch_test, column)

        df_labels = np.unique(list(label_train) + list(label_test))

        num_classes = len(df_labels)
        input_shape = (mean_trace, mean_trace, 1)
        label_encoder = preprocessing.LabelEncoder()
        integer_encoded = label_encoder.fit_transform(df_labels)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoder.fit(integer_encoded)
        onehot_encoded = onehot_encoder.transform(integer_encoded)

        train_integer_encoded = label_encoder.transform(label_train).reshape(-1, 1)
        train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
        Y_train = np.asarray(train_onehot_encoded)

        test_integer_encoded = label_encoder.transform(label_test).reshape(-1, 1)
        test_onehot_encoded = onehot_encoder.transform(test_integer_encoded)
        Y_test = np.asarray(test_onehot_encoded)
        Y_test_int = np.asarray(test_integer_encoded)

        with open("img/" + self._eventlog + "/" + self._eventlog + 'train.pickle','wb') as handle:
            pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("img/" + self._eventlog + "/" + self._eventlog + 'test.pickle','wb') as handle:
            pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("img/" + self._eventlog + "/" + self._eventlog + 'train_label.pickle','wb') as handle:
            pickle.dump(train_onehot_encoded, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("img/" + self._eventlog + "/" + self._eventlog + 'test_label.pickle','wb') as handle:
            pickle.dump(test_onehot_encoded, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("img/" + self._eventlog + "/" + self._eventlog + '_Y_test_int.pickle','wb') as handle:
            pickle.dump(Y_test_int, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def generate_prefix_trace(self, log):
        grouped = log.groupby("case")
        start_timestamps = grouped["timestamp"].min().reset_index()
        start_timestamps = start_timestamps.sort_values("timestamp", ascending=True, kind="mergesort")
        train_ids = list(start_timestamps["case"])[:int(0.66 * len(start_timestamps))]
        train = log[log["case"].isin(train_ids)].sort_values("timestamp", ascending=True,kind='mergesort')
        test = log[~log["case"].isin(train_ids)].sort_values("timestamp", ascending=True,kind='mergesort')
        return train, test
    def readView(self):
            self._list_cat_cols = []
            self._list_num_cols = []
            df = pd.read_csv("fold/" + self._eventlog + ".csv", sep=',')

            if self._eventlog == 'bpi12w_complete' or self._eventlog == 'bpi12_all_complete' or self._eventlog == 'bpi12_work_all':
                df['resource'] = 'Res' + df['resource'].astype(str)

            col = df.columns.values.tolist()
            col.remove('activity')
            col.remove('case')
            col.remove('timestamp')
            col.insert(0, 'activity')
            col.insert(2,'timesincecasestart')
            cont_trace = df['case'].value_counts(dropna=False)
            max_trace = max(cont_trace)
            mean_trace = int(round(np.mean(cont_trace)))
            self.extract_views(col, df, max_trace, mean_trace)






