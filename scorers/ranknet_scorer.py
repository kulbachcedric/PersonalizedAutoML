from datetime import datetime
import pickle
from typing import List

import keras
from keras import Sequential, Input, Model
from keras.layers import Dense, Subtract, Activation, Dropout
from keras.utils import plot_model
from sklearn.preprocessing import normalize

from preference_controller.judgement import Judgement
from scorers.scorer import Scorer
import numpy as np

import matplotlib.pyplot as plt

class RanknetScorer(Scorer):
    def __init__(self, judgements:List[Judgement],scoring_functions, test_judgements:List[Judgement]=None):
        self.__scorers = scoring_functions
        self.__BATCH_SIZE = 10
        self.__NUM_EPOCHS = 100
        def _create_base_network(input_dim):
            '''Base network to be shared (eq. to feature extraction).
            '''
            seq = Sequential()
            # seq.add(Dense(input_dim, input_shape=(input_dim,)))
            # seq.add(Dropout(0.1))
            # seq.add(Dense(128, activation='relu'))
            # seq.add(Dropout(0.1))
            # seq.add(Dense(64, activation='relu'))
            # seq.add(Dropout(0.1))
            seq.add(Dense(16))
            seq.add(Dense(1))
            return seq

        def _create_meta_network(input_dim, base_network):
            input_a = Input(shape=(input_dim,))
            input_b = Input(shape=(input_dim,))

            rel_score = base_network(input_a)
            irr_score = base_network(input_b)

            # subtract scores
            diff = Subtract()([rel_score, irr_score])

            # Pass difference through sigmoid function.
            prob = Activation("sigmoid")(diff)

            # Build model.
            model = Model(inputs=[input_a, input_b], outputs=prob)
            model.compile(optimizer="adadelta", loss="binary_crossentropy") #loss="binary_crossentropy"

            return model

        data = self._get_dataset(judgements=judgements)

        INPUT_DIM = np.array(data[0][0]).shape[0]

        base_network = _create_base_network(INPUT_DIM)
        model = _create_meta_network(INPUT_DIM, base_network)
        model.summary()
        X_1_train, X_2_train, y_train = self._get_X_y(data=data)



        es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=0,
                                           patience=2,
                                           verbose=1, mode='auto')
        logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        es2 = keras.callbacks.TensorBoard(log_dir=logdir)
        if test_judgements is not None:
            test_data = self._get_dataset(judgements=test_judgements)
            X_1_test, X_2_test, y_test = self._get_X_y(data=test_data)
            history = model.fit([X_1_train, X_2_train], y_train,
                            validation_data=([X_1_test, X_2_test], y_test),
                            batch_size=self.__BATCH_SIZE, epochs=self.__NUM_EPOCHS, verbose=3, callbacks=[es,es2])
            self.__history = history
            self.__auc_score = (np.sum(base_network.predict(X_1_test) > base_network.predict(X_2_test)) + 0.0) / \
                               X_1_test.shape[0]
        else:
            history = model.fit([X_1_train, X_2_train], y_train,
                                #validation_data=([X_1_test, X_2_test], y_test),
                                batch_size=self.__BATCH_SIZE, epochs=self.__NUM_EPOCHS, verbose=3, callbacks=[es, es2])

        if False:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right')
            plt.show()
        self.base_network = base_network
        self.meta_network = model
        #a = np.mean(base_network.predict(X_1_test)), np.mean(base_network.predict(X_2_test))
        #b = (np.sum(base_network.predict(X_1_test) > base_network.predict(X_2_test)) + 0.0) / X_1_test.shape[0]
    def get_history(self):
        return self.__history

    def _get_dataset(self, judgements:List[Judgement]):
        data = []
        for judgement in judgements:
            (seg_1, seg_2) = judgement.get_segments()
            scores_1 = []
            scores_2 = []
            for scorer in self.__scorers:
                scores_1.append(scorer(seg_1.get_y_ground(), seg_1.get_y_pred()))
                scores_2.append(scorer(seg_2.get_y_ground(), seg_2.get_y_pred()))
            choice = judgement.get_winner_float()
            data.append((scores_1, scores_2, choice))
        return data

    def _get_X_y(self, data):
        X_1 = []
        X_2 = []

        for seg1, seg2, r in data:
            if r is None:
                continue
            if float(r) == 1.:
                X_1 = X_1 + [seg2]
                X_2 = X_2 + [seg1]
            elif float(r) == 0.:
                X_1 = X_1 + [seg1]
                X_2 = X_2 + [seg2]

        X_1 = np.array(X_1)
        X_2 = np.array(X_2)
        y = np.ones((X_1.shape[0], 1))
        X_1 = normalize(X=X_1)
        X_2 = normalize(X=X_2)
        return X_1, X_2, y

    def get_auc(self):
        return self.__auc_score

    def get_scorer(self):
        def ensemble_metrics(estimator, X, y):
            res = 0
            scores = []
            for scorer in self.__scorers:
                y_pred = estimator.predict(X)
                scores.append(scorer(y,y_pred))
            res = self.base_network.predict(np.array([scores]))
            #print(str(res))
            res = res
            __format__ = res
            return float(res)
        return ensemble_metrics

    def score(self, y_ground, y_pred) -> float:
        scores = []
        for scorer in self.__scorers:
            scores.append(scorer(y_ground,y_pred))
        res = self.base_network.predict(np.array([scores]))
        return float(res)