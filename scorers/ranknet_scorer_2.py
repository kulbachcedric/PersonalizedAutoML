from datetime import datetime
from typing import List

import keras
import numpy as np
from keras import backend
from keras.layers import Activation, Dense, Input, Subtract
from keras.models import Model

from preference_controller.judgement import Judgement
from scorers.scorer import Scorer


class RanknetScorer2(Scorer):

    def __init__(self, judgements, scoring_functions, test_judgements:List[Judgement]=None):
        self.__scorers = scoring_functions
        self.__BATCH_SIZE = 10
        self.__NUM_EPOCHS = 1000

        data = self._get_dataset(judgements=judgements)

        INPUT_DIM = np.array(data[0][0]).shape[0]

        h_1 = Dense(128, activation="relu")
        h_2 = Dense(64, activation="relu")
        h_3 = Dense(32, activation="relu")
        s = Dense(1, activation="sigmoid")

        # Relevant document score.
        rel_doc = Input(shape=(INPUT_DIM,), dtype="float32")
        h_1_rel = h_1(rel_doc)
        h_2_rel = h_2(h_1_rel)
        h_3_rel = h_3(h_2_rel)
        rel_score = s(h_3_rel)

        # Irrelevant document score.
        irr_doc = Input(shape=(INPUT_DIM,), dtype="float32")
        h_1_irr = h_1(irr_doc)
        h_2_irr = h_2(h_1_irr)
        h_3_irr = h_3(h_2_irr)
        irr_score = s(h_3_irr)

        # Subtract scores.
        diff = Subtract()([rel_score, irr_score])

        # Pass difference through sigmoid function.
        prob = Activation("sigmoid")(diff)

        # Build model.
        model = Model(inputs=[rel_doc, irr_doc], outputs=prob)
        model.compile(optimizer="adadelta", loss="binary_crossentropy")

        # Generate Data
        X_1, X_2, y = self._get_X_y(data=data)
        test_data = self._get_dataset(judgements=test_judgements)
        X_1_test, X_2_test, y_test = self._get_X_y(data=test_data)

        es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=0,
                                           patience=2,
                                           verbose=1, mode='auto')
        logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        es2 = keras.callbacks.TensorBoard(log_dir=logdir)

        #Train Model
        history = model.fit([X_1, X_2], y, batch_size=self.__BATCH_SIZE,validation_data=([X_1_test, X_2_test], y_test), epochs=self.__NUM_EPOCHS, verbose=1, callbacks=[es, es2])
        self.get_score = backend.function([rel_doc], [rel_score])
        self.get_score([X_1])
        self.get_score([X_2])
        self.__history = history
        self.__auc_score = (np.sum(self.get_score([X_1_test])[0] > self.get_score([X_2_test])[0]) + 0.0) / \
                           X_1_test.shape[0]


    def get_history(self):
        return self.__history

    def get_auc(self):
        return self.__auc_score

    def get_scorer(self):
        def ensemble_metrics(cls, X, y):
            res = 0
            scores = []
            for scorer in self.__scorers:
                y_pred = cls.predict(X)
                scores.append(scorer(y, y_pred))
            res = self.base_network.predict(np.array([scores]))
            # print(str(res))
            res = res
            __format__ = res
            return float(res)

        return ensemble_metrics

    def score(self, y_ground, y_pred) -> float:
        scores = []
        for scorer in self.__scorers:
            scores.append(scorer(y_ground, y_pred))
        res = self.get_score([[scores]])
        return float(res[0])




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

        return X_1, X_2, y