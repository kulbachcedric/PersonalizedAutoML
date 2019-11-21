from typing import List

from sklearn.metrics import make_scorer
from sklearn.preprocessing import normalize

from preference_controller.judgement import Judgement
from scorers.scorer import Scorer

class LinearScorer(Scorer):
    def __init__(self, scoring_functions, judgements=None, scoring_weights =None):
        self.scorers = scoring_functions
        if judgements != None:
            self.__scoring_weights = self.__get_weights(judgements)
        elif scoring_weights !=None:
            self.__scoring_weights = scoring_weights
        else:
            raise ValueError("Precise weights or judgements")

    def __get_weights(self, judgements:List[Judgement]):
        for judgement in judgements:
            seg = judgement.get_winner()
            #todo implement getting weights from judgements

    def get_scorer(self):
        def new_scorer(estimator, X, y):
            res = 0.0
            X = normalize(X=X)
            for val, scorer in enumerate(self.scorers):
                res = res + self.weight_scorers[val] * make_scorer(scorer)(estimator, X, y)
            return float(res)
        return new_scorer

    def score(self, y_ground, y_pred) ->float:
        res = 0.0
        for val, scorer in enumerate(self.scorers):
            res = res + self.__scoring_weights[val] * scorer(y_ground, y_pred)
        return float(res)