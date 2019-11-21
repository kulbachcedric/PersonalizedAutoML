from typing import List

from preference_controller.judgement import Judgement
from preference_controller.segment_pair import SegmentPair
from scorers.scorer import Scorer


class Agent:
    def __init__(self, scorer:Scorer):
        self.__scorer = scorer

    def score(self,y_ground, y_pred)->float:
        return self.__scorer.score(y_ground, y_pred)

    def get_scoring_function(self):
        return self.__scorer.get_scorer()

    def get_judgements(self,segment_pairs:List[SegmentPair])->List[Judgement]:
        judgements = []
        for segment_pair in segment_pairs:
            (seg_1,seg_2) = segment_pair.get_segments()
            judgement = Judgement(seg_1=seg_1, seg_2=seg_2)
            score_seg_1 = self.score(y_ground=seg_1.get_y_ground(),y_pred=seg_1.get_y_pred())
            score_seg_2 = self.score(y_ground=seg_2.get_y_ground(),y_pred=seg_2.get_y_pred())
            if score_seg_1 > score_seg_2:
                judgement.set_winner("seg_1")
            else:
                judgement.set_winner("seg_2")
            judgements.append(judgement)
        return judgements