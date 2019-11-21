import pickle
import random
from typing import List

from preference_controller.judgement import Judgement
from preference_controller.segment import Segment
from preference_controller.segment_pair import SegmentPair


class PreferenceController:
    def __init__(self):
        self.__segment_pairs = []
        self.__judgements = []

    def generate_segment_pairs(self,segments:List[Segment],amount=None)->List[SegmentPair]:
        random.shuffle(segments)
        for idx in range(amount):
            seg_1,seg_2 = random.choices(segments,k=2)
            self.__segment_pairs.append(SegmentPair(seg_1=seg_1, seg_2=seg_2))
        return self.get_segment_pairs_without_reset()

    def get_segment_pairs_without_reset(self)->List[SegmentPair]:
        return self.__segment_pairs

    def get_segment_paris_with_reset(self)->List[SegmentPair]:
        segment_paris = self.__segment_pairs
        self.__segment_pairs = []
        return segment_paris

    def add_judgements(self,judgements:List[Judgement]):
        self.__judgements.extend(judgements)

    def add_judgment(self,judgment:Judgement):
        self.__judgements.append(judgment)
