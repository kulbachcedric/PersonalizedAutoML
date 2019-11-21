from preference_controller.segment import Segment
from preference_controller.segment_pair import SegmentPair


class Judgement(SegmentPair):

    def set_winner(self, winner:str):
        if winner == "seg_1":
            self.__choice = 0.0
        elif winner == "seg_2":
            self.__choice = 1.0
        else:
            raise ValueError("String doesn't match seg_1 or seg_2")

    def get_winner_float(self)->float:
        return self.__choice

    def get_winner(self):
        if self.__choice > 0:
            return self.__seg_2
        elif self.__choice < 0:
            return self.__seg_1
        else:
            raise ValueError('No winner specified.')