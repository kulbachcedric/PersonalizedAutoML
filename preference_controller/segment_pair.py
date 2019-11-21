from preference_controller.segment import Segment


class SegmentPair:
    def __init__(self, seg_1:Segment,seg_2:Segment):
        self.__seg_1 = seg_1
        self.__seg_2 = seg_2

    def get_segments(self)->(Segment,Segment):
        return (self.__seg_1,self.__seg_2)


