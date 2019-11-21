class Segment:
    def __init__(self,y_ground,y_pred):
        self.__y_ground = y_ground
        self.__y_pred = y_pred
    def get_y_ground(self):
        return self.__y_ground
    def get_y_pred(self):
        return self.__y_pred
