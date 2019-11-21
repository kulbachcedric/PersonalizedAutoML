from pathlib import Path
from typing import List

import numpy
from deap import creator
from sklearn.datasets import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from tpot import TPOTClassifier

from preference_controller.segment import Segment

class AutomlInstance:
    def __init__(self, openML_id, scoring_function, memory_path = None, max_time=None):
        self.y_class_dict = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_dataset(openML_id)
        if memory_path != None:
            if Path(memory_path).is_file():
                self.tpot = TPOTClassifier(memory=memory_path,warm_start=True,scoring=scoring_function,verbosity=3)
            else:
                self.tpot = TPOTClassifier(memory=memory_path,max_time_mins=max_time, scoring=scoring_function,verbosity=3)
        else:
            self.tpot = TPOTClassifier(max_time_mins=max_time, scoring=scoring_function,verbosity=3)
        self.tpot.fit(self.X_train,self.y_train)

    def predict(self, X):
        return self.tpot.predict(X)

    def get_segments(self)->List[Segment]:
        segments = []
        for model in self.tpot.evaluated_individuals_:
            try:
                classifier = self.tpot._toolbox.compile(creator.Individual.from_string(model, self.tpot._pset))
                classifier.fit(self.X_train,self.y_train)
                y_pred = classifier.predict(self.X_test)
                segments.append(Segment(y_ground=self.y_test,y_pred=y_pred))
            except ValueError:
                print("One classifier could not be evaluated.")
            except RuntimeError:
                print("One classifier could not be evaluated.")
        return segments

    def get_dataset(self, openMl_id, test_size=0.2):
        X, y = openml.fetch_openml(data_id=openMl_id, return_X_y=True)
        self.dataset_categories = openml.fetch_openml(data_id=31).categories
        openml_data = openml.fetch_openml(data_id=openMl_id, return_X_y=False)
        self.feature_names_X = openml_data.feature_names
        imp = Imputer()
        self.target_categories = numpy.unique(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        x_imp = imp.fit(X_train)
        X_train = x_imp.transform(X_train)
        x_imp = imp.fit(X_test)
        X_test = x_imp.transform(X_test)
        y_train = self._y_string_2_int(y_train)
        y_test = self._y_string_2_int(y_test)
        return X_train, X_test, y_train, y_test

    def _y_string_2_int(self, y: numpy.ndarray):
        if self.y_class_dict == None:
            self._create_class_dict(y)
        transdict = {y:x for x,y in self.y_class_dict.items()}
        return numpy.array([transdict[val] for val in y])

    def _create_class_dict(self, y:numpy.ndarray):
        res = {}
        unique_values = numpy.unique(y)
        counter = 0
        for x in unique_values.tolist():
            res[counter] = x
            counter = counter +1
        self.y_class_dict = res
