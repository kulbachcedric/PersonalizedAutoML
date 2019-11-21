import logging
import os

import numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, \
    jaccard_similarity_score, jaccard_score, matthews_corrcoef, zero_one_loss, hamming_loss, log_loss, hinge_loss, \
    brier_score_loss, roc_auc_score, max_error, mean_squared_error, explained_variance_score

from scorers.linear_scorer import LinearScorer

SCORING_FUNCTIONS = [
    # accuracy_score,\
    # brier_score_loss,\
    # explained_variance_score,\
    # f1_score,\
    # hamming_loss,\
    # hinge_loss, \
    # jaccard_score, \
    # log_loss, \
    # matthews_corrcoef,\
    # max_error,
    # mean_squared_error,
    precision_score,\
    recall_score,\
    # roc_auc_score,\
    # zero_one_loss,\
]

logging.basicConfig(level=logging.DEBUG)

#WEIGHT_SCORERS = numpy.random.dirichlet(numpy.ones(len(SCORING_FUNCTIONS)), size=1).tolist()[0]
#WEIGHT_SCORERS = [1.0,0.0,0.0]
#WEIGHT_SCORERS = [0.4,0.6]
WEIGHT_SCORERS = [1.0]
OPENML_ID = os.getenv("OPENML_ID",179)
#AGENT_SCORER=LinearScorer(scoring_functions=SCORING_FUNCTIONS, scoring_weights=WEIGHT_SCORERS)
AGENT_SCORER=LinearScorer(scoring_functions=[f1_score], scoring_weights=WEIGHT_SCORERS)
MAX_TIME = 60
