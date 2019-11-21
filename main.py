import pickle

from sklearn.metrics import f1_score, make_scorer

from agent.agent import Agent
from automl.automl import AutomlInstance
from config import SCORING_FUNCTIONS, OPENML_ID, AGENT_SCORER, MAX_TIME
from preference_controller.preference_controller import PreferenceController
from scorers.linear_scorer import LinearScorer
from scorers.ranknet_scorer import RanknetScorer
import logging

if __name__ == '__main__':
    automl_instances = []
    segments = []
    logging.info("=== OPENML ID "+str(OPENML_ID)+" ===")
    for scorer in SCORING_FUNCTIONS:
        print("Starting with "+scorer.__name__)
        tpot = AutomlInstance(openML_id=OPENML_ID, scoring_function=scorer, max_time=MAX_TIME)
        automl_instances.append(tpot)
        segs = tpot.get_segments()
        segments.extend(segs)
        pickle.dump(segs, open("./data/segments_"+str(OPENML_ID)+"_"+scorer.__name__+".p", "wb+"))
    preference_controller = PreferenceController()
    preference_controller.generate_segment_pairs(segments=segments,amount=1000)
    segment_pairs = preference_controller.get_segment_pairs_without_reset()
    agent = Agent(scorer=AGENT_SCORER)
    judgements = agent.get_judgements(segment_pairs=segment_pairs)

    new_scorer = RanknetScorer(judgements=judgements, scoring_functions=SCORING_FUNCTIONS)

    agent_scorer = make_scorer(f1_score)

    tpot_agent_scorer = AutomlInstance(openML_id=OPENML_ID, scoring_function=agent_scorer, max_time=MAX_TIME)
    tpot_ranknet_scorer = AutomlInstance(openML_id=OPENML_ID, scoring_function=new_scorer.get_scorer(), max_time=MAX_TIME)

    y_pred_agent_scorer = tpot_agent_scorer.predict(tpot_agent_scorer.X_train)
    y_pred_ranknet_scorer = tpot_ranknet_scorer.predict(tpot_ranknet_scorer.X_train)
    y_ground = tpot_ranknet_scorer.y_train

    logging.info("=============")
    logging.info("")
    logging.info("Agent Score: " + str(AGENT_SCORER.score(y_ground, y_pred_agent_scorer)))
    logging.info("Ranknet Score: " + str(AGENT_SCORER.score(y_ground, y_pred_ranknet_scorer)))

