import traceback
from fertilizer_hybrid import FertilizerRecommender

try:
    fr = FertilizerRecommender(input_dim=7)
    fr.train_recommender()
except Exception as e:
    print("FERT ERROR:")
    traceback.print_exc()
