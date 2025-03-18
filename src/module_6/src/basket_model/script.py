import random
import os
import pandas as pd

from services.basket_model import BasketModel
from services.feature_store import FeatureStore


STORAGE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..", "data")
)

def main():
    """orders = pd.read_parquet(os.path.join(STORAGE, "orders.parquet"))
    random_user_id = random.choice(orders['user_id'].tolist())
    print(random_user_id)"""
    feature_store = FeatureStore()
    basket_model = BasketModel()

    features = feature_store.get_features("89a261acaa26b722b3874fc6d960492136bc2f0d9494f31177ea491f5b41ab121374eba89cc270cb07a99d31ba60a83ad675fcec28fe4b812d7e56f42a60f4a1")
    predictions = basket_model.predict(features.to_numpy())

    print(predictions)


if __name__ == "__main__":
    main()