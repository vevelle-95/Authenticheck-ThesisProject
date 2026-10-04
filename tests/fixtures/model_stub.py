def predict(reviews):
    return [{"id": review["id"], "sentiment": "neutral"} for review in reviews]
