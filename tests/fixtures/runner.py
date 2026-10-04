import json
import sys

reviews = json.load(sys.stdin)["reviews"]
print(json.dumps({"reviews": [{"id": row["id"], "sentiment": "neutral"} for row in reviews]}))
