"""Run a configured Lu et al. callable with the baseline's Python interpreter.

LU_ET_AL_ENTRYPOINT names module:function. The callable accepts the request's
review list and returns either a review-result list or {"reviews": [...]}.
"""

import importlib
import json
import os
from pathlib import Path
import sys


def main():
    root = Path(os.environ["LU_ET_AL_REPO"]).resolve()
    module_name, separator, function_name = os.environ["LU_ET_AL_ENTRYPOINT"].partition(":")
    if not separator or not module_name or not function_name or not function_name.isidentifier():
        raise ValueError("LU_ET_AL_ENTRYPOINT must be module:function")
    sys.path.insert(0, str(root))
    function = getattr(importlib.import_module(module_name), function_name)
    reviews = json.load(sys.stdin)["reviews"]
    result = function(reviews)
    if isinstance(result, list):
        result = {"reviews": result}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
