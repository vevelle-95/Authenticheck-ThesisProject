"""Run offline adaptation checks without downloading pretrained models."""

import sys
import unittest

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover("tests", pattern="test_adaptation.py")
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if outcome.wasSuccessful() else 1)
