"""
backtester/cut_simulator.py
----------------------------
Simple cut model: estimates P(make_cut) from sg_composite.
With Option B we use DataGolf's make_cut probability directly.
"""

import sys
import logging
import numpy as np
from scipy.special import expit

log = logging.getLogger(__name__)


class CutModel:
    """P(make_cut) = sigmoid(a * sg_composite + b)"""
    DEFAULT_A = 1.2
    DEFAULT_B = 0.6

    def __init__(self):
        self.a = self.DEFAULT_A
        self.b = self.DEFAULT_B

    def predict(self, sg_composite: float) -> float:
        return float(expit(self.a * sg_composite + self.b))


_cut_model = None

def get_cut_model() -> CutModel:
    global _cut_model
    if _cut_model is None:
        _cut_model = CutModel()
    return _cut_model
