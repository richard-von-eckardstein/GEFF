import sys
import os
abspath = os.path.dirname(__file__)
path = os.path.join(abspath, "../")
sys.path.append(path)

from Benchmark import Benchmark
import numpy as np
from src.Tools.ModeByMode import ModeByMode, ReadMode

if __name__ == "__main__":
    model="Classic"
    setting={}

    G, spec = Benchmark(model, setting)

    MbM = G.MbM(G)

    MbM.CompareToBackgroundSolution(spec)

