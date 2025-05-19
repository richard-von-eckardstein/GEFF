import sys
import os
abspath = os.path.dirname(__file__)
path = os.path.join(abspath, "../")
sys.path.append(path)

from Benchmark import Benchmark
from src.GEF import GEF
import numpy as np

if __name__ == "__main__":
    model="Classic"
    setting={}

    G, spec = Benchmark(model, setting, loadGEF=False, loadspec=False)

    sol = G.Solver.RunGEF(100, 120, 1e-20, 1e-6, nmodes=None, ensureConvergence=False)



