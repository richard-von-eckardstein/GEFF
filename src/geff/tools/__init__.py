"""
This module contains tools to compute physical observables from a GEF result.

Currently, the following tools are available:
1. Compute the tensor power spectrum using `geff.tools.pt`.
2. Compute and analyze the gravitational wave spectrum using `geff.tools.gw`.
"""
from .pt import PowSpecT
from .gw import omega_gw