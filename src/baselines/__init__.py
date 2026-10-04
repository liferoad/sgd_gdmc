"""Baseline optimizers: SGD, Momentum, Adam, Projected-GD, SGLD.

The first three are thin wrappers around ``torch.optim``. ``ProjectedGD``
is a wrapper that, after each ``step``, snaps every weight to a grid
(so it is the QAT-style "GD-then-project" baseline). ``SGLD`` is a
small Langevin-dynamics implementation that adds Gaussian noise scaled
by the learning rate at each step (Welling & Teh, 2011).
"""

from .sgd import make_sgd
from .momentum import make_momentum
from .adam import make_adam
from .projected_gd import ProjectedGD, make_projected_gd, grid_delta
from .adam8bit import Adam8Bit, make_adam8bit
from .signsgd import SignSGD, make_signsgd
from .sgld import SGLD, make_sgld

__all__ = [
    "make_sgd",
    "make_momentum",
    "make_adam",
    "ProjectedGD",
    "make_projected_gd",
    "grid_delta",
    "Adam8Bit",
    "make_adam8bit",
    "SignSGD",
    "make_signsgd",
    "SGLD",
    "make_sgld",
]
