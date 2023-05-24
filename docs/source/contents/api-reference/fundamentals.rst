.. _fundamentals:

.. module:: mappy.fundamentals

============
Fundamentals
============

.. currentmodule:: mappy

Mode
----

.. autosummary::
    :toctree: generated/

    Mode
    ContinuousMode
    DiscreteMode


Trajectory
----------

.. autosummary::
    :toctree: generated/

    Traj
    ModeTraj
    Sol
    ModeSol


Diffeomorphism
--------------

.. autosummary::
    :toctree: generated/

    Diffeomorphism
    PoincareMap


Initial Value Boundary Mode Problem
-----------------------------------

.. autosummary::
    :toctree: generated/

    solve_ivbmp
    solve_poincare_map


Result Classes
--------------

.. autosummary::
    :toctree: generated/

    BasicResult
    ModeStepResult
    SolveIvbmpResult
    DiffeomorphismResult

Exceptions
----------

.. autosummary::
    :toctree: generated/

    SomeJacUndefined
    SomeHesUndefined
    TransitionKeyError
    AllModesKeyError
    NextModeNotFoundError

Tools
-----

.. autosummary::
    :toctree: generated/

    convert_y_ndarray
    revert_y_ndarray