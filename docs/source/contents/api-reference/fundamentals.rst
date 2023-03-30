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

Initial Value Boundary Mode Problem
-----------------------------------

.. autosummary::
    :toctree: generated/

    solve_ivbmp
    solve_poincare_map
    PoincareMap

Result Classes
--------------

.. autosummary::
    :toctree: generated/

    BasicResult
    ModeStepResult
    SolveIvbmpResult

Exceptions
----------

.. autosummary::
    :toctree: generated/

    SomeJacUndefined
    SomeHesUndefined
    TransitionKeyError
    AllModesKeyError
    NextModeNotFoundError