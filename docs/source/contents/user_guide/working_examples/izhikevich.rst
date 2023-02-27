Izhikevich neuron model
-----------------------

System definitions
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Model
    @CM.function(dimension = 2) # Decorator
    def izhikevich (y, param):
        v, u = y
        a = param.a
        b = param.b
        c = param.c

        return np.array([
            0.04 * (v**2) + 5.0 * v + 140.0 - u + I,
            a * (b * v - u)
        ])

    # Firing border
    @CM.border(direction = 1)   # Decorator
    def fire_border(y, param):
        return y[0] - 30.0      # Assume y is 2-D numpy.array

    # Firing jump
    @DM.function(domain_dimension = 2, codomain_dimenstion = 2) # Decorator
    def jump(y, param):
        C = np.array([-30 + param.c, param.d])
        return y + C

System parameters
^^^^^^^^^^^^^^^^^

For the parameters to pass to the dynamical system,
the package provides a useful class named `DictVector`.
The class implements the constructor accepting a `dict` type variable
and setting its values as attributes of the class instance.

For the Izhikevich neuron model, the following class is compatible with the above source.

.. code-block:: python

    # Parameter
    class Parameter(DictVector):
        a: float = 0.2
        b: float = 0.2
        c: float = -50
        d: float = 2.0
        I: float = 10

Giving all default values (0.2, -50, and so on), we can flexibly set the parameter value as follows.

.. code-block:: python

    >>> p1 = Parameter()
    >>> p2 = Parameter({'a': 0.1)

.. note::
    JSON file is a good idea to store the parameter setting (and also other data)
    since the class accepts dictionary type variables.

    .. code-block:: json
        :caption: param.json

        {
            "param": {
                "a": 0.1,
                "b": 0.1,
                "c": -50,
                "d": 1.9,
                "I": 12
            }
        }

    .. code-block:: python

        >>> import json
        >>> with open('param.json', 'r') as f:
        >>>     dict_data = json.load(f)
        >>> p = Parameter(dict_data["param"])

Mode settings
^^^^^^^^^^^^^

.. code-block:: python

    mode0 = CM(izhikevich, borders=[fire_border])
    mode1 = DM(jump)

    mode0.next = [mode1]
    mode1.next = mode0

Run
^^^

.. code-block:: python

    # Initial value and parameters
    y0    = np.array([-50, -1.7])
    param = Parameter()

    # Run from `mode0` to `mode0`
    result = solve_ivbmp(y0=y0, initial_mode=mode0, args=param)

    print(result)
    # {'y': array([-50.        ,  -1.70979509]), 'jac': None}