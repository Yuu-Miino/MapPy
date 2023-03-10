import numpy as np

Hg = np.random.randint(100, size=18).reshape(2, 3, 3)   # X, W*, W*
Hg = (Hg + Hg.transpose(0, 2, 1)) / 2                   # Make Hg symmetric
Jf = np.random.randint(100, size=12).reshape(3, 4)      # W, V*

# (2, 4) contraction (for W*, W)
cont24 = np.trace(np.tensordot(Hg, Jf, axes=0), axis1=1, axis2=3)

# (3, 4) contraction (for W*, W)
cont34 = np.trace(np.tensordot(Hg, Jf, axes=0), axis1=2, axis2=3)

# @ operation
atop = Hg @ Jf

# Comparison
print(cont24 == atop) # (2, 3, 4) tensor in X, W*, V*, with All True
print(cont34 == atop) # (2, 3, 4) tensor in X, W*, V*, with All True