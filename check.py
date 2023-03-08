import numpy as np

jac = np.array([[ 1.177370266281e+02, -8.310425277442e+01],
    [ 1.045829162634e+00, -8.970633870958e-02]])
djac = np.array([[ 1.177373546259e+02, -8.310452106479e+01],
    [ 1.045832112570e+00, -8.970902202297e-02]])

print((djac-jac)/1e-5)

jac = np.array([[0.,             0.            ],
 [0.37237016737,  0.385652262573]])
djacx = np.array([[ 0, 0],
 [ 3.723701576455e-01,  3.856522759075e-01]])
djacy = np.array([[0.,             0.,            ],
 [0.372370180705, 0.385652249253]])

print('x', (djacx-jac)/1e-7)
print('y', (djacy-jac)/1e-7)

hes = np.array([[[ 0.,              0.            ],
  [-0.122314388198,  0.107378966938]],

 [[ 0.,              0.            ],
  [ 0.151043394255, -0.114869206755]]])

print('hes(i)x\n', (djacx-jac)/1e-7 - hes[0])
print('hes(i)y\n', (djacy-jac)/1e-7 - hes[1])

jac = np.array([[ 1.177371217003e+02, -8.310432022971e+01],
 [ 1.045829351875e+00, -8.970645965926e-02]])
djac = np.array([[ 1.177371249802e+02, -8.310432291256e+01],
 [ 1.045829381374e+00, -8.970648649233e-02]])

print((djac-jac)/1e-7)

hest= np.array([[[ 266.584540505949, -180.078865977687],
  [   1.237136385353,   -0.805915927272]],

 [[-180.078865977687,  123.58922972701 ],
  [  -0.805915927272,    0.540469526027]]])

dfdy = np.array([[ 7.4,  -1.  ],
 [ 0.04, -0.2 ]])

jact = np.array([[ 1.177371217003e+02, -8.310432022971e+01],
 [ 1.045829351875e+00, -8.970645965926e-02]])

dtaudx0 = np.array([-0.34657522963,   0.244628868541])
ddtaudx0 = np.array([-0.346575239324,  0.244628876466])
dtaudx02 = (ddtaudx0-dtaudx0)/1e-7

dydt = np.array([339.715916299391,  1.943183259878])

jac = jact + np.outer(dydt, dtaudx0)
hes = hest[1] + np.outer(dfdy@ jact[:,1], dtaudx0) + np.outer(dydt, dtaudx02)
print('jac\n', jac)
print('hes\n', hes)
print('term1\n', hest[1])
print('term2\n', np.outer(dfdy@ jact[:,1], dtaudx0))
print('term3\n', np.outer(dydt, dtaudx02))


B = np.array([[ 0.,              0.            ],
 [-0.005720024193,  1.            ]])
Bx = np.array([[ 0,  0],
 [-5.720024171406e-03,  1]])
By = np.array([[ 0.,              0.            ],
 [-0.005720024171,  1.            ]])

print('dBdx0\n', (Bx-B)/1e-7)
print('dBdy0\n', (By-B)/1e-7)