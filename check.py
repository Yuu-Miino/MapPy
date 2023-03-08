import numpy as np

jac = np.array([[ 1.177370266281e+02, -8.310425277442e+01],
    [ 1.045829162634e+00, -8.970633870958e-02]])
djac = np.array([[ 1.177373546259e+02, -8.310452106479e+01],
    [ 1.045832112570e+00, -8.970902202297e-02]])

print((djac-jac)/1e-5)

jac = np.array([[0.,             0.            ],
 [0.37237016737,  0.385652262573]])
djac = np.array([[0.,             0.            ],
 [0.372371500858 ,0.385650930572]])

print((djac-jac)/1e-5)