#kalman w/ vp + angles

import numpy
import pylab
import csv
from math import atan, cos, pi

FOCAL_IN_PX = 378.0
WIDTH = 640.0
HEIGHT = 480.0



# parse csv file
data = []
with open('actual_vp.txt', 'rb') as csvfile:
   reader = csv.reader(csvfile, delimiter=',', quotechar='"')
   for row in reader:
      data.append(row)

K = numpy.array(    [[FOCAL_IN_PX,   0.0,           WIDTH/2],
                    [0.0,           FOCAL_IN_PX,    HEIGHT/2],
                    [0.0,           0.0,            1.0]] )

Kinv = numpy.linalg.inv(K)

print Kinv
#print data
#convert strings to pixel value floats
x_data = numpy.array( [float(row[0]) for row in data] )
y_data = numpy.array( [float(row[1]) for row in data] )



#convert x data to world coordinates
xWC = [ x * Kinv[0][0] + Kinv[0][2] for x in x_data ]
yWC = [ y * Kinv[1][1] + Kinv[1][2] for y in y_data ]
#use world coords to calculate theta, gamma
thetas = [ atan(y) for y in yWC ]
gammas = [ atan( -x / cos(theta) ) for x,theta in zip(xWC, thetas) ]

thetasDeg = [ 180.0 / pi * theta for theta in thetas]
gammasDeg = [ 180.0 / pi * gamma for gamma in gammas]

x_data = gammasDeg
y_data = thetasDeg

# intial parameters
n_iter = len(x_data)
sz = (n_iter,) # size of array
z = x_data
Q = 1e-5 # process variance

# allocate space for arrays
x_xhat=numpy.zeros(sz)      # a posteri estimate of x
x_P=numpy.zeros(sz)         # a posteri error estimate
x_xhatminus=numpy.zeros(sz) # a priori estimate of x
x_Pminus=numpy.zeros(sz)    # a priori error estimate
x_K=numpy.zeros(sz)         # gain or blending factor
x = x_data
x_R = .0005 # estimate of measurement variance, change to see effect
# intial guesses
x_xhat[0] = 0.0
x_P[0] = 1.0

# allocate space for arrays
y_xhat=numpy.zeros(sz)      # a posteri estimate of x
y_P=numpy.zeros(sz)         # a posteri error estimate
y_xhatminus=numpy.zeros(sz) # a priori estimate of x
y_Pminus=numpy.zeros(sz)    # a priori error estimate
y_K=numpy.zeros(sz)         # gain or blending factor
y = y_data
y_R = .005 # estimate of measurement variance, change to see effect
# intial guesses
y_xhat[0] = 0.0
y_P[0] = 1.0

for k in range(1,n_iter):
    # time update
    x_xhatminus[k] = x_xhat[k-1]
    x_Pminus[k] = x_P[k-1]+Q

    x_delta = abs(x_xhat[k] - x[k])
    if( x_delta > 25.0 ):
        x_xhat[k] = x_xhat[k-1]
        x_K[k] = x_K[k-1]
        x_P[k] = x_P[k-1]
        
    else:
        # measurement update
        x_K[k] = x_Pminus[k]/( x_Pminus[k]+x_R )
        x_xhat[k] = x_xhatminus[k]+x_K[k]*(x[k]-x_xhatminus[k])
        x_P[k] = (1-x_K[k])*x_Pminus[k]

    # time update
    y_xhatminus[k] = y_xhat[k-1]
    y_Pminus[k] = y_P[k-1]+Q

    y_delta = abs(y_xhat[k] - y[k])
    if( y_delta > 15.0):
        y_xhat[k] = y_xhat[k-1]
        y_K[k] = y_K[k-1]
        y_P[k] = y_P[k-1]
    else:
        # measurement update
        y_K[k] = y_Pminus[k]/( y_Pminus[k]+y_R )
        y_xhat[k] = y_xhatminus[k]+y_K[k]*(y[k]-y_xhatminus[k])
        y_P[k] = (1-y_K[k])*y_Pminus[k]


pylab.figure()
pylab.plot(x_data,'k+',label='measurements')
pylab.plot(x_xhat,'b-',label='estimate')
pylab.legend()
pylab.xlabel('frame')
pylab.ylabel('gamma (deg)')

pylab.figure()
pylab.plot(y_data,'k+',label='measurements')
pylab.plot(y_xhat,'b-',label='estimate')
pylab.legend()
pylab.xlabel('frame')
pylab.ylabel('theta (deg)')

pylab.show()

