import numpy as np
import matplotlib.pyplot as plt

'''Creates a set of noisey points along the line 3x+2y+8=0 and
  then fits a line to those points (and visualises the input and output)'''

# Parameters for line [a,b,c] i.e. for line ax+by+c=0
lineparams = np.array([3,2,8])
# Creates an array of 11 equally spaced values from 0 to 10
# These will be our x values from which we will compute our y values
xpts = np.linspace(0,10,20)
# Compute the correspondening ground truth y values i.e. y = -(c+ax)/b
ypts = (-(lineparams[2] + lineparams[0]*xpts) / lineparams[1])

# Add noise to the ground truth values
yptsn = ypts + np.random.normal(0,1,np.size(ypts))

# Create the A matrix for the system
A = np.zeros((np.size(yptsn),3))
A[0:,0] = xpts.transpose()
A[0:,1] = yptsn.transpose()
A[0:,2] = 1

# Compute the eigenvalues and eigen vectors of A'A
# Note that the eigenvalues are not necessarily in order so
# we will need to use argmin below to find the index of the minimum
# eigenvalue
D,V = np.linalg.eig(A.transpose().dot(A))

# plot the noisey values in green
plt.ion() #set interactive plot
plt.cla()
plt.show()
plt.plot(xpts,yptsn,'g*')

# extract the estimated line parameters as the eigenvector corresponding
# to the smallest eigenvalue. In this case this corresponds to the col
# vector of V indexed using the index of the minimum eigenvalue in D
estlp = V[:,np.argmin(D)]

# Now compute the estimated y values for each of the original x's using
# the estimated line parameters and plot the result as a red line
ypts_est = (-(estlp[2] + estlp[0]*xpts) / estlp[1])
plt.plot(xpts, ypts_est, '-r.')
