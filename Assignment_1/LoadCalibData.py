'''Visualises the data file for cs410 camera calibration assignment
To run: %run LoadCalibData.py
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# From Niall ...
np.set_printoptions(threshold=np.inf, suppress=True)

data = np.loadtxt('data.txt')

def lectureExample():
	fig = plt.figure()
	ax = fig.gca(projection="3d")
	# Added comment
	# Accesssing the first three columns -> 3D data.
	ax.plot(data[:,0], data[:,1], data[:,2],'k.')

	fig = plt.figure()
	# Adding comment
	# Accesssing the last two columns -> 2D data.
	ax = fig.gca()
	ax.plot(data[:,3], data[:,4],'r.')

	plt.show()

def calibrateCamera3D(data):
	two = data[::,3:5]
	three = data[::,0:3]

	A = np.zeros((len(data) * 2, 12)) #982

	X_list = []
	Y_list = []

	for i in range(len(data)): #491
		threex = three[i,0]
		threey = three[i,1]
		threez = three[i,2]

		twox = two[i,0]
		twoy = two[i,1]

		X = threex, threey, threez, 1, 0, 0, 0, 0, -twox*threex, -twox*threey, -twox*threez, -twox*1
		Y = 0, 0, 0, 0, threex, threey, threez, 1, -twoy*threex, -twoy*threey, -twoy*threez, -twoy*1

		X_list.append(list(X))
		Y_list.append(list(Y))
		# print(X_list[i][2])

	a = 0

	for j in range(0, 982, 2):
		if j <= 982:
			#a = a + 1
			# print(X_list[a])
			# print(Y_list[a])
			A[j] = X_list[a]
			A[j+1] = Y_list[a]
			# moved below
			a = a + 1

	D,V = np.linalg.eig(A.transpose().dot(A))
	est = V[:,np.argmin(D)]

	# Build the camera matrix
	P = np.zeros((3,4))
	P[0:] = est[0:4]
	P[1:] = est[4:8]
	P[2:] = est[8:12]

	# print(P)
	return P

# Camera Matrix
P = calibrateCamera3D(data)
print(P)

def visualiseCameraCalibration3D(data, P):
	# The 2D representation.
	fig = plt.figure()
	ax = fig.gca()
	ax.plot(data[:,3], data[:,4],'r.')
	#plt.show()

	threex = data[:,0]
	threey = data[:,1]
	threez = data[:,2]

	# Now onto the 3D aspect.

	twox = data[:,3]
	twoy = data[:,4]

	# Convert the 3d points in the data into homogenous coordinates
	threepoints = np.ones((491, 4))
	for i in range(len(data)):
		threepoints[i,0] = threex[i]
		threepoints[i,1] = threey[i]
		threepoints[i,2] = threez[i]

	# Convert the 2d points in the data into homogenous coordinates
	twopoints = np.ones((491, 4))
	for i in range(len(data)):
		twopoints[i,0] = twox[i]
		twopoints[i,1] = twoy[i]

	# print(twopoints)
	# print(threepoints)

	# 3d points
	threepoints_new = P.dot(threepoints.transpose())
	finalpoints = threepoints_new.transpose()

	# 2d points
	twopoints_new = P.dot(twopoints.transpose())
	finalpoints_two = twopoints_new.transpose()
	print(finalpoints_two.shape)

	fig = plt.figure()
	ax = fig.gca()
	ax.plot(finalpoints_two[:,0]*100, finalpoints_two[:,1]*100, 'g.')
	ax.plot(data[:,3], data[:,4],'r.')
	plt.show()

visualiseCameraCalibration3D(data, P)

"""
Function to get the mean, vaiance, minimum and maximum
"""
def evaluateCameraCalibration3D(data, P):
	X=data[:,0]
	Y=data[:,1]
	Z=data[:,2]
	data.shape
    #problem is they are not equal sized matrices, change the data one to be 3 cols
	data1=(data[:,0], data[:,1], data[:,2])

    # now do the 3D image representation
    #objpoints = (data[:,0],data[:,1],data[:,2])
	obj3D = np.ones((491,4))
	obj3D.shape
	for i in range(0,491):
		obj3D[i,0] = X[i]
		obj3D[i,1] = Y[i]
		obj3D[i,2] = Z[i]

	obj3D
    #so we now multiply the obj3D points by the P camera matrix
	obj3D_P = P.dot(obj3D.transpose())
    #checking the shape
	obj3D_P.shape
    #transpose it so we can plot all of it easier
	obj3D_PT=obj3D_P.transpose()
	obj3D_PT.shape
    #calculate the distance, use the original data, with the matrix.P called obj3D
	dist = sp.spatial.distance.cdist(data1,obj3D_P,'euclidean')
	dist

def linefit():
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
	A = np.zeros((np.size(yptsn), 3))
	A[0:,0] = xpts.transpose()
	A[0:,1] = yptsn.transpose()
	A[0:,2] = 1

	# Compute the eigenvalues and eigen vectors of A'A
	# Note that the eigenvalues are not necessarily in order so
	# We will need to use argmin below to find the index of the minimum
	# eigenvalue
	D,V = np.linalg.eig(A.transpose().dot(A))

	# plot the noisey values in green
	# plt.ion() #set interactive plot
	plt.cla()
	plt.plot(xpts,yptsn,'g*')

	# extract the estimated line parameters as the eigenvector corresponding
	# to the smallest eigenvalue. In this case this corresponds to the col
	# vector of V indexed using the index of the minimum eigenvalue in D
	estlp = V[:,np.argmin(D)]

	# Now compute the estimated y values for each of the original x's using
	# the estimated line parameters and plot the result as a red line
	ypts_est = (-(estlp[2] + estlp[0]*xpts) / estlp[1])
	plt.plot(xpts, ypts_est, '-r.')

	# plt.show()
