import numpy as np

def dist(v1,v2):
	#return math.sqrt(math.pow(v1[0]-v2[0],2) + math.pow(v1[1]-v2[1],2))
	return norm(v1 - v2)

def norm(v):
	return np.sqrt((v*v).sum(axis=0))

def density(pix):
	x,y = pix.shape
	center = np.array([(x-1)/2.0,(y-1)/2.0])

	radius = min(x,y) / 2.0

	density = 0.0
	counter = 0
	for sx in xrange(x):
		for sy in xrange(y):
			point = np.array([sx,sy])
			if dist(point,center) < radius:
				density += pix[sx,sy]
				counter += 1

	print counter

	return density / counter

# returns weight matrix s.t. if pixel is within radius (which equals size / 2.0), pixel = 1 else 0
def getWeights(size):
	wts = np.empty((size,size))
	center = np.array([(size-1)/2.0,(size-1)/2.0])

	radius = size / 2.0

	for sx in xrange(size):
		for sy in xrange(size):
			point = np.array([sx,sy])
			if dist(point,center) < radius:
				wts[sx,sy] = 1
			else:
				wts[sx,sy] = 0

	return wts