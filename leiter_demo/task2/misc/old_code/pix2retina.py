import math
import numpy as np
from PIL import Image

pi = math.pi

nWedges = 25
#nRings = 36
h0 = 1	# height of "trapezoidal" "bubble" at ring 0 in pixels

R0 = h0 * nWedges / (2*pi)
ratio = pi * (2*R0 + h0) / (nWedges - pi) / h0
radius_split = math.log(1 + h0/(2*R0),ratio)

print R0,ratio,radius_split

def main(pix):
	#radii, heights = detRadiiAndHeights()
	#printRings()

	x,y = pix.shape
	center = np.array([(x-1)/2.0,(y-1)/2.0])
	#print center

	target_radius = min(x,y) / 2.0
	nRings = findRing(target_radius) + 1

	#print nRings

	retina = np.empty((nWedges,nRings),dtype=list)

	for px in xrange(x):
		for py in xrange(y):
			p = np.array([px,py])
			diff = p - center
			ring = findRing(norm(diff))

			#print px,py,ring

			if (ring >= nRings or ring == -1):
				continue

			angle = math.atan2(diff[1],diff[0])
			if (angle < 0):
				angle += 2*pi

			wedge = int(nWedges * angle / (2*pi))

			if (retina[wedge,ring] is None):
				retina[wedge,ring] = [pix[px,py]]
			else:
				retina[wedge,ring].append(pix[px,py])

	retina = np.array([[np.average(values) if values is not None else 0.0 for values in row] for row in retina])

	#return retina
	
	
	for w in retina:
		print w

	test = np.array([[int(round(value)) for value in row] for row in retina])
	for w in test:
		print w
      

	im = Image.fromarray(test.astype('uint8'))
	im.show()
	



	
	'''
	filter = ImageFilter.GaussianBlur(radius=3)
	filter = ImageFilter.BLUR
	blurred_im = im.filter(filter)
	'''

def printRings():
	for i in xrange(20):
		print('%d\t%f\t%f' % (i,R0*math.pow(ratio,i),h0*math.pow(ratio,i)))

def getRatio():
	return pi * (2*R0 + h0) / (nWedges - pi) / h0

def findRing(r):
	#print r
	if (r == 0):
		return -1	# inside central, "special" pixel

	level = math.log(r/R0,ratio)
	if (level < 0):
		if (radius_split - 1 < level):
			return 0
		else:
			return -1

	level_lb = int(level)

	if (level - level_lb < radius_split):
		return level_lb
	else:
		return level_lb + 1


def dist(v1,v2):
	#return math.sqrt(math.pow(v1[0]-v2[0],2) + math.pow(v1[1]-v2[1],2))
	return norm(v1 - v2)

def norm(v):
	return np.sqrt((v*v).sum(axis=0))



def detRadiiAndHeights():
	R0 = h0 * nWedges / (2*pi) 
	radii = [R0]
	heights = [h0]

	for i in xrange(1,nRings):
		h = pi * (2*radii[i-1] + heights[i-1]) / (nWedges - pi)
		radii.append(h * nWedges / (2*pi))
		heights.append(h)

	return (radii,heights)

if __name__ == "__main__":
    #main(np.random.rand(50,50))
    main(np.asarray(Image.open('symmetric_test.jpg').convert('L')))
