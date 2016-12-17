import math
import numpy as np
#import heapq as hq
from random import randrange
from PIL import Image
from copy import deepcopy

from collections import namedtuple
RelativeRetina = namedtuple("RelativeRetina","ret start density") # simple container used to hold a Retina object and its location in its TargetSlice
Result = namedtuple("Result","start wedge_index sim")	# simple container used to hold results for primary,secondary,etc. regions for each choice

pi = math.pi

# used if nWedges isn't specified in Template ctor
# if less than 1, is fraction of total_size
# else is the number of pixels
# *****only an approximation
default_blindspot_diameter = 0.2

# used if h0 isn't specified in Template ctor
# exact
default_h0 = 1

# used if threshold isn't specified in TargetSlice ctor
# is fraction of densest region of target
# purpose: discard anomalous blackest regions
default_threshold = 0.75

class Template(object):
	"""Performs all the calculations to construct a Retina object but does not actually create one
	   Useful for finding weights before pixels to be converted are known"""

	# these are instantiated in self.findWeights()
	wts = None
	inverse_wts = None
	as_wts = None # anti-symmetric weights
	as_inverse_wts = None

	def __init__(self, size, nWedges=None, h0=None):
		self.size = size 	# side length of square template
		midpoint = (self.size - 1) / 2.0
		self.center = np.array([midpoint,midpoint])

		self.h0 = default_h0 if h0 == None else h0

		if nWedges is None:
			if default_blindspot_diameter < 1:
				bs_radius = 0.5*(default_blindspot_diameter * self.size)
			else:
				bs_radius = 0.5*(default_blindspot_diameter)

			self.nWedges = int(round(2*pi * (bs_radius + 0.5*self.h0) / self.h0))
		else:
			self.nWedges = nWedges

		self.R0 = self.h0 * self.nWedges / (2*pi)
		self.bs_radius = self.R0 - 0.5*self.h0

		self.ratio = pi * (2*self.R0 + self.h0) / (self.nWedges - pi) / self.h0
		self.radius_split = math.log(1 + self.h0/(2*self.R0),self.ratio)

		self.nRings = self.findRing(self.size / 2.0)

		# Explanation of rationale behind self.wts for when they are applied to Retina objects and calculating densities:
		# (self.inverse_wts is also calculated within the method and serve the opposite purpose)

		# -has dimension self.size by self.size
		# -used when calculating density = np.average(pix,weights=self.wts)
		# 
		# -biased towards more central rings with highest resolution; weight of pixels in ring i is inversely proportional to number of pixels mapped to ring i
		# 	-two-fold purpose:
		#	 	-maximizes information gain in central regions where info has greater import due to its higher-res (densest_square)
		#		-increases likelihood that the similarities in density between the target and subspaces of the choices arise because of similarity in 
		# 		 more central, high-res regions areas, which subsequently can be more accurately assessed by the similarity algorithm (sim_density_search)
		#
		# -biased towards blindspot density since density is the only heuristic that can be used to determine similarity (can't rotate)
		#	-pixels in blindspot have weight equal to that which is assigned to pixels in ring 0
		#
		# -flaw #1: selects regions that are anomalously black; that's why self.inverse_wts was introduced
		# -flaw #2: has potential to select symmetric regions which are less distinguishable when determining proper rotation
		#	-solution: add anti-symmetric weights/inverse_weights: 0/1 if in left half in circle, 1/0 otherwise
		self.findWeights()

	def findWeights(self):
		wts_by_ring = np.empty(self.nRings)
		inverse_wts_by_ring = np.empty(self.nRings)
		for i in xrange(self.nRings):
			wts_by_ring[i] = math.pow(self.ratio**2,-i)
			inverse_wts_by_ring[i] = math.pow(self.ratio**2, i - (self.nRings - 1))

		wts = np.empty((self.size,self.size))
		inverse_wts = np.empty((self.size,self.size))

		as_wts = np.empty((self.size,self.size))
		as_inverse_wts = np.empty((self.size,self.size))

		for px in xrange(self.size):
			for py in xrange(self.size):
				p = np.array([px,py])
				diff = p - self.center
				ring = self.findRing(norm(diff))
	
				if (ring >= self.nRings):
					wts[px,py] = 0.0
					inverse_wts[px,py] = 0.0

					as_wts[px,py] = 0.0
					as_inverse_wts[px,py] = 0.0
					continue

				if (py < (self.size - 1) / 2):
					as_wts[px,py] = 0.0
					as_inverse_wts[px,py] = 1.0

				if (ring == -1):
					wts[px,py] = wts_by_ring[0]
					inverse_wts[px,py] = inverse_wts_by_ring[0]
					continue

				wts[px,py] = wts_by_ring[ring]
				inverse_wts[px,py] = inverse_wts_by_ring[ring]

		self.wts = wts
		self.inverse_wts = inverse_wts

		self.as_wts = as_wts
		self.as_inverse_wts = as_inverse_wts

	def findRing(self,r):
		#print r
		if (r == 0):
			return -1	# inside central, "special" pixel
	
		level = math.log(r/self.R0,self.ratio)
		if (level < 0):
			if (self.radius_split - 1 < level):
				return 0
			else:
				return -1
	
		level_lb = int(level)
	
		if (level - level_lb < self.radius_split):
			return level_lb
		else:
			return level_lb + 1

	def printRings(self):
		for i in xrange(self.nRings):
			print('%d\t%f\t%f' % (i,self.R0*math.pow(self.ratio,i),self.h0*math.pow(self.ratio,i)))

	# Note: pix must be of dimension self.size by self.size (same as self.wts)
	def calcDensity(self,pix):
		return np.average(pix,weights=self.wts)
		'''
		return np.mean([np.average(1-pix,weights=self.wts),
						np.average(pix,weights=self.inverse_wts)])
		'''

	# Creates Retina object using self and pix
	def createRetina(self,pix):
		return Retina(self,pix)

	def retina_size(self):
		return (self.nWedges,self.nRings)


# Note: requires template object to create
class Retina(object):
	"""Numpy array of pixels arranged in polar fashion"""

	def __init__(self, template, pix, retina=None):
		self.template = template
		self.pix = pix
		self.retina = self.findRetina() if retina is None else retina

		self.density = self.template.calcDensity(self.pix)

	# if pix is not specified, returns deep copy
	# else returns new Retina object with same template but different pix
	def copy(self,pix=None):
		if pix == None:
			return deepcopy(self)
		else:
			return self.template.createRetina(pix)	

	def findRetina(self):	
		retina = np.empty(self.retina_size(),dtype=list)

		for px in xrange(self.template.size):
			for py in xrange(self.template.size):
				p = np.array([px,py])
				diff = p - self.template.center
				ring = self.findRing(norm(diff))
	
				if (ring >= self.template.nRings or ring == -1):
					continue
	
				angle = math.atan2(diff[1],diff[0])
				if (angle < 0):
					angle += 2*pi
	
				wedge = int(self.template.nWedges * angle / (2*pi))
	
				if (retina[wedge,ring] is None):
					retina[wedge,ring] = [self.pix[px,py]]
				else:
					retina[wedge,ring].append(self.pix[px,py])
	
		retina = np.array([[np.average(values) if values is not None else 0.0 for values in row] for row in retina])
	
		return retina

	def findRing(self,r):
		return self.template.findRing(r)

	def printRings(self):
		return self.template.printRings()

	def pix_size(self):
		return self.pix.shape

	def retina_size(self):
		return self.template.retina_size()

	def rotate(self,nRotations=None):
		if nRotations == None:
			self.retina = np.roll(self.retina, 1, axis=0)
		else:
			self.retina = np.roll(self.retina, nRotations, axis=0) 

	# saves and returns retina as image file
	def save(self,fname):
		temp = (1.0 - self.retina) * 255
		temp = np.array([[int(round(value)) for value in row] for row in temp])
		im = Image.fromarray(temp.astype('uint8'))

		im.save(fname)

		return im

class TargetSlice(object):
	"""Collection of Retina Objects"""
	def __init__(self, pix, t_name, threshold=None, size=None, nWedges=None, h0=None):
		self.pix = pix
		self.t_name = t_name
		self.threshold = default_threshold if threshold is None else threshold
		self.size = min(self.pix.shape) if size is None else size

		self.template = Template(self.size,nWedges,h0)

		# ordered by density; all entries are < self.threshold * highest_density to protect against anomalous blackest regions
		self.ordered_densities = self.findDensities()	# format: (density,start)

		self.primary = self.createRelativeRetina(0)
		p_start = self.primary.start

		secondary_index = 1
		s_start = np.array(self.ordered_densities[secondary_index][1])
		sp_distance = dist(p_start,s_start)
		while not self.goldilocks(sp_distance):
			secondary_index += 1
			s_start = np.array(self.ordered_densities[secondary_index][1])
			sp_distance = dist(p_start,s_start)

		#secondary_index = randomInt(1,len(self.ordered_densities))
		self.secondary = self.createRelativeRetina(secondary_index)

		'''
		tertiary_index = -1
		t_start = np.array(self.ordered_densities[tertiary_index][1])
		tp_distance = dist(p_start,t_start)
		while tp_distance > self.template.nWedges * self.size / (2*pi): # margin of error is more than self.size
			tertiary_index -= 1
			if tertiary_index == secondary_index:
				tertiary_index = -1
				break
			t_start = np.array(self.ordered_densities[secondary_index][1])
			tp_distance = dist(p_start,t_start)

		self.tertiary = self.createRelativeRetina(tertiary_index)
		'''

		self.sp_difference = self.secondary.start - self.primary.start
		self.sp_dist = norm(self.sp_difference)

		# Note: for primary or secondary, the most_sim heuristic only depends on maximising the similarity of either the primary or secondary region
		# Note: for composite, the most_sim heuristic depends on maximising both the similarity of the primary and secondary regions 
		self.results = {'primary':	[],		# format: (primary_result, associated_secondary_result) for each choice
						'secondary':[],		# format: (associated_primary_result, secondary_result) for each choice
						'composite':[]}		# format: (coupled_primary_result, coupled_secondary_result) for each choice

		'''
		tertiary_index = randomInt(1,len(self.ordered_densities),[secondary_index])
		self.tertiary = self.createRelativeRetina(tertiary_index)
		'''

	# i is the index of self.ordered_densities at which the RelativeRetina container will be created
	def createRelativeRetina(self,i):
		density, start = self.ordered_densities[i]
		sx, sy = start

		return RelativeRetina(self.template.createRetina(self.pix[sx:sx+self.size,sy:sy+self.size]),np.array(start),density)

	def findDensities(self):
		densities = []
		for sx in xrange(self.pix.shape[0] - self.size + 1):
			for sy in xrange(self.pix.shape[1] - self.size + 1):
				densities.append((self.template.calcDensity(self.pix[sx:sx+self.size,sy:sy+self.size]),(sx,sy)))

		densities.sort(reverse=True)

		cutOff = -1
		for i in xrange(len(densities)):
			if densities[i][0] < self.threshold*densities[0][0]:
				cutOff = i
				break

		return densities[cutOff:]

	def goldilocks(self,distance):
		return self.size <= distance and distance <= self.template.nWedges * self.size / (2*pi) # margin of error is more than self.size

# returns random integer in [a,b) that is not in list of invalids (if specified)
def randomInt(a,b,invalids=None):
	i = randrange(a,b)
	if invalids is None:
		return i 

	while i in invalids:
		i = randrange(a,b)

	return i

def calcDensity(pix,nWedges=None,h0=None):
	size,size = pix.shape
	return Template(size,nWedges,h0).calcDensity(pix)

def pix2ret(pix,nWedges=None,h0=None):
	size,size = pix.shape
	return Template(size,nWedges,h0).createRetina(pix)

def dist(v1,v2):
	#return math.sqrt(math.pow(v1[0]-v2[0],2) + math.pow(v1[1]-v2[1],2))
	return norm(v1 - v2)

def norm(v):
	return np.sqrt((v*v).sum(axis=0))

def createRetina(file,nWedges=None,h0=None):
	im = Image.open(file).convert('L')
	pix = 1.0 - np.asarray(im) / 255.0

	size = min(pix.shape)
	template = Template(size,nWedges,h0)

	return template.createRetina(pix)


'''
def getRatio():
	return pi * (2*R0 + h0) / (nWedges - pi) / h0

def detRadiiAndHeights():
	R0 = h0 * nWedges / (2*pi) 
	radii = [R0]
	heights = [h0]

	for i in xrange(1,nRings):
		h = pi * (2*radii[i-1] + heights[i-1]) / (nWedges - pi)
		radii.append(h * nWedges / (2*pi))
		heights.append(h)

	return (radii,heights)
'''