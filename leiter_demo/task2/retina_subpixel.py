import math
import numpy as np
#import heapq as hq
from random import randrange
from PIL import Image, ImageDraw
from copy import deepcopy

from collections import namedtuple, deque
RelativeRetina = namedtuple("RelativeRetina","ret start") # simple container used to hold a Retina object and its location in its TargetSlice
Result = namedtuple("Result","start wedge_index sim")	# simple container used to hold results for primary,secondary,etc. regions for each choice

pi = math.pi

# used if nRings isn't specified in Template ctor
# if less than 1, is fraction of total_size
# else is the number of pixels
# *****only an approximation
default_blindspot_diameter = 0.2

fix_hn = True	# if False, input hn (default or ctor) is ideal; hn will be shifted lower to get integer number of nWedges and maintain radius of retina as size / 2
				# else input hn is exact and retinal radius and Rn will be lowered to get integer number of nWedges

# used if hn isn't specified in Template ctor
default_hn = 2

# used if threshold isn't specified in TargetSlice ctor
# is fraction of densest region of target
# purpose: discard anomalous blackest regions
default_threshold = 0.75

bs_bias = 2	# weighting of blindspot pixels in density search relative to those in retina

visual_h0 = 2	# how large sectors in the innermost ring should be represented in the visualization

class Template(object):
	"""Performs all the calculations to construct a Retina object but does not actually create one
	   Useful for finding weights before pixels to be converted are known"""

	# initialized in findWeights()
	unrefined_density_wts = None

	# initialized in createVisualizer()
	visualization = None

	def __init__(self, size, nRings=None, hn=None):
		self.size = size 	# side length of square template
		midpoint = (self.size - 1) / 2.0
		self.center = np.array([midpoint,midpoint])

		hn_ideal = default_hn if hn == None else hn
		Rn_ideal = 0.5 * (self.size - hn_ideal)
		nWedges_ideal = 2*pi * Rn_ideal / hn_ideal

		if fix_hn:
			self.hn = hn_ideal
			self.nWedges = int(math.floor(nWedges_ideal))
		else:
			self.nWedges = int(math.ceil(nWedges_ideal))
			self.hn = pi * self.size / (self.nWedges + pi)

		self.Rn = self.nWedges * self.hn / (2*pi)
		self.retina_radius = self.Rn + 0.5*self.hn 		# equals self.size / 2.0 if not fix_hn

		self.ratio = (2*self.Rn - self.hn) / ((1 + self.nWedges/pi) * self.hn)
		self.radius_split = math.log(1 - self.hn/(2*self.Rn),self.ratio)

		self.nRings = 42	# placeholder

		self.bs_radius = -1	# placeholder

		if nRings is None:
			if default_blindspot_diameter < 1:
				bs_radius_ideal = 0.5*(default_blindspot_diameter * self.size)
			else:
				bs_radius_ideal = 0.5*(default_blindspot_diameter)

			self.nRings -= self.findRing(bs_radius_ideal)
		else:
			self.nRings = nRings

		self.bs_radius = self.Rn*math.pow(self.ratio,self.nRings - 1 + self.radius_split)
			
		# weights are 1.0 in retina, 1.0 * bs_bias in blindspot, and 0.0 otherwise
		# used in initial densest search
		#self.findWeights()

		# closure that accepts pix and creates Retina object, closure that accepts pix and calculates density
		self.createRetina, self.calcDensity = self.createRetinaGeneratorAndDensityCalculator()

		# closure that accepts Retina object and visualizes it
		self.visualize = self.createVisualizer()

	def findWeights(self):
		wts = np.ones((self.size,self.size))

		for px in xrange(self.size):
			for py in xrange(self.size):
				p = np.array([px,py])
				diff = p - self.center
				ring = self.findRing(norm(diff))
	
				if (ring >= self.nRings):
					wts[px,py] = 0.0
				elif (ring == -1):
					wts[px,py] *= bs_bias

		self.unrefined_density_wts = wts

	def findRing(self,r):
		#print r
		if (r < self.bs_radius):
			return -1	# inside central, "special" pixel
	
		level = math.log(r/self.Rn,self.ratio)
		if (level < 0):
			if (self.radius_split - 1 < level):
				return self.nRings - 1
			else:
				return self.nRings
	
		level_lb = int(level)
	
		if (level - level_lb < self.radius_split):
			return self.nRings - 1 - level_lb
		else:
			return self.nRings - 1 - (level_lb + 1)

	# locates which wedge/ring sector the pixel is inside
	def findSector(self,pixel):
		p = np.array(pixel)
		diff = p - self.center
		ring = self.findRing(norm(diff))

		angle = math.atan2(diff[1],diff[0])
		if (angle < 0):
			angle += 2*pi

		wedge = int(self.nWedges * angle / (2*pi))

		return wedge, ring

	# returns set of pixels within ring/wedge sector starting with closest pixel to its center (which may or may not actually be in sector)
	def findPixels(self,sector,start):
		pixels = []
		# BFS
		q = deque()
		q.append(start)
		while len(q) > 0:
			cur_pixel = q.popleft()
			if self.findSector(cur_pixel) == sector:
				pixels.append(cur_pixel)
				x,y = cur_pixel
				for pixel in [(x-1,y-1),(x-1,y),(x-1,y+1),
						  	  (x,y-1),          (x,y+1),
						  	  (x+1,y-1),(x+1,y),(x+1,y+1)]:
					if pixel not in q and pixel not in pixels:
						q.append(pixel)

		return pixels

	# for each sector, finds set of pixels within ring/wedge sector unioned with the singleton set of the pixel closest to the sector's center
	# then it uses this to create a closure that generates a retina object
	#
	# for each ring, unions all of the pixels within each of its wedges
	# then it uses this to create a closure that calculates the density
	def createRetinaGeneratorAndDensityCalculator(self):
		sector_pixels = np.empty(self.retina_size(),dtype=list)		# list of pixels in each sector for each sector
		density_ring_mappers = []									# mappers that channel pixels to the correct ring for each ring
		for i in xrange(self.nRings):
			density_ring_mappers.append(set([]))
			
		for wedge in xrange(self.nWedges):			
			central_angle = (wedge + 0.5) / self.nWedges * 2*pi
			unit_vector = np.array([math.cos(central_angle),math.sin(central_angle)])
			
			for ring in xrange(self.nRings):			
				R_ring = self.Rn*math.pow(self.ratio,self.nRings-1-ring)
				sector_center = self.center + R_ring * unit_vector
				closest_pixel = (int(round(sector_center[0])),int(round(sector_center[1])))

				pixels_inside_sector = self.findPixels((wedge,ring),closest_pixel)
				
				sector_pixels[wedge,ring] = pixels_inside_sector
				if closest_pixel not in pixels_inside_sector:
					sector_pixels[wedge,ring].append(closest_pixel)

				density_ring_mappers[ring] = density_ring_mappers[ring].union(sector_pixels[wedge,ring])

		density_bs_mapper = []				# list of pixels that map to blindspot
		for x in xrange(self.size):
			for y in xrange(self.size):
				if dist(np.array([x,y]),self.center) < self.bs_radius:
					density_bs_mapper.append((x,y))

		density_wts = np.ones((self.nRings+1,))
		density_wts[0] = bs_bias

		density_mappers = [density_bs_mapper] + density_ring_mappers

		# closure that creates Retina object
		def createRetina(pix):
			retina = np.empty(self.retina_size(),dtype=list)
			for wedge in xrange(self.nWedges):
				for ring in xrange(self.nRings):
					retina[wedge,ring] = [pix[sector_pixels[wedge,ring][0]]]	# each sector has at least one pixel mapped to it - its closest pixel
					for other_pixel in sector_pixels[wedge,ring][1:]:
						retina[wedge,ring].append(pix[other_pixel])
		
			retina = np.array([[np.average(values) for values in row] for row in retina])
		
			return Retina(self,pix,retina)

		# closure that calculates density
		def calcDensity(retina=None,pixels=None):
			if retina is None:
				pix = pixels
			else:
				pix = retina.pix

			# list comprehension that holds the averages of values of pixels mapping to blindspot ([0]) and ring r ([r+1]) respectively
			# thus len(densities) = 1 + self.nRings
			densities = np.array([np.mean([pix[pixel] for pixel in mapper]) for mapper in density_mappers])

			if pixels is None:
				retina.density_mappers = density_mappers
				retina.densities = densities
				retina.density_wts = density_wts

			return np.average(densities,weights=density_wts)

		return createRetina, calcDensity

	# creates template retina visualizer (using ImageDraw to draw each ring/wedge sector)
	# then it uses this to create a closure that will take retina object info and visualize it
	def createVisualizer(self):
		h0 = self.hn*math.pow(self.ratio,self.nRings-1)
		factor = visual_h0 / h0 	# factor by which retina must be scaled for visualization

		visual_size = int(math.ceil(factor * self.size))
		visual_midpoint = (visual_size - 1) / 2.0
		visual_center = np.array([visual_midpoint,visual_midpoint])
		visual_retina_radius = factor * self.retina_radius

		im = Image.new('RGB',(visual_size,visual_size))
		draw = ImageDraw.Draw(im)

		delta_angle = 2*pi / self.nWedges
		angle = 0
		start = tuple(visual_center)

		# this draws all the "spokes" of the retina
		for i in xrange(self.nWedges):
			end_unit = np.array([math.sin(angle),math.cos(angle)])		# note: coordinates are flipped because draw.line uses diff. coord. system
			end = tuple(visual_center + visual_retina_radius*end_unit)

			draw.line([start,end],fill='red')
			angle += delta_angle

		# this draws the circles that partition the retina into rings
		for ring in xrange(-1,self.nRings):
			radius = factor * self.Rn*math.pow(self.ratio,ring + self.radius_split)
			xy = (visual_midpoint - radius, visual_midpoint - radius,
				  visual_midpoint + radius, visual_midpoint + radius)	# bounding box for circular ring partition

			if ring < self.nRings - 1:
				draw.ellipse(xy,outline='red')
			else:
				draw.ellipse(xy,fill='black',outline='red')	# erases extraneous wedge lines that clutter the blindspot

		self.visualization = im

		visual_pix = np.asarray(im)
		mapper = np.empty((visual_size,visual_size),dtype=list)		# initializes each element to None super quickly; 1000 times faster than using fill()
		for x in xrange(visual_size):
			for y in xrange(visual_size):
				if np.array_equal(visual_pix[x,y],[255,0,0]):
					continue

				p = np.array([x,y])
				diff = p - visual_center
				dist = norm(diff)

				if dist > visual_retina_radius:
					continue

				ring = self.findRing(dist / factor)

				if ring == -1:	# inside blindspot
					continue
		
				angle = math.atan2(diff[1],diff[0])
				if (angle < 0):
					angle += 2*pi
		
				wedge = int(self.nWedges * angle / (2*pi))

				mapper[x,y] = (wedge,ring)

		def visualize(retina,fname=None):
			visual_pix.flags.writeable = True	# need this for some reason

			r_pix = retina.retina.copy()

			# converts from our faux grayscale values to "real" ones from 0 to 255 (unrounded)
			r_pix = (1.0 - r_pix) * 255
	
			# converts each grayscale value v to RGB list [v,v,v]
			r_pix = np.array([[[x,x,x] for x in [int(round(y)) for y in z]] for z in r_pix])

			for x in xrange(visual_size):
				for y in xrange(visual_size):
					index = mapper[x,y]
					if index is None:
						continue

					visual_pix[x,y] = r_pix[index]

			retina.visualization = Image.fromarray(visual_pix.astype('uint8'))

			if not fname is None:
				retina.visualization.save(fname)

		return visualize

	def printRings(self):
		for i in xrange(self.nRings):
			print('%d\t%f\t%f' % (self.nRings-1-i,self.Rn*math.pow(self.ratio,i),self.hn*math.pow(self.ratio,i)))

	# Note: pix must be of dimension self.size by self.size (same as self.unrefined_density_wts)
	# does not distinguish between various pixel densities
	# good for initial densest search in TargetSlice; faster than refined ring-based density calculation
	# bad for later density comparison search in choices - use self.calcDensity(pixels=pix) for that
	def calcUnrefinedDensity(self,pix):
		return np.average(pix,weights=self.unrefined_density_wts)
		'''
		return np.mean([np.average(1-pix,weights=self.wts),
						np.average(pix,weights=self.inverse_wts)])
		'''

	def retina_size(self):
		return (self.nWedges,self.nRings)


# Note: requires template object to create
class Retina(object):
	"""Numpy array of pixels arranged in polar fashion"""

	# initialized or updated whenever self.visualize is called
	# PIL.Image object
	visualization = None

	density = -1				# ring-based; calculated only if explicity asked to with getDensity() 
	unrefinedDensity = -1		# calculated only if explicity asked to with getUnrefinedDensity()

	# -------------------------------------------------------------------------------------------------
	# the following three arrays are instantiated when self.getDensity() is called for the first time
	# used in calculating self.ringwiseDensityDelta(pix2)

	# density_mappers[0] is density_bs_mapper and density_mappers[r+1] is mapper for ring r
	density_mappers = None
	# densities[0] is bs_density and densities[r+1] is density of ring r
	densities = None
	density_wts = None
	# -------------------------------------------------------------------------------------------------

	def __init__(self, template, pix, retina):
		self.template = template
		self.pix = pix
		self.retina = retina

	# if pix is not specified, returns deep copy
	# else returns new Retina object with same template but different pix
	def copy(self,pix=None):
		if pix == None:
			return deepcopy(self)
		else:
			return self.template.createRetina(pix)

	def getDensity(self):
		if self.density == -1:
			self.density = self.template.calcDensity(retina=self)

		return self.density

	def getUnrefinedDensity(self):
		if self.unrefinedDensity == -1:
			self.unrefinedDensity = self.template.calcUnrefinedDensity(self.pix)

		return self.unrefinedDensity

	def findRing(self,r):
		return self.template.findRing(r)

	def findSector(self,pixel):
		return self.template.findSector(pixel)

	def printRings(self):
		return self.template.printRings()

	def pix_size(self):
		return self.pix.shape

	def retina_size(self):
		return self.template.retina_size()

	# calculates the density deltas of the bs & ring regions of self.pix and another pix2 and averages based on self.density_wts
	def ringwiseDensityDelta(self,pix2):
		# list comprehension that holds the averages of values of pix2 pixels mapping to blindspot ([0]) and ring r ([r+1]) respectively
		# thus len(pix2_densities) = 1 + self.template.nRings
		pix2_densities = np.array([np.mean([pix2[pixel] for pixel in mapper]) for mapper in self.density_mappers])

		return np.average(abs(self.densities - pix2_densities),weights=self.density_wts)

	def rotate(self,nRotations=None):
		if nRotations == None:
			self.retina = np.roll(self.retina, 1, axis=0)
		else:
			self.retina = np.roll(self.retina, nRotations, axis=0) 

	# saves and returns retina as PIL.Image object
	def save(self,fname):
		temp = (1.0 - self.retina) * 255
		temp = np.array([[int(round(value)) for value in row] for row in temp])
		im = Image.fromarray(temp.astype('uint8'))

		im.save(fname)

		return im

	def visualize(self,fname=None):
		self.template.visualize(self,fname)

class TargetSlice(object):
	"""Collection of Retina Objects"""
	def __init__(self, pix, t_name, threshold=None, size=None, nRings=None, hn=None):
		self.pix = pix
		self.t_name = t_name
		self.threshold = default_threshold if threshold is None else threshold
		self.size = min(self.pix.shape) if size is None else size

		self.template = Template(self.size,nRings,hn)

		# ordered by density; all entries are < self.threshold * highest_density to protect against anomalous blackest regions
		self.ordered_unrefined_densities = self.findUnrefinedDensities()	# format: (density,start)

		self.primary = self.createRelativeRetina(0)
		p_start = self.primary.start

		secondary_index = 1
		s_start = np.array(self.ordered_unrefined_densities[secondary_index][1])
		sp_distance = dist(p_start,s_start)
		while not self.goldilocks(sp_distance):
			secondary_index += 1
			s_start = np.array(self.ordered_unrefined_densities[secondary_index][1])
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

	# i is the index of self.ordered_unrefined_densities at which the RelativeRetina container will be created
	def createRelativeRetina(self,i):
		unrefinedDensity, start = self.ordered_unrefined_densities[i]
		sx, sy = start

		ret = self.template.createRetina(self.pix[sx:sx+self.size,sy:sy+self.size])
		ret.unrefinedDensity = unrefinedDensity

		ret.getDensity()

		return RelativeRetina(ret,np.array(start))

	def findUnrefinedDensities(self):
		unrefinedDensities = []
		for sx in xrange(self.pix.shape[0] - self.size + 1):
			for sy in xrange(self.pix.shape[1] - self.size + 1):
				unrefinedDensities.append((self.template.calcUnrefinedDensity(self.pix[sx:sx+self.size,sy:sy+self.size]),(sx,sy)))

		unrefinedDensities.sort(reverse=True)

		cutOff = -1
		for i in xrange(len(unrefinedDensities)):
			if unrefinedDensities[i][0] < self.threshold*unrefinedDensities[0][0]:
				cutOff = i
				break

		return unrefinedDensities[cutOff:]

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

def calcDensity(pix,nRings=None,hn=None,unrefined=True):
	size,size = pix.shape
	if unrefined:
		return Template(size,nRings,hn).calcUnrefinedDensity(pix)
	else:
		return Template(size,nRings,hn).calcDensity(pixels=pix)

def pix2ret(pix,nRings=None,hn=None):
	size,size = pix.shape
	return Template(size,nRings,hn).createRetina(pix)

def dist(v1,v2):
	#return math.sqrt(math.pow(v1[0]-v2[0],2) + math.pow(v1[1]-v2[1],2))
	return norm(v1 - v2)

def norm(v):
	return np.sqrt((v*v).sum(axis=0))

def createRetina(file,nRings=None,hn=None):
	im = Image.open(file).convert('L')
	pix = 1.0 - np.asarray(im) / 255.0

	size = min(pix.shape)
	template = Template(size,nRings,hn)

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