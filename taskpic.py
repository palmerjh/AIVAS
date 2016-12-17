import numpy as np
from PIL import Image, ImageFilter
import csv
import copy

from collections import deque

import sys
sys.path.append(r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\task2')
import retina_subpixel as r

box_line_width = 1

class TaskPic(object):
	def __init__(self,fname,im=None):
		self.fname = fname
		if im is None:
			self.im = Image.open(self.fname).convert('L')
		else:
			self.im = im
		self.pix = self.im2pix()

	def size(self):
		return self.pix.shape

	# Returns new TaskPic object resized using PIL.Image.resize() 
	def resize(self,factor):
		x,y = self.size()
		new_x = int(round(float(x) / factor))
		new_y = int(round(float(y) / factor))

		name, ext = self.fname.split('.')

		resized = copy.deepcopy(self)

		resized.fname = name+'_small.'+ext
		resized.im = resized.im.resize((new_y,new_x))
		resized.pix = resized.im2pix()

		return resized

		'''
		new_im = self.pix2im().resize((new_y,new_x))
		#new_im.show()

		return TaskPic(name+'_small.'+ext,self.im.resize((new_y,new_x)))
		'''

	# updates self.im and saves
	def save(self,fname=None):
		#self.im = self.pix2im()
		if fname is not None:
			self.fname = fname

		self.im.save(self.fname)

	def im2pix(self):
		#print self.size()
		return 1.0 - np.asarray(self.im) / 255.0

	def pix2im(self):
		temp = (1.0 - self.pix) * 255
		temp = np.array([[int(round(value)) for value in row] for row in temp])
		return Image.fromarray(temp.astype('uint8'))

	# modifies self.pix; rectangular (pix) --> polar (retina)
	def pix2retina(self,nWedges=None,h0=None):
		name, ext = self.fname.split('.')
		self.fname = name + '_retina.' + ext
		size = min(self.size())
		template = r.Template(size,nWedges,h0)
		self.pix = template.createRetina(self.pix).retina

	# modifies self.pix AND self.im; blurs image
	def blur(self):
		filter = ImageFilter.BLUR
		#filter = ImageFilter.GaussianBlur(radius=2)
		self.im = self.im.filter(filter)

		name, ext = self.fname.split('.')
		self.fname = name + '_blur.' + ext
		self.pix = self.im2pix()

	# modifies self.pix; rotates object if already retina
	def rotate(self,n=None):
		if n is None:
			self.pix = np.roll(self.pix, 1, axis=0)
		else:
			self.pix = np.roll(self.pix, n, axis=0)

	# returns deep copy
	def copy(self):
		return copy.deepcopy(self)

	# updates self.im from self.pix
	def update(self):
		self.im = self.pix2im()

	# returns densest square with w = l = size as new TaskPic object
	def densest_square(self,size):
		x,y = self.size()
		most_dense = (0,(0,0))
		#square = np.empty((size,size))
		for sx in range(x-size+1):
			for sy in range(y-size+1):
				'''
				square = self.pix[sx:sx+size,sy:sy+size]
				density = np.mean(square)
				'''
				density = np.mean(self.pix[sx:sx+size,sy:sy+size])
				if density > most_dense[0]:
					most_dense = (density,(sx,sy))
		print most_dense

		r_density, r_start = most_dense
		rx, ry = r_start

		return (self.pix[rx:rx+size,ry:ry+size], r_density)

class Target(TaskPic):
	"""Object to be found in Leiter Task 1"""
	most_sim = {'sp':[],   # format: [(pixel)]
                'mm':[],
                'alt':[]}
	def __init__(self,fname,threshhold,im=None):
		super(Target,self).__init__(fname,im)
		self.threshhold = threshhold
		#self.nClear = self.findClear()

	'''
	# Returns new Target object resized using PIL.Image.resize() 
	def resize(self,factor):
		x,y = self.size()
		new_x = int(round(float(x) / factor))
		new_y = int(round(float(y) / factor))

		#print new_x,new_y

		name, ext = self.fname.split('.')

		new_im = self.im.resize((new_y,new_x))
		#new_im.show()

		return Target(name+'_small.'+ext,self.threshhold,self.im.resize((new_y,new_x)))
	'''

	def findClear(self):
		#return self.helper(0,0) - DFS

		# BFS
		size = self.size()
		q = deque()
		q.append((0,0))
		nClear = 0
		while len(q) > 0:
			x,y = q.popleft()			
			val = self.pix[x,y]
			
			if val > self.threshhold:
				continue
			
			self.pix[x,y] = 42.0
			nClear += 1

			'''
			for nx in range(x-1,x+2):
				if self.inside(nx,xMax):
					for ny in range(y-1,y+2):
						nxy = (nx,ny)
						
						if nxy == (x,y):
							continue

						if self.inside(ny,yMax) and nxy not in q:
							q.append(nxy)
			'''
			for pixel in [(x,y-1),(x,y+1),(x-1,y),(x+1,y)]:
				if self.inside(pixel,size) and pixel not in q:
					q.append(pixel)

		return nClear

	def inside(self,pixel,size):
		return pixel[0] >= 0 and pixel[0] < size[0] and pixel[1] >= 0 and pixel[1] < size[1]
	'''
	def inside(self,val,max):
		return val >= 0 and val < max
	'''
	# Places boxes around "found" objects; color corresponds to algorithm used
	# 	-sum-prod: red
	# 	-min-max: green
	# 	-alt: blue
	# Creates new version of original search space
	def box(self,tp):
		boxed_tp = tp.pix.copy()

		# converts from our faux grayscale values to "real" ones from 0 to 255 (unrounded)
		boxed_tp = (1.0 - boxed_tp) * 255

		# converts each grayscale value v to RGB list [v,v,v]
		boxed_tp = np.array([[[x,x,x] for x in [int(round(y)) for y in z]] for z in boxed_tp])

		self.drawBox(boxed_tp,self.most_sim['sp'][0],0)	# draws red box for sum-prod result 
		#self.drawBox(boxed_tp,self.most_sim['mm'][0],1)	# draws green box for min-max result
		self.drawBox(boxed_tp,self.most_sim['alt'][0],2)	# draws blue box for alt result

		#self.debug(boxed_tp,'s1' + self.fname.split('.')[0])

		solved_im = Image.fromarray(boxed_tp.astype('uint8'))

		tp_fname, ext = tp.fname.split('.')
		solved_fname = '%s_%s.%s' % (tp_fname,self.fname.split('.')[0],ext) # origName_targetName.ext

		solved_im.save(solved_fname)

		print('Done!')

	def drawBox(self,boxed_tp,start,color):
		sx,sy = start
		tx,ty = self.size()
		blw = box_line_width

		self.drawLine(boxed_tp,start,(blw,ty),color)						# top
		self.drawLine(boxed_tp,(sx+blw,sy),(tx-2*blw,blw),color)			# left side
		self.drawLine(boxed_tp,(sx+tx-blw,sy),(blw,ty),color)				# bottom
		self.drawLine(boxed_tp,(sx+blw,sy+ty-blw),(tx-2*blw,blw),color)		# right side

	def drawLine(self,boxed_tp,start,dim,color):
		sx,sy = start
		dx,dy = dim

		drgb = np.array([0]*3) # delta RGB
		drgb[color] = 255

		for x in range(sx,sx+dx):
			for y in range(sy,sy+dy):
				rgb = np.array(boxed_tp[x,y])

				# if True, means pixel has been colored before, so don't clear RGB entries
				if rgb[2] == 0 and not np.array_equal(rgb,[0,0,0]):
					pass
				else:
					rgb *= 0
				rgb += drgb
				boxed_tp[x,y] = rgb

	def debug(self,pix,fname):
		with open(fname + '.csv', 'wb') as file:
			wr = csv.writer(file,delimiter=',')
			wr.writerows(pix)

'''
	# DFS
	def helper(self,x,y):
		if self.outside(x,y):
			#print((x,y,'outside'))
			return 0
		
		val = self.pix[x,y]
		#if val == 42.0:
		#	return 0
		if val > self.threshhold:
			#print((x,y,val,'>t'))
			return 0
		
		self.pix[x,y] = 42.0
		nClear = 1

		nClear += self.helper(x-1,y-1)
		nClear += self.helper(x-1,y)
		nClear += self.helper(x-1,y+1)

		nClear += self.helper(x,y-1)
		nClear += self.helper(x,y+1)

		nClear += self.helper(x+1,y-1)
		nClear += self.helper(x+1,y)
		nClear += self.helper(x+1,y+1)

		return nClear

	def outside(self,x,y):
		xMax, yMax = self.pix.shape
		return x < 0 or x >= xMax or y < 0 or y >= yMax
'''