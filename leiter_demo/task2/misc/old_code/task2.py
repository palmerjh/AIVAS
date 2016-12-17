#import numpy as np
#from PIL import Image
import csv
import heapq as hq
import os
import sys
sys.path.append(r'C:\Users\palmerjh\Documents\AIVAS')

import similarity2 as sim
import taskpic as tp
#import pix2retina as p2r
import retina as r

square_size = 35
priority_queue_size = 1000

choices_dir = 'choices'
targets_dir = 'targets' 

template = r.Template(square_size)      

def main():

    print('Initializing task elements...')
    choices = [tp.TaskPic(os.getcwd() + '\\%s\\%s' % (choices_dir,c)) for c in os.listdir(choices_dir)]
    targets = [tp.TaskPic(os.getcwd() + '\\%s\\%s' % (targets_dir,t)) for t in os.listdir(targets_dir)]
    print('...DONE!\n')

    for t in targets:
        # TODO identify slices
        print(t.fname + ':\n')

        print('Identifying densest sqaure...')
        rot_ret,r_start = densest_square(t) # returns retina for rotated target
        print('...DONE!\n')

        print('Starting density search...')
        locations = sim_density_search(choices,rot_ret,priority_queue_size)
        #print locations
        print('...DONE!\n')

        print('Starting rotational search...')
        result = search(choices,rot_ret,locations) # format: [sim, location, wedge_index]
        print('...DONE!\n')

        local_t_fname = t.fname.split('\\')[-1] if '\\' in t.fname else t.fname 
        local_c_fname = choices[result[1][0]].fname.split('\\')[-1] if '\\' in choices[result[1][0]].fname else choices[result[1][0]].fname
        print('%s densest at pixel %s with density %f:' % (local_t_fname,str(r_start),rot_ret.density))
        print('\t-Location: choice %d (AKA %s) at pixel %s' % (result[1][0],local_c_fname,str((result[1][1:]))))
        print('\t-Rotation: wedge %d out of %d, which corresponds to the target being rotated %.2f degrees anti-clockwise' 
            % (result[2],template.nWedges,round(float(result[2])/template.nWedges * 360, 2))) 
        print('\t-Similarity: %f' % result[0])

# searches through queue of possibilities, invoking a rotational search for each
# locations is a list of regions to investigate: 3-tuples of form (choice_index, sx, sy)
def search(choices,rot_ret,locations):
    most_sim = [0,(0,0,0),0]   # format: [most_sim, location, wedge_index]
    rx, ry = rot_ret.pix_size()

    nIterations = len(locations)

    counter = 0
    for loc in locations:
        c_index, sx, sy = loc

        # creates subspace specified by loc and transforms it to retina-form
        goal_ret = template.createRetina(choices[c_index].pix[sx:sx+rx,sy:sy+ry])
        
        most_sim_loc = rotational_search(rot_ret,goal_ret)    # format: [most_sim, wedge_index] 

        if most_sim_loc[0] > most_sim[0]:
            most_sim = [most_sim_loc[0],loc,most_sim_loc[1]]    

        if counter % 50 == 0:
            print(str(round(100*(float(counter)/nIterations),2)) + r'% done') 

        counter += 1

    return most_sim

def rotational_search(rot_ret,goal_ret):
    most_sim = [0,0] # format: [most_sim, wedge_index]

    for wedge_index in xrange(template.nWedges):

        mm = sim.maxMin(rot_ret.retina,goal_ret.retina)
        if mm > most_sim[0]:
            most_sim = [mm,wedge_index]

        rot_ret.rotate()

    return most_sim

# searches through all the choices for areas of similar density to rot_ret.density
def sim_density_search(choices,rot_ret,queueSize):
    most_sim = []   # format: [(1.0 - density, (choice_index, sx, sy))]

    size = template.size
    #subSpace = np.empty((rx,ry))
    
    nIterations = 0
    for c in choices:
        cx,cy = c.size()
        nIterations += (cx-size+1) * (cy-size+1)
        
    print nIterations

    counter = 0
    choice_index = 0
    for c in choices:
        cx,cy = c.size()
        for sx in xrange(cx-size+1):
            for sy in xrange(cy-size+1):
                '''
                subSpace = np.array(c.pix[cx:cx+rx,cy:cy+ry])
                density = np.mean(subSpace)
                '''
                density = template.calcDensity(c.pix[sx:sx+size,sy:sy+size])
                delta = abs(density - rot_ret.density)

                '''
                if counter % 200 == 0 or counter % 200 == 1:
                    print('cur_density: %f\ndelta: %f\n1.0 - delta: %f' % (density,delta,1.0-delta))
                    print('\nbefore:')
                    for elem in most_sim:
                        print elem
                '''
                
                if counter < queueSize:
                    hq.heappush(most_sim,(1.0 - delta,(choice_index,sx,sy)))
                else:
                    hq.heappushpop(most_sim,(1.0 - delta,(choice_index,sx,sy)))

                '''
                if counter % 200 == 0 or counter % 200 == 1:
                    print('\nafter:')
                    for elem in most_sim:
                        print elem
                '''

                if counter % 50000 == 0:
                    print(str(round(100*(float(counter)/nIterations),2)) + r'% done')
                    
                #print(str(counter) + ': ' + str(most_sim['sp']))

                counter += 1
        choice_index += 1

    '''
    for elem in most_sim:
        print elem
    '''

    return [tup[1] for tup in most_sim]

    return most_sim

# returns Retina object representing densest square
def densest_square(t):
    x,y = t.size()
    most_dense = (0,(0,0))
    #square = np.empty((size,size))

    for sx in range(x-square_size+1):
        for sy in range(y-square_size+1):
            '''
            square = self.pix[sx:sx+size,sy:sy+size]
            density = np.mean(square)
            '''
            density = template.calcDensity(t.pix[sx:sx+square_size,sy:sy+square_size])
            if density > most_dense[0]:
                most_dense = (density,(sx,sy))

    print most_dense
    r_density, r_start = most_dense
    rx, ry = r_start

    return template.createRetina(t.pix[rx:rx+square_size,ry:ry+square_size]),r_start

def debug(pix,fname):
    with open(fname + '.csv', 'wb') as file:
        wr = csv.writer(file,delimiter=',')
        wr.writerows(pix)

'''
# needed in case of equal first column values in heapq
class FirstList(list):
    def __lt__(self, other):
        return self[0] < other[0]   
'''

if __name__ == "__main__":
    main()

# using all of the similarity heuristics:

'''
def rotational_search(rotated,goal):

    most_sim = {'sp':[0,0],   # format: [most_sim, wedge_index]
                'mm':[0,0],
                'alt':[0,0]}

    nWedges, nRings = rotated.shape

    for wedge_index in xrange(nWedges):

        sp = sim.sumProd(rotated.pix,original.pix)
        mm = sim.maxMin(rotated.pix,original.pix)
        a = sim.alt(rotated.pix,original.pix)

        #print('%d\tsp: %f\tmm: %f\ta: %f' % (i,sp,mm,a))
        
        if sp > most_sim['sp'][0]:
            most_sim['sp'] = [sp,i]
        if mm > most_sim['mm'][0]:
            most_sim['mm'] = [mm,i]
        if a > most_sim['alt'][0]:
            most_sim['alt'] = [a,i]

        rotated = np.roll(rotated, 1, axis=0)

    return most_sim
'''