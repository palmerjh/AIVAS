import numpy as np
#from PIL import Image
import csv
#import heapq as hq
import sys
sys.path.append(r'C:\Users\palmerjh\Documents\AIVAS')

import pix2retina as p2r
import similarity2 as sim
import taskpic as tp

original_filename = 'test_original.jpg'
rotated_filename = 'test_rotated.jpg'
          

def main():
    original = tp.TaskPic(original_filename)
    original.blur()
    #original.save()
    original.pix2retina()
    
    #original.pix = p2r.main(original.pix)

    rotated = tp.TaskPic(rotated_filename)
    rotated.blur()
    #rotated.save()
    rotated.pix2retina()

    #rotated.pix = p2r.main(rotated.pix)
 
    print('pictures initialized. Now beginning search...\n')
    results = search(original,rotated)

    for key in results.keys():
        print('%s:\t%f at %s' % (key,round(results[key][0],3),str(results[key][1])))

# locations is a list of "boxes": 4-tuples of form (x0,y0,xn+1,yn+1)
def search(original,rotated):
    most_sim = {'sp':[0,0],   # format: [(most_sim, (pixel))]
                'mm':[0,0],
                'alt':[0,0]}

    nWedges, nRings = rotated.size() 

    for i in xrange(nWedges):
        sp = sim.sumProd(rotated.pix,original.pix)
        mm = sim.maxMin(rotated.pix,original.pix)
        a = sim.alt(rotated.pix,original.pix)

        print('%d\tsp: %f\tmm: %f\ta: %f' % (i,sp,mm,a))

        if sp > most_sim['sp'][0]:
            most_sim['sp'] = [sp,i]
        if mm > most_sim['mm'][0]:
            most_sim['mm'] = [mm,i]
        if a > most_sim['alt'][0]:
            most_sim['alt'] = [a,i]

        rotated.rotate()

    return most_sim

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