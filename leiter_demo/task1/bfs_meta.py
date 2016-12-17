import numpy as np
#from PIL import Image
import csv
import heapq as hq

import similarity2 as sim
import taskpic as tp

search_space_filename = 'search.jpg'
target_filenames = ['target1.jpg',
                    'target2.jpg']
resize_factor = 4    # factor by which resolution is decreased for first pass; speedup is ~O(n^3)   
threshhold = 0.5     # used in determining clear spaces 
priority_queue_size = 1    # how many results are returned after first pass              

def main():
    searchSpace = tp.TaskPic(search_space_filename)
    ssx, ssy = searchSpace.size()

    ssSmall = searchSpace.resize(resize_factor)
    sssx, sssy = ssSmall.size()

    for t in target_filenames:
        target = tp.Target(t,threshhold)
        tx, ty = target.size()

        targetSmall = target.resize(resize_factor)
        tsx, tsy = targetSmall.size()
        
        print(t + ' initialized. Now beginning initial search...\n')
        search(ssSmall,targetSmall,[(0,0,sssx-tsx+1,sssy-tsy+1)],priority_queue_size)

        #refined_search_locations = addLocations(targetSmall.most_sim['alt'],(ssx,ssy),(tx,ty))

        
        refined_search_locations = []
        for key in targetSmall.most_sim.keys():
            temp = [refined_search_locations,addLocations(targetSmall.most_sim[key],(ssx,ssy),(tx,ty))]
            refined_search_locations = set().union(*temp)
        
        
        print('Initial search completed. Now beginning refined search...\n')
        results = search(searchSpace,target,refined_search_locations,1)

        #debug(spArr,'sp'+target.fname.split('.')[0])
        #debug(altArr,'alt'+target.fname.split('.')[0])

        print(t + ' similarities:\n')
        for key in results.keys():
            print('%s:\t%f at %s' % (key,round(results[key][0][0],3),str(results[key][0][1])))
        #print results

        print('Drawing boxes and saving...')
        target.box(searchSpace)

def addLocations(pix,searchSpaceSize,targetSize):
    locs = []
    ssx, ssy = searchSpaceSize
    tx, ty = targetSize

    for p in pix:
        x,y = p
        x *= resize_factor
        y *= resize_factor

        x0 = max(x-resize_factor,0)
        y0 = max(y-resize_factor,0)
        xn = min(x+resize_factor,ssx-tx)
        yn = min(y+resize_factor,ssy-ty)

        if (ssx-tx-xn < resize_factor): # handles literal edge cases where pixels are left out due to rounding troubles
            xn = ssx-tx
        if (ssy-ty-yn < resize_factor): # handles literal edge cases where pixels are left out due to rounding troubles
            yn = ssy-ty

        locs.append((x0,y0,xn+1,yn+1))

    return locs

# locations is a list of "boxes": 4-tuples of form (x0,y0,xn+1,yn+1)
def search(searchSpace,target,locations,queueSize):
    most_sim = {'sp':[],   # format: [(most_sim, (pixel))]
                'mm':[],
                'alt':[]}

    tx, ty = target.size()  
    subSpace = np.empty((tx,ty))

    '''
    ssx, ssy = searchSpace.size()
    spArr = np.empty((ssx-tx+1,ssy-ty+1))
    altArr = np.empty((ssx-tx+1,ssy-ty+1))
    '''
    
    nIterations = 0
    for box in locations:
        nIterations += (box[2] - box[0]) * (box[3] - box[1])
        
    print nIterations

    counter = 0
    for box in locations:
        for sx in xrange(box[0],box[2]):
            for sy in xrange(box[1],box[3]):
                subSpace = searchSpace.pix[sx:sx+tx,sy:sy+ty]
                
                sp = sim.sumProd(target.pix,subSpace)
                #sp = 0
                mm = sim.maxMin(target.pix,subSpace)
                #mm = 0
                a = sim.alt(target.pix,subSpace)

                '''
                if queueSize == 1:
                    print('\n%d:\tsp %f.3, mm %f.3, alt %f.3 at %s' % (counter,sp,mm,a,str((sx,sy))))
                    print('before:')
                    if len(most_sim['sp']) == 0:
                        print('sp:\nmm:\nalt:')
                    else:
                        for key in most_sim.keys():
                            print('%s:\t%s' % (key,str(most_sim[key])))
                              
                if not queueSize == 1:
                    spArr[sx,sy] = sp
                    altArr[sx,sy] = a
                '''

                if counter < queueSize:
                    hq.heappush(most_sim['sp'],(sp,(sx,sy)))
                    hq.heappush(most_sim['mm'],(mm,(sx,sy)))
                    hq.heappush(most_sim['alt'],(a,(sx,sy)))
                else:
                    hq.heappushpop(most_sim['sp'],(sp,(sx,sy)))
                    hq.heappushpop(most_sim['mm'],(mm,(sx,sy)))
                    hq.heappushpop(most_sim['alt'],(a,(sx,sy)))

                '''
                if queueSize == 1:
                    print('\nafter:')
                    for key in most_sim.keys():
                        #print('%s:\t%f at %s' % (key,round(most_sim[key][0],3),str(most_sim[key][1])))
                        print('%s:\t%s' % (key,str(most_sim[key])))
                '''  
                

                if counter % 10000 == 0:
                    print(str(round(100*(float(counter)/nIterations),2)) + r'% done')
                    
                #print(str(counter) + ': ' + str(most_sim['sp']))

                counter += 1

    for key in target.most_sim.keys():
        #print(most_sim[key])
        target.most_sim[key] = [tup[1] for tup in most_sim[key]]

    '''
    if not queueSize == 1:
        debug(spArr,'sp'+target.fname.split('.')[0])
        debug(altArr,'alt'+target.fname.split('.')[0])
    '''

    return most_sim if queueSize == 1 else None

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