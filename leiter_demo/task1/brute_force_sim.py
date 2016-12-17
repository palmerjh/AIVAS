import numpy as np
from PIL import Image
import csv

import similarity as sim
import taskpic as tp

search_space_filename = 'test_search.jpg'
target_filenames = ['test_target1.jpg',
                    'test_target2.jpg']
threshhold = 0.5     # used in determining clear spaces                    

# Takes image file and converts it into 2D numpy array
#   -Elements of array are grayscale: 0.0 (white) to 1.0 (black)
def file2pix(fname):
    return 1.0 - np.asarray(Image.open(fname).convert('L')) / 255.0

def main():
    searchSpace = tp.TaskPic(search_space_filename)
    ssx, ssy = searchSpace.size()

    for t in target_filenames:
        target = tp.Target(t,threshhold)
        print(t + ' initialized. now searching...\n')
        tx, ty = target.size()

        most_sim = {'sp':[0.0,(0,0)],   # format: [most_sim, (pixel)]
                    'mm':[0.0,(0,0)],
                    'alt':[0.0,(0,0)]}

        subSpace = np.empty((tx,ty))
        #spArr = np.empty((ssx-tx+1,ssy-ty+1))
        #altArr = np.empty((ssx-tx+1,ssy-ty+1))

        nIterations = (ssx-tx+1)*(ssy-ty+1)
        counter = 0
        for sx in range(ssx-tx+1):
            for sy in range(ssy-ty+1):
                subSpace = np.array(searchSpace.pix[sx:sx+tx,sy:sy+ty])
                
                sp = sim.sumProd(target.pix,subSpace)
                mm = sim.maxMin(target.pix,subSpace)
                a = sim.alt(target.pix,subSpace)

                #spArr[sx,sy] = sp
                #altArr[sx,sy] = a

                if sp > most_sim['sp'][0]:
                    most_sim['sp'] = [sp,(sx,sy)]
                if mm > most_sim['mm'][0]:
                    most_sim['mm'] = [mm,(sx,sy)]
                if a > most_sim['alt'][0]:
                    most_sim['alt'] = [a,(sx,sy)]

                counter += 1 

                if counter % 500 == 0:
                    print(str(round(100*(float(counter)/nIterations),1)) + r'% done')

        target.sp = most_sim['sp'][1]
        target.mm = most_sim['mm'][1]
        target.alt = most_sim['alt'][1]

        #debug(spArr,'sp'+target.fname.split('.')[0])
        #debug(altArr,'alt'+target.fname.split('.')[0])

        print(t + ' similarities:\n')
        print('sp: ' + str(most_sim['sp']))
        print('mm: ' + str(most_sim['mm']))
        print('alt: ' + str(most_sim['alt']) + '\n')

        print('Drawing boxes and saving...')
        target.box(searchSpace)

def debug(pix,fname):
    with open(fname + '.csv', 'wb') as file:
        wr = csv.writer(file,delimiter=',')
        wr.writerows(pix)


if __name__ == "__main__":
    main()