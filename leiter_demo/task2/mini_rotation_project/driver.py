import numpy as np
from PIL import ImageDraw
import matplotlib.pyplot as plt
import csv
import heapq as hq
import os
import sys
sys.path.append(r'C:\Users\palmerjh\Documents\AIVAS')
import cProfile
import time

import math
pi = math.pi

import similarity2 as sim
import taskpic as tp
#import pix2retina as p2r
import retina_efficient as r

search_size = 125            # size of squares to search for in choices
results_dir = os.getcwd() + '\\results\\sumProd_hn=1'
nSamples = 6                 # other than the "best" rotation
input_dir = os.getcwd() + '\\inputs'
#template = r.Template(square_size)

def main():
    '''
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    '''

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    #start_t = time.time()
    print "Creating template..."
    template = r.Template(search_size)
    print "...finished"

    dirs = os.listdir(input_dir)
    for directory in dirs:
        '''
        script_name = os.path.basename(sys.argv[0])
        print script_name
        if directory == script_name:
            continue
        '''
        print "\nStarting rotation analysis for %s..." % directory
        input_path = input_dir + '\\%s' % directory
        results_path = results_dir + '\\%s' % directory
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # should be two different files that are to be analyzed
        image_files = [(input_path + '\\%s' % im) for im in os.listdir(input_path)]

        fixed = template.createRetina(r.file2pix(image_files[0]))
        rotated = template.createRetina(r.file2pix(image_files[1]))

        # saves what fixed retina looks like in polar coordinates
        fixed.save(results_path + '\\fixed_polar_retina.jpg')

        # saves what fixed retina looks like re-visualized in rect coordinates
        fixed.visualize(results_path + '\\fixed_visualized_retina.jpg')

        nWedges = template.nWedges
        sample_separation = float(nWedges) / nSamples
        samples = [int(math.floor(i * sample_separation)) for i in xrange(nSamples)]

        similarities = []

        rotated_samples_dir = results_path + '\\rotated_samples'
        if not os.path.exists(rotated_samples_dir):
            os.makedirs(rotated_samples_dir)

        most_sim = {'value': 0, 'wedge_index': 0, 'retina': None}
        for wedge_index in xrange(nWedges):
            # calculates similarity
            sp = sim.sumProd(fixed.retina,rotated.retina)
            similarities.append(sp)

            if sp > most_sim['value']:
                most_sim = {'value': sp, 'wedge_index': wedge_index, 'retina': rotated.copy()}

            if wedge_index in samples:
                # saves what rotated retina looks like in polar coordinates
                rotated.save(rotated_samples_dir + '\\polar_retina_wedge_%d.jpg' % wedge_index)

                # saves what rotated retina looks like re-visualized in rect coordinates
                rotated.visualize(rotated_samples_dir + '\\visualized_retina_wedge_%d.jpg' % wedge_index)

            rotated.rotate()

        # creates excel file with all of the similarities by rotation
        data2excel(similarities, results_path + '\\similarities')

        with open(results_path + '\\most_sim.txt', 'w') as f:
            f.write('Greatest similarity: %f\r\n' % most_sim['value'])
            f.write('Rotation: wedge %d out of %d, which corresponds to the\r\nrotating image being rotated %.2f degrees anti-clockwise\r\n'
                % (most_sim['wedge_index'],nWedges,round(float(most_sim['wedge_index'])/nWedges * 360, 2)))

        # saves what rotated retina looks like in polar coordinates at best match
        most_sim['retina'].save(results_path + '\\polar_retina_most_sim.jpg')

        # saves what rotated retina looks like re-visualized in rect coordinates at best match
        most_sim['retina'].visualize(results_path + '\\visualized_retina_most_sim.jpg')

        print "...finished"

def data2excel(data,fname):
    with open(fname + '.csv', 'wb') as file:
        wr = csv.writer(file,delimiter=' ')
        for elem in data:
            wr.writerow([elem])

if __name__ == "__main__":
    #cProfile.run('main()', 'task2_efficient_stats_1')
    main()
