import csv

#import brute_force_sim as bfs
import taskpic

test = taskpic.Target('test_pic3.jpg',0.5)
print(test.nClear,test.threshhold)

with open('test2.csv', 'wb') as file:
    print('test')
    wr = csv.writer(file,delimiter=',')
    wr.writerows(test.pix)