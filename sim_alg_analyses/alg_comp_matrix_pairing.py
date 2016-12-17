import numpy as np
import csv

x_trials = 10
y_trials = 10 
n = 4

# creates random matrix from two-humped distribution skewed towards 0.0 and 1.0
def randMatrix():
    #return np.matrix(np.random.rand(n,n))
    m = np.matrix(np.random.normal(0.0,0.25,n**2))
    for j in range(n**2):
        e = m.item(j)
        if e < 0:
            if e < -0.5:
                e = -0.5
            e += 1.0
        elif e > 0.5:
            e = 0.5
        np.put(m,j,e)
            
    return m

def sumProd(m1,m2):
    summation = (m1.sum(dtype='float') + m2.sum(dtype='float'))/2
    if summation == 0.0:
        return 1.0
    prod = 1
    for i in range(n**2):
        prod += m1.item(i) * m2.item(i) 

    return prod / summation

def maxMin(m1,m2):
    maximum = 0.0
    minimum = 0.0
    for i in range(n**2):
        maximum += max(m1.item(i),m2.item(i))
        minimum += min(m1.item(i),m2.item(i))

    return minimum / maximum

def alt(m1,m2):
    absError = 0.0
    for i in range(n**2):
        absError += abs(m1.item(i) - m2.item(i))

    return 1.0 - absError / n**2 

def main():
    spMat = [[]]
    mmMat = [[]]
    aMat = [[]]

    for x in range(x_trials):
        m1 = randMatrix()
        
        spList = []
        mmList = []
        aList = [] 
        for y in range(y_trials):
            m2 = randMatrix()
            
            sp = sumProd(m1,m2)
            mm = maxMin(m1,m2)
            a = alt(m1,m2)

            spList.append(sp)
            mmList.append(mm)
            aList.append(a)

        spMat.append(spList)
        mmMat.append(mmList)
        aMat.append(aList)

    with open('simAnalysisSmallMatPairing25.csv', 'wb') as file:
        print('test')
        wr = csv.writer(file,delimiter=',')
        wr.writerows(spMat + [[]] + mmMat + [[]] + aMat)

if __name__ == "__main__":
    main()

'''
    spmmDelta /= trials
    spaDelta /= trials
    mmaDelta /= trials

    print('spmmDelta:\t%f' % spmmDelta)
    print('spaDelta:\t%f' % spaDelta)
    print('mmaDelta:\t%f' % mmaDelta)
'''
