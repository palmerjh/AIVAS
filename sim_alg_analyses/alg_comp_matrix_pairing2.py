import numpy as np
import csv

# NOTE in this iteration, the second matrix is formed based on the first

x_trials = 10
y_trials = 10 
n = 100

# creates random matrix from two-humped distribution skewed towards 0.0 and 1.0
def randM1():
    #return np.matrix(np.random.rand(n,n))
    m1 = np.matrix(np.random.normal(0.0,0.25,n**2))
    for j in range(n**2):
        e = m1.item(j)
        if e < 0:
            if e < -0.5:
                e = -0.5
            e += 1.0
        elif e > 0.5:
            e = 0.5
        np.put(m1,j,e)
            
    return m1
    
def randM2(m1):
    m2 = m1 + np.matrix(np.random.normal(0.0,0.05,n**2))
    for j in range(n**2):
        e = m2.item(j)
        if e < 0:
            np.put(m2,j,0.0)
        elif e > 1:
            np.put(m2,j,1.0)     
            
    return m2

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
        m1 = randM1()
        
        spList = []
        mmList = []
        aList = [] 
        for y in range(y_trials):
            m2 = randM2(m1)
            
            sp = sumProd(m1,m2)
            mm = maxMin(m1,m2)
            a = alt(m1,m2)

            spList.append(sp)
            mmList.append(mm)
            aList.append(a)

        spMat.append(spList)
        mmMat.append(mmList)
        aMat.append(aList)

    with open('simAnalysisMatPairing25_2.csv', 'wb') as file:
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
