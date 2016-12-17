import numpy as np
import csv

trials = 100
n = 100

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
    summation = m1.sum(dtype='float') + m2.sum(dtype='float')
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
    sp = 0.0
    mm = 0.0
    a = 0.0
    spList = ['sum-product']
    mmList = ['max-min']
    aList = ['alt']

    spmmDelta = 0.0
    spaDelta = 0.0
    mmaDelta = 0.0
    spmmDeltaList = ['spmmDelta']
    spaDeltaList= ['spaDelta']
    mmaDeltaList= ['mmaDelta']
    for i in range(trials):
        m1 = randMatrix()
        m2 = randMatrix()

        #print m1, m2
        sp = sumProd(m1,m2)
        mm = maxMin(m1,m2)
        a = alt(m1,m2)

        spList.append(sp)
        mmList.append(mm)
        aList.append(a)

        spmmDelta = sp - mm
        spaDelta = sp - a
        mmaDelta = mm - a

        spmmDeltaList.append(spmmDelta)
        spaDeltaList.append(spaDelta)
        mmaDeltaList.append(mmaDelta)

    with open('simAnalysisSkewed0001.csv', 'w') as file:
        print('test')
        wr = csv.writer(file,delimiter=',')
        wr.writerows([spList,
                 mmList,
                 aList,
                 spmmDeltaList,
                 spaDeltaList,
                 mmaDeltaList])


if __name__ == "__main__":
    main()

'''
    spmmDelta /= trials
    spaDelta /= trials
    mmaDelta /= trials

    print('spmmDelta:\t%f' % spmmDelta)
    print('spaDelta:\t%f' % spaDelta)
    print('mmaDelta:\t%f' % mmaDelta)
    
            for j in range(n**2):
            e1 = m1.item(j)
            e2 = m2.item(j)
            if e1 < 0:
                if e1 < -0.5:
                    e1 = -0.5
                e1 += 1.0
            elif e1 > 0.5:
                e1 = 0.5
            np.put(m1,j,e1)
            
            if e2 < 0:
                if e2 < -0.5:
                    e2 = -0.5
                e2 += 1.0
            elif e2 > 0.5:
                e2 = 0.5
            np.put(m2,j,e2)
'''
