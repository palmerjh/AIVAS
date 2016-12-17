import numpy as np

def sumProd(m1,m2):
    summation = 0.0
    prod = 1.0

    for e1,e2 in np.nditer([m1,m2]):
        # clear pixel - ignore
        if e1 == 42.0 or e2 == 42.0:
            continue

        summation += e1 + e2
        prod += e1 * e2

    if summation == 0.0:
        return 0.0

    return 2 * prod / summation

def maxMin(m1,m2):
    maximum = 0.0
    minimum = 0.0
    for e1,e2 in np.nditer([m1,m2]):
        # clear pixel - ignore
        if e1 == 42.0 or e2 == 42.0:
            continue

        maximum += max(e1,e2)
        minimum += min(e1,e2)

    if maximum == 0.0:
        return 1.0

    return minimum / maximum

def alt(m1,m2):
    absError = 0.0
    nRelevantPix = 0
    for e1,e2 in np.nditer([m1,m2]):
        # clear pixel - ignore
        if e1 == 42.0 or e2 == 42.0:
            continue

        nRelevantPix += 1
        absError += abs(e1 - e2)

    if absError == 0.0:
        return 1.0

    return 1.0 - absError / nRelevantPix 

'''
def sumProd(m1,m2):
    summation = (m1.sum(dtype='float') + m2.sum(dtype='float')) / 2.0
    if summation == 0.0:
        return 1.0
    prod = 1
    for i in range(m1.size):
        prod += m1.item(i) * m2.item(i) 

    return prod / summation

def maxMin(m1,m2):
    maximum = 0.0
    minimum = 0.0
    for i in range(m1.size):
        maximum += max(m1.item(i),m2.item(i))
        minimum += min(m1.item(i),m2.item(i))

    return minimum / maximum

def alt(m1,m2):
    absError = 0.0
    for i in range(m1.size):
        absError += abs(m1.item(i) - m2.item(i))

    return 1.0 - absError / m1.size 
'''