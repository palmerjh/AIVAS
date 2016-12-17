import numpy as np

def sumProd(m1,m2):
    summation = np.sum(m1+m2)
    prod = np.sum(m1*m2)

    if summation == 0.0:
        return 0.0

    return 2 * prod / summation

def maxMin(m1,m2):
    minimum = np.sum(np.minimum(m1,m2))
    maximum = np.sum(np.maximum(m1,m2))

    if maximum == 0.0:
        return 1.0

    return minimum / maximum

def alt(m1,m2):

    return 1.0 - np.sum(abs(m1-m2)) / m1.size 

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