import numpy as np
import similarity2 as sim

x = 200
y = 150

m = np.asfortranarray(np.random.rand(10*x,10*y))
n = np.asfortranarray(np.random.rand(x,y))
#m = np.random.rand(10*x,10*y)
#n = np.random.rand(x,y)
for sx in xrange(150):
    for sy in xrange(150):
        m_sub = np.array(m[sx:sx+x,sy:sy+y])
    
        #jojo = np.sum(m[i:i+500,i:i+500]*n)
        jojo = sim.sumProd(m_sub,n)
