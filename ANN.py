import numpy as np
import matplotlib.pylab as plt
x=np.array(1,4,0.1)
f=1/(1+np.exp(-x))
plt.plot(x,f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()