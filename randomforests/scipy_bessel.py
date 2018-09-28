'''
Python script to compute the maximum of a Bessel function
'''

from scipy import special, optimize
import numpy as np

f = lambda x: -special.jv(3,x) # (negative of) Bessel function of the first kind; 3rd order
sol = optimize.minimize(f, 1.0, method='Nelder-Mead') # minimisation of Bessel function starting from x0=1.0
                                                      # and using Nelder-Mead local optimisation algorithm
print sol # info on procedure incl result
print sol.x # result (position of max)


#x = np.linspace(0, 10, 5000)
#plot(x, special.jv(3,x), '-', sol.x, -sol.fun, 'o')
