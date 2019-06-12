# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:32:04 2019

@author: mwdocter

trial how to use a reference to a function 
"""

def polynomial_creator(*coefficients):
    """ coefficients are in the form a_n, ... a_1, a_0 
    """
    def polynomial(x):
        res = 0
        for index, coeff in enumerate(coefficients[::-1]):
            res += coeff * x** index
        return res
    return polynomial
  
p1 = polynomial_creator(4)
p2 = polynomial_creator(2, 4)
p3 = polynomial_creator(1, 8, -1, 3, 2)
p4  = polynomial_creator(-1, 2, 1)


for x in range(-2, 2, 1):
    print(x, p1(x), p2(x), p3(x), p4(x))
    
def polynomial_creator2(*coeffs):
    """ coefficients are in the form a_n, a_n_1, ... a_1, a_0 
    """
    def polynomial(x):
        res = coeffs[0]
        for i in range(1, len(coeffs)):
            res = res * x + coeffs[i]
        return res
                 
    return polynomial

p1 = polynomial_creator2(4)
p2 = polynomial_creator2(2, 4)
p3 = polynomial_creator2(1, 8, -1, 3, 2)
p4 = polynomial_creator2(-1, 2, 1)


for x in range(-2, 2, 1):
    print(x, p1(x), p2(x), p3(x), p4(x))