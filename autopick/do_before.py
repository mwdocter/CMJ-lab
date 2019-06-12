# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:08:00 2019

@author: https://github.com/spyder-ide/spyder/issues/2563

how to get rid of all variables, to have a clean initial state
"""
import matplotlib.pyplot as plt

plt.close("all")
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
if __name__ == "__main__":
    clear_all()
