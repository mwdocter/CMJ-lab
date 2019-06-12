# -*- coding: utf-8 -*-
"""
Created on Jun 2019
@author: carlos

a seperate file for doing mapping. 
this should be adapted to start with manual 3 points, then automated numerous points.
like the idl code, and save .coeff and .map files
"""
import autopick.pick_spots_akaze_ALL

class Mapping:#(object):
    def __init__(self, tetra_fname):
        self._tetra_fn = tetra_fname
        self.manual_align()
        self.automatic_align()
        

    @property
    def tetra_fn(self):
        return self._tetra_fn

    def manual_align(self):
        self._tf1_matrix= autopick.pick_spots_akaze_ALL.mapping_manual(self._tetra_fn,
                                                                             show=1,
                                                                             bg=None,
                                                                             tol=0, f=10000)
        return self._tf1_matrix
    
    #@tetra_fn.setter
    def automatic_align(self):
        self._tf2_matrix = autopick.pick_spots_akaze_ALL.mapping_automatic(self._tetra_fn,self._tf1_matrix,
                                                                        show=1,
                                                                        bg=None,
                                                                        tol=0, f=10000)
        #print(self._tf1_matrix)
        #print(self._tf2_matrix)
        return self._tf2_matrix
    
    

