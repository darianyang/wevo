'''
Main functions for running the WESTPA implementation of REVO (WEVO).

TODO: for now, this will be a simple function for testing which will be updated later.

TODO: maybe eventually have something like group.py but resampler.py
      so it would take the input coordinates and output labels for split and merge
'''

import numpy as np
import itertools

class WEVO:
    '''
    New resampling module.
    '''
    
    # TODO: may need to get segments and intercept at assign step
    # so may need WEVO mapper and sim manager
    # see binless PR for details: https://github.com/westpa/westpa/pull/240/files

    #def __init__(self, segments, pcoords, weights):
    def __init__(self, pcoords, weights):
        '''
        1. Input ensemble of walkers
        2. Run WEVO to decide which walkers to split and which to merge
        3. Return the decisions for splitting and merging

        Eventually incorporate into the following workflow (from REVO):
            - Calculate the pairwise all-to-all distance matrix using the distance metric
            - Decide which walkers should be merged or cloned
            - Apply the cloning and merging decisions to get the resampled walkers
            - Create the resampling data that includes:
                - distance_matrix : the calculated all-to-all distance matrix
                - n_walkers : the number of walkers. number of walkers is
                              kept constant thought the resampling.
                - variation : the final value of trajectory variation
                - images : the images of walkers that is defined by the distance object
                - image_shape : the shape of the image
    
        Parameters
        ----------
        segments : westpa segments object
            Each segment also has a weight attribute.
        pcoords : array
            Last pcoord value of each segment for the current iteration
        weights : array
            Weight of each segment.
        '''
        # no segs for now
        #self.segments = segments
        self.pcoords = pcoords
        self.weights = weights

    def decide_split_merge(self):
        '''
        Test function for picking which segments to split and merge.
        For now, split the highest weight and merge the lowest weight.
        '''
        pass

    def _all_to_all_distance(self):
        '''
        Calculate the pairwise all-to-all distances between segments.

        Returns
        -------
        dist_matrix : 2d array
            Distance matrix between each segment coordinate value.
        '''
        # initialize an all-to-all matrix, with 0.0 for self distances (diagonal)
        dist_matrix = np.zeros((len(self.pcoords), len(self.pcoords)))

        # build distance matrix from pcoord value distances
        for i, pcoord_i in enumerate(self.pcoords):
            for j, pcoord_j in enumerate(self.pcoords):
                # calculate Euclidean distance between two points 
                # points can be n-dimensional
                dist = np.linalg.norm(pcoord_i - pcoord_j)
                dist_matrix[i,j] = dist
        
        return dist_matrix
        


if __name__ == '__main__':
    # generate some fake pcoords and weights
    pcoords = np.array([2.5, 3.0, 4.0, 3.2, 3.8])
    weights = np.array([0.2, 0.1, 0.3, 0.15, 0.25])
    # run wevo tests
    resample = WEVO(pcoords, weights)
    resample._all_to_all_distance()
