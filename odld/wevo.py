'''
Main functions for running the WESTPA implementation of REVO (WEVO).

TODO: for now, this will be a simple function for testing which will be updated later.

TODO: maybe eventually have something like group.py but resampler.py
      so it would take the input coordinates and output labels for split and merge
'''

import numpy as np

class WEVO:
    '''
    New resampling module.
    '''
    
    # TODO: may need to get segments and intercept at assign step
    # so may need WEVO mapper and sim manager
    # see binless PR for details: https://github.com/westpa/westpa/pull/240/files

    def __init__(self, segments):
        '''
        1. Input ensemble of walkers
        2. Run WEVO to decide which walkers to split and which to merge
        3. Return the decisions for splitting and merging

        Eventually incorporate into the following workflow (from REVO):
            - Calculate the pairwise all-to-all distance matrix using the distance metric
            - Decide which walkers should be merged or cloned
            - Apply the cloning and merging decisions to get the resampled walkers
            - Create the resampling data that includes:
                # maybe the distance matrix can at first be the distance between the pcoord datapoints?
                # like at the end of the iteration dynamics
                # if so, I can prob use the odld data for now
                # later can maybe use the seg.rst files at each segment of the current iteration
                - distance_matrix : the calculated all-to-all distance matrix
                - n_walkers : the number of walkers. number of walkers is
                              kept constant thought the resampling.
                - variation : the final value of trajectory variation
                - images : the images of walkers that is defined by the distance object
                - image_shape : the shape of the image
    
        Parameters
        ----------
        segments : westpa segments object
        '''
        self.segments = segments

    def decide_split_merge(self):
        '''
        Test function for picking which segments to split and merge.
        For now, split the highest weight and merge the lowest weight.
        '''
        pass
