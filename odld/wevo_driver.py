import logging

import numpy as np
import operator
from westpa.core.we_driver import WEDriver
#from westpa.core.segment import Segment
#from westpa.core.states import InitialState
from wevo import WEVO

log = logging.getLogger(__name__)


class WEVODriver(WEDriver):
    '''
    This replaces the Huber & Kim based WE algorithm with a resampling scheme based 
    on REVO (Resampling of Ensembles by Variance Optimization - Donyapour 2019).
    '''

    # def _run_we(self):
    #     '''
    #     Run recycle/split/merge. Do not call this function directly; instead, use
    #     populate_initial(), rebin_current(), or construct_next().
    #     '''
    #     self._recycle_walkers()
        
    #     # sanity check
    #     self._check_pre()
        
    #     # Regardless of current particle count, always split overweight particles and merge underweight particles
    #     # Then and only then adjust for correct particle count
    #     for (ibin,bin) in enumerate(self.next_iter_binning):
    #         if len(bin) == 0: 
    #             continue
                        
    #         self._split_by_weight(ibin)
    #         self._merge_by_weight(ibin)
    #         if self.do_adjust_counts:
    #             self._adjust_count(ibin)
            
    #     self._check_post()
        
    #     self.new_weights = self.new_weights or []
        
    #     log.debug('used initial states: {!r}'.format(self.used_initial_states))
    #     log.debug('available initial states: {!r}'.format(self.avail_initial_states))

    # def _adjust_count(self, ibin):
    #     bin = self.next_iter_binning[ibin]
    #     target_count = self.bin_target_counts[ibin]
    #     weight_getter = operator.attrgetter('weight')

    #     # split        
    #     while len(bin) < target_count:
    #         log.debug('adjusting counts by splitting')
    #         # always split the highest probability walker into two
    #         segments = sorted(bin, key=weight_getter)
    #         self._split_walker(segments[-1], 2, bin)
            
    #     # merge
    #     while len(bin) > target_count:
    #         log.debug('adjusting counts by merging')
    #         # always merge the two lowest-probability walkers
    #         segments = sorted(bin, key=weight_getter)
    #         self._merge_walkers(segments[:2], cumul_weight=None, bin=bin)

    # def _run_we(self):
    #     '''
    #     Run recycle/split/merge. Do not call this function directly; instead, use
    #     populate_initial(), rebin_current(), or construct_next().
    #     '''
    #     self._recycle_walkers()

    #     # sanity check
    #     self._check_pre()

    #     # Regardless of current particle count, always split overweight particles and merge underweight particles
    #     # Then and only then adjust for correct particle count
    #     total_number_of_subgroups = 0
    #     total_number_of_particles = 0
    #     for (ibin, bin) in enumerate(self.next_iter_binning):
    #         if len(bin) == 0:
    #             continue

    #         # Splits the bin into subgroups as defined by the called function
    #         target_count = self.bin_target_counts[ibin]
    #         subgroups = self.subgroup_function(self, ibin, **self.subgroup_function_kwargs)
    #         total_number_of_subgroups += len(subgroups)
    #         # Clear the bin
    #         segments = np.array(sorted(bin, key=operator.attrgetter('weight')), dtype=np.object_)
    #         weights = np.array(list(map(operator.attrgetter('weight'), segments)))
    #         ideal_weight = weights.sum() / target_count
    #         bin.clear()
    #         # Determines to see whether we have more sub bins than we have target walkers in a bin (or equal to), and then uses
    #         # different logic to deal with those cases.  Should devolve to the Huber/Kim algorithm in the case of few subgroups.
    #         if len(subgroups) >= target_count:
    #             for i in subgroups:
    #                 # Merges all members of set i.  Checks to see whether there are any to merge.
    #                 if len(i) > 1:
    #                     (segment, parent) = self._merge_walkers(
    #                         list(i),
    #                         np.add.accumulate(np.array(list(map(operator.attrgetter('weight'), i)))),
    #                         i,
    #                     )
    #                     i.clear()
    #                     i.add(segment)
    #                 # Add all members of the set i to the bin.  This keeps the bins in sync for the adjustment step.
    #                 bin.update(i)

    #             if len(subgroups) > target_count:
    #                 self._adjust_count(bin, subgroups, target_count)

    #         if len(subgroups) < target_count:
    #             for i in subgroups:
    #                 self._split_by_weight(i, target_count, ideal_weight)
    #                 self._merge_by_weight(i, target_count, ideal_weight)
    #                 # Same logic here.
    #                 bin.update(i)
    #             if self.do_adjust_counts:
    #                 # A modified adjustment routine is necessary to ensure we don't unnecessarily destroy trajectory pathways.
    #                 self._adjust_count(bin, subgroups, target_count)
    #         if self.do_thresholds:
    #             for i in subgroups:
    #                 self._split_by_threshold(bin, i)
    #                 self._merge_by_threshold(bin, i)
    #             for iseg in bin:
    #                 if iseg.weight > self.largest_allowed_weight or iseg.weight < self.smallest_allowed_weight:
    #                     log.warning(
    #                         f'Unable to fulfill threshold conditions for {iseg}. The given threshold range is likely too small.'
    #                     )
    #         total_number_of_particles += len(bin)
    #     log.debug('Total number of subgroups: {!r}'.format(total_number_of_subgroups))

    #     self._check_post()

    #     self.new_weights = self.new_weights or []

    #     log.debug('used initial states: {!r}'.format(self.used_initial_states))
    #     log.debug('available initial states: {!r}'.format(self.avail_initial_states))

    # ATB resampler base
    # def _segment_index_converter(self, mode, pcoords, curr_pcoords, scaled_diffs):

    #     if mode == "split":
    #         to_split_idx = np.argmin(scaled_diffs)
    #         curr_pcoords_to_split = curr_pcoords[:,-1][to_split_idx]
    #         converted_idx = int(np.where(pcoords[:,0] == curr_pcoords_to_split)[0])

    #         return converted_idx

    #     if mode == "merge":
    #         to_merge_idx = np.argsort(-scaled_diffs,axis=0)[:2]
    #         curr_pcoords_to_merge = curr_pcoords[:,-1][to_merge_idx]
    #         if curr_pcoords_to_merge.shape[0] > 1:
    #             converted_idx = np.zeros(curr_pcoords_to_merge.shape[0], dtype=int)
    #             for idx, val in enumerate(curr_pcoords_to_merge):
    #                 converted_idx[idx] = int(np.where(pcoords[:,0] == val)[0])
    #         else: 
    #             converted_idx = np.where(pcoords[:,0] == curr_pcoords_to_merge)[0]

    #         return converted_idx

    def _split_by_wevo(self, bin, to_split, split_into):
        
        # remove the walker being split
        bin.remove(to_split)
        # get the n split walker children
        new_segments_list = self._split_walker(to_split, split_into, bin)
        # add all new split walkers back into bin, maintaining history
        bin.update(new_segments_list)


    def _merge_by_wevo(self, bin, to_merge, cumul_weight=None):

        # removes every walker in to_merge 
        bin.difference_update(to_merge)
        #new_segment, parent = self._merge_walkers(to_merge, None, bin)
        new_segment, parent = self._merge_walkers(to_merge, cumul_weight, bin)
        # add in new merged walker
        bin.add(new_segment)


    def _run_we(self):
        '''
        Run recycle/split/merge. Do not call this function directly; instead, use
        populate_initial(), rebin_current(), or construct_next().
        '''
        self._recycle_walkers()

        # sanity check
        self._check_pre()

        # dummy resampling block
        # TODO: wevo is really only using one bin
        # ibin only needed right now for temp split merge option (TODO)
        for ibin, bin in enumerate(self.next_iter_binning):
            # TODO: is this needed?
            if len(bin) == 0:
                continue
            else:
                # TODO: maybe also pull iter number and put into wevo log?

                # this will just get you the final pcoord for each segment... which may not be enough
                segments = np.array(sorted(bin, key=operator.attrgetter('weight')), dtype=np.object_)
                pcoords = np.array(list(map(operator.attrgetter('pcoord'), segments)))
                weights = np.array(list(map(operator.attrgetter('weight'), segments)))

                log_weights = -1 * np.log(weights)
 
                nsegs = pcoords.shape[0]
                nframes = pcoords.shape[1]
                
                #print("weights", weights)
                #print("before pcoords", pcoords)
                pcoords = pcoords.reshape(nsegs,nframes)
                #print("after pcoords", pcoords)

                # # this will allow you to get the pcoords for all frames
                # these setps may not be needed with wevo (TODO)
                # current_iter_segments = self.current_iter_segments

                # curr_segments = np.array(sorted(current_iter_segments, key=operator.attrgetter('weight')), dtype=np.object_)
                # curr_pcoords = np.array(list(map(operator.attrgetter('pcoord'), curr_segments)))
                # curr_weights = np.array(list(map(operator.attrgetter('weight'), curr_segments)))

                # log_weights = -1 * np.log(weights)
 
                # nsegs = pcoords.shape[0]
                # nframes = pcoords.shape[1]

                #diffs = np.zeros((nsegs))

                # for wevo, I don't need all pcoords, just the final pcoord vals
                #curr_pcoords = curr_pcoords.reshape(nsegs,nframes)
                
                # # find percent change between first and last frame
                # for idx, ival in enumerate(curr_pcoords):
                #     diff = ((ival[-1] - ival[0]) / ival[0]) * 100
                #     diffs[idx] = diff
                # diffs[diffs > 0] = 0
                # init_check = np.any(diffs)
                # scaled_diffs = diffs * log_weights
                
                # print for sanity check
                print("pcoords", pcoords[:,0])
                #print("current pcoords", curr_pcoords)
                print("weights", weights)
                #print("log_weights", log_weights)
                #print("diffs", diffs)
                #print("scaled_diffs", scaled_diffs)

                # if init_check:

                #     # split walker with largest scaled diff
                #     split_into = 2
                #     to_split_index = self._segment_index_converter("split", pcoords, curr_pcoords, scaled_diffs)
                #     to_split = segments[to_split_index]

                #     self._split_by_diff(bin, to_split, split_into)
    
                #     # merge walker with lowest scaled diff into next lowest
                #     cumul_weight = np.add.accumulate(weights)
                #     to_merge_index = self._segment_index_converter("merge", pcoords, curr_pcoords, scaled_diffs)
                #     to_merge = segments[to_merge_index]

                #     self._merge_by_diff(bin, to_merge, cumul_weight)

                # run wevo and do split merge based on wevo decisions
                resample = WEVO(pcoords[:,0], weights, merge_dist=0.5, char_dist=1.13, merge_alg="pairs")
                split, merge, variation = resample.resample()
                print(f"Final variation value after {resample.count} wevo cycles: ", variation)

                # go through each seg and split merge
                segs = 0
                splitting = 0
                merging = 0
                
                # TODO: need to make sure the index is consistent
                for i, seg in enumerate(segments):
                    #print(f"seg {i}: split: {split[i]} merge: {merge[i]}")
                    #print(f"pcoord val: {seg.pcoord[0]}, weight: {seg.weight}, parent: {seg.parent_id}")
                    
                    # split into n walkers based on split value
                    
                    # split or merge on a segment-by-segment basis
                    if split[i] != 0:
                        self._split_by_wevo(bin, seg, split[i])
                        #self._split_by_wevo(bin, seg, split[i] - 1)
                        splitting += split[i]
                    if len(merge[i]) != 0:
                        # list of all segs objects in the current merge list element
                        to_merge = [segment for num, segment in enumerate(segments) if num in merge[i]]
                        
                        # cumul_weight should be the total weights of all the segments being merged
                        #cumul_weight = np.add.accumulate(weights)
                        #self._merge_by_wevo(bin, to_merge, cumul_weight)
                        self._merge_by_wevo(bin, to_merge)
                        merging += len(to_merge)
                    
                    segs += 1

                # make bin target count consistent via splitting high weight and merging low weight
                # TODO: maybe do this with variance sorting instead of weight sorting?
                # if self.do_adjust_counts:
                #     self._adjust_count(ibin)

                print("Final weight sum: ", np.sum(weights))
                print(f"Total = {segs}, splitting = {splitting}, merging = {merging}")

                # TODO: temp fix for no initial splitting needed for wevo to start working
                # i.e. no wevo cycles were able to run
                # if resample.count == 1:
                #     # running H&K resampling
                #     target_count = self.bin_target_counts[ibin]
                #     ideal_weight = weights.sum() / target_count
                #     self._split_by_weight(bin, target_count, ideal_weight)
                #     self._merge_by_weight(bin, target_count, ideal_weight)
                #     # if self.do_adjust_counts:
                #     #     self._adjust_count(bin)
                

        # another sanity check
        self._check_post()

        self.new_weights = self.new_weights or []

        log.debug('used initial states: {!r}'.format(self.used_initial_states))
        log.debug('available initial states: {!r}'.format(self.avail_initial_states))