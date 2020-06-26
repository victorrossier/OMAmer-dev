import os
import sys
import tables
import numba
import numpy as np
from ete3 import Tree
from matplotlib import pyplot as plt
from itertools import repeat
from property_manager import cached_property

from .flat_search import DIAMONDsearch, FlatSearch
from .hierarchy import get_root_leaf_hog_offsets, get_sispecies_candidates, compute_inparalog_coverage_new
from .search import Search
from .index import SequenceBuffer, IndexValidation

from tables import open_file
from tqdm import tqdm
import collections
import random

'''
TO DO:
 - think and dev filter for duplication non knowable given species sampling (lcasis)
 - merge some code between Validation and ValidationFamily
 - include LCA filter in ValidationFamily for FN but it would mean a family defined at metazoa with >= 6 platypus only proteins...
'''

class Validation():

	def __init__(self, se, name=None, path=None, val_mode='golike'):

		assert se.mode == 'r', 'Search must be opened in read mode.'

		self.db = se.db
		self.ki = se.ki
		self.fs = se.fs
		self.se = se

		# hdf5 file
		self.name = "{}_{}".format(self.se.name, name if name else 'subfamily_{}'.format(val_mode))
		self.path = path if path else self.se.path
		self.file = "{}{}.h5".format(self.path, self.name)

		if os.path.isfile(self.file):
			self.mode = 'r'
			self.va = tables.open_file(self.file, self.mode)

		else:
			self.mode = 'w'
			self.va = tables.open_file(self.file, self.mode)

		# mode of validation
		self.val_mode = val_mode

	def __enter__(self):
	    return self

	def __exit__(self, *_):
	    self.va.close()
	    
	def clean(self):
	    '''
	    close and remove hdf5 file
	    '''
	    self.__exit__()
	    try:
	        os.remove(self.file)
	    except FileNotFoundError:
	        print("{} already cleaned".format(self.file))

	# # tax and HOG filters --> moved to IndexValidation

	### easy access to data via properties ###
	@property
	def _thresholds(self):
	    if '/Threshold' in self.va:
	        return self.va.root.Threshold
	    else:
	        return None

	@property
	def _tp_pre(self):
	    if '/TP_pre' in self.va:
	        return self.va.root.TP_pre
	    else:
	        return None

	@property
	def _tp_rec(self):
	    if '/TP_rec' in self.va:
	        return self.va.root.TP_rec
	    else:
	        return None

	@property
	def _fp(self):
	    if '/FP' in self.va:
	        return self.va.root.FP
	    else:
	        return None

	@property
	def _fn(self):
	    if '/FN' in self.va:
	        return self.va.root.FN
	    else:
	        return None

	####################################################################################################################################
	def validate(self, thresholds, hog2x=None, x_num=None, hf_lca=True, hf_lcasis=False, prob=False):

		assert (self.mode in {'w', 'a'}), 'Validation must be opened in write mode.'

		# store thresholds
		self.va.create_carray('/', 'Threshold', obj=np.array(thresholds, dtype=np.float64), filters=self.db._compr)

		# is smaller better?
		# prob = True if self.se.score in {'theo_prob', 'kmerfreq_prob'} else False

		# run validation by loading once in memory the required args
		tp_pre_query2tax2tresh, tp_rec_query2tax2tresh, fn_query2tax2tresh, fp_query2tax2tresh = self._validate(self._thresholds[:], self.se._res_tab[:], 
			self.db._prot_tab[:], self.db._fam_tab[:], self.db._hog_tab.col('ParentOff'), self.se._bestpath_mask[:] , self.se._hog_score[:], get_root_leaf_hog_offsets, 
			prob, hog2x, x_num, self.ki.hog_filter_lca if hf_lca else np.array([]), self.ki.hog_filter_lcasis if hf_lcasis else np.array([]), self.val_mode)

		# store results
		self.va.create_carray('/', 'TP_pre', obj=tp_pre_query2tax2tresh, filters=self.db._compr)
		self.va.create_carray('/', 'TP_rec', obj=tp_rec_query2tax2tresh, filters=self.db._compr)
		self.va.create_carray('/', 'FN', obj=fn_query2tax2tresh, filters=self.db._compr)
		self.va.create_carray('/', 'FP', obj=fp_query2tax2tresh, filters=self.db._compr)

		# close and re-open in read mode
		self.va.close()
		self.mode = 'r'
		self.va = tables.open_file(self.file, self.mode)

		# delete cache
		if hf_lca:
			del self.ki.tax_filter, self.ki.hog_filter_lca			

	@staticmethod
	def _validate(thresholds, res_tab, prot_tab, fam_tab, hog2parent, bestpath_mask, hog_score, fun_root_leaf, prob, hog2x, x_num, hog_filter_lca, hog_filter_lcasis, val_mode):
		'''
		args:
		 - pv: whether p-value type of score (the lower the better)
		 - hog2x: mapper between HOGs and any HOG grouping such as taxonomy or amount of duplications in query sister
		 - x_num: number of group in hog2x mapper
		'''
		def _compute_tp_fp_fn(
			tp_hogs, fn_hogs, fp_hogs, hog2x, x_num, tp_pre_query2x2tresh, tp_rec_query2x2tresh, fn_query2x2tresh, fp_query2x2tresh, val_mode):
			'''
			3 options to define TPs, FNs and FPs
			'''
			# map to bins and keep track of HOG numbers
			tp_x, tp_nr = np.unique(hog2x[tp_hogs], return_counts=True) if x_num else tp_hogs
			fn_x, fn_nr = np.unique(hog2x[fn_hogs], return_counts=True) if x_num else fn_hogs
			fp_x, fp_nr = np.unique(hog2x[fp_hogs], return_counts=True) if x_num else fp_hogs

			# my custom approach: split TPs in to recall and precision TPs
			if val_mode == 'custom':
			    tp_x_pre = np.setdiff1d(tp_x, fp_x)
			    tp_x_rec = np.setdiff1d(tp_x, fn_x)

			    # and counts TPs, FPs and FNs by query
			    tp_pre_query2x2tresh[q, tp_x_pre, t_off] = 1
			    tp_rec_query2x2tresh[q, tp_x_rec, t_off] = 1
			    fn_query2x2tresh[q, fn_x, t_off] = 1
			    fp_query2x2tresh[q, fp_x, t_off] = 1

			# stringent approach ignoring hierarchy
			elif val_mode == 'stringent':   
			    tp_x = np.setdiff1d(tp_x, np.union1d(fn_x, fp_x))

			    #also counts TPs, FPs and FNs by query
			    tp_pre_query2x2tresh[q, tp_x, t_off] = 1
			    tp_rec_query2x2tresh[q, tp_x, t_off] = 1
			    fn_query2x2tresh[q, fn_x, t_off] = 1
			    fp_query2x2tresh[q, fp_x, t_off] = 1

			# approach where TPs, FPs and FNs are counted by HOG
			elif val_mode == 'golike':
			    tp_pre_query2x2tresh[q, tp_x, t_off] = tp_nr
			    tp_rec_query2x2tresh[q, tp_x, t_off] = tp_nr
			    fn_query2x2tresh[q, fn_x, t_off] = fn_nr
			    fp_query2x2tresh[q, fp_x, t_off] = fp_nr

		thresholds=np.array(thresholds, dtype=np.float64)

		# store validation results
		tp_pre_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)
		tp_rec_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)
		fn_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)
		fp_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)

		# iterage over queries
		for q in tqdm(range(res_tab.size)):

			# true data
			prot_off = res_tab[q]['QueryOff']
			true_fam = prot_tab[prot_off]['FamOff']
			true_leafhog = prot_tab[prot_off]['HOGoff']
			true_hogs = fun_root_leaf(true_leafhog, hog2parent)

			# remove hogs specific to hidden taxa from true hogs
			#true_hogs = true_hogs[~tax_filter[hog2lcatax[true_hogs]]]
			if hog_filter_lca.size > 0:
				true_hogs = true_hogs[~hog_filter_lca[true_hogs]]

			# pred fam
			pred_fam = res_tab[q]['FamOff']

			# hogs of pred fam
			hog_off = fam_tab[pred_fam]['HOGoff']
			hog_offsets = np.arange(hog_off, hog_off + fam_tab[pred_fam]['HOGnum'], dtype=np.uint64)

			# best path hogs and score
			hogs_mask = bestpath_mask[q][hog_offsets]
			hogs_bestpath = hog_offsets[hogs_mask]
			hogs_bestpath_score = hog_score[q][hogs_bestpath]

			# elif idx_type == 'cs': --> used for orlov

			#     pred_leafhog = res_tab[q]['HOGoff']
			#     score = res_tab[q]['Score']

			#     # cs can have NA result (-1) with BLAST
			#     if pred_leafhog != -1:
			#     	hogs_bestpath = fun_root_leaf(pred_leafhog, hog2parent)
			#     else:
			#     	hogs_bestpath = np.array([pred_leafhog], dtype=np.uint64)
			#     # some broadcasting to use same function later
			#     hogs_bestpath_score = np.full(hogs_bestpath.shape, score)

			# iterate over thresholds
			for t_off in range(thresholds.size):
				t_val = thresholds[t_off]

				# pred hogs
				pred_hogs = hogs_bestpath[(hogs_bestpath_score < t_val) if prob else (hogs_bestpath_score >= t_val)]
				
				if hog_filter_lcasis.size > 0:
					# remove HOGs with only sister HOGs that are specific to hidden taxa
					pred_hogs = pred_hogs[~hog_filter_lcasis[pred_hogs]]

				# confront true classes against predicted classes to get benchmark results
				# not supported by numba ...
				tp_hogs = np.intersect1d(true_hogs, pred_hogs, assume_unique=True)
				fn_hogs = np.setdiff1d(true_hogs, pred_hogs, assume_unique=True)
				fp_hogs = np.setdiff1d(pred_hogs, true_hogs, assume_unique=True)

				# map to bins after benchmark if bins
				tp_x = np.unique(hog2x[tp_hogs]) if x_num else tp_hogs
				fn_x = np.unique(hog2x[fn_hogs]) if x_num else fn_hogs
				fp_x = np.unique(hog2x[fp_hogs]) if x_num else fp_hogs

				_compute_tp_fp_fn(tp_hogs, fn_hogs, fp_hogs, hog2x, x_num, tp_pre_query2x2tresh, tp_rec_query2x2tresh,
					fn_query2x2tresh, fp_query2x2tresh, val_mode)

		return tp_pre_query2x2tresh, tp_rec_query2x2tresh, fn_query2x2tresh, fp_query2x2tresh

	### bin taxa #################################################################################################################################
	@staticmethod
	def bin_taxa(species_tree_path, root_tax_level, taxoff2tax, tax_tab, query_species, hog2taxoff, bin_num, root=False, merge_post_lca_taxa=True):

		def _bin_taxa(bin_num, tax2dist, lca_tax, root, merge_post_lca_taxa):
		    
		    # remove one bin if not merging taxa descending from lca taxon
		    bin_num = bin_num if merge_post_lca_taxa else bin_num - 1
		    
		    # grab distance of lca taxon
		    lca_dist = tax2dist[lca_tax] if lca_tax else max(tax2dist.values())
		    
		    if root:
		        dist_range_size = lca_dist / (bin_num - 1)
		        dist_ranges = [-1] + [dist_range_size*n for n in range(0, bin_num)]
		    else:
		        dist_range_size = lca_dist / bin_num
		        dist_ranges = [-1] + [dist_range_size*n for n in range(1, bin_num + 1)]
		    
		    tax2taxbin = {}

		    # fill bins with taxa within distance ranges
		    for bn in range(bin_num):
		        bin_taxa = {k for k,v in tax2dist.items() if v > dist_ranges[bn] and v <= dist_ranges[bn + 1]}
		        tax2taxbin.update(dict(zip(bin_taxa, repeat(bn))))
		    
		    # deal with taxa descending from lca taxon
		    max_dist = max(tax2dist.values())
		    post_lca_taxa = {k for k,v in tax2dist.items() if v > lca_dist and v <= max_dist}
		    if merge_post_lca_taxa:
		        tax2taxbin.update(dict(zip(post_lca_taxa, repeat(bin_num - 1))))
		    else:
		        tax2taxbin.update(dict(zip(post_lca_taxa, repeat(bin_num))))

		    return tax2taxbin
		
		# select subtree
		stree = Tree(species_tree_path, format=1, quoted_node_names=True)
		tl_stree = [x for x in stree.traverse() if x.name == root_tax_level][0]
		tax2dist = {''.join(x.name.split()):x.get_distance(tl_stree) for x in tl_stree.traverse()}

		# grab lca taxon between query and reference
		lca_tax = taxoff2tax[tax_tab[np.searchsorted(taxoff2tax, query_species.encode('ascii'))]['ParentOff']].decode('ascii')

		tax2taxbin = _bin_taxa(bin_num, tax2dist, lca_tax, root, merge_post_lca_taxa)

		taxoff2taxbin = {np.searchsorted(taxoff2tax, tax.encode('ascii')):b for tax, b in tax2taxbin.items()}

		# because opistokonta error, add an exception with -1
		return np.array([taxoff2taxbin.get(tax, -1) for tax in hog2taxoff], np.int64), tax2taxbin

	
	def compute_inparalog_coverage(self, htax):

		tax_off = np.searchsorted(self.db._tax_tab.col('ID'), htax)
		hidden_taxa = [np.searchsorted(self.db._tax_tab.col('ID'), x) for x in self.ki.hidden_species]

		# compute inparalog coverage for all queries of htax
		sispecies_cands = get_sispecies_candidates(tax_off, self.db._tax_tab[:], self.db._ctax_arr[:], hidden_taxa)

		query_ids = self.fs._query_id[:].flatten()
		prot_tab = self.db._prot_tab[:]
		cprot_buff = self.db._cprot_arr[:]
		hog_tab = self.db._hog_tab[:]
		chog_buff = self.db._chog_arr[:]

		ip_covs = np.array([compute_inparalog_coverage_new(q, query_ids, prot_tab, 
			cprot_buff, hog_tab, chog_buff, hidden_taxa, sispecies_cands) for q in range(query_ids.size)])

		return ip_covs

	@staticmethod
	def partition_queries(thresholds, query_values, parameter_name):
		part_names = []
		partitions = []
		curr_thresh = -1
		for thresh in thresholds:
		    part = np.full(query_values.size, False)
		    part[(query_values > curr_thresh) & (query_values <= thresh)] = True
		    partitions.append(part)
		    part_names.append('{} < {} <= {}'.format(curr_thresh if curr_thresh != -1 else 0, parameter_name, thresh))
		    curr_thresh = thresh

		# last one
		part = np.full(query_values.size, False)
		part[query_values > curr_thresh] = True
		partitions.append(part)
		part_names.append('{} < {} <= {}'.format(curr_thresh if curr_thresh != -1 else 0, parameter_name, 1))

		return np.array(partitions), part_names


	@staticmethod
	def compute_precision_recall(tp_pre_query2bin2tresh, tp_rec_query2bin2tresh, fn_query2bin2tresh, fp_query2bin2tresh, partitions=np.array([])):

		if partitions.size == 0:
		    partitions = np.array([np.full(tp_pre_query2bin2tresh.shape[0], True)])

		part_num = partitions.shape[0]
		bin_num = tp_pre_query2bin2tresh.shape[1]
		thresh_num = tp_pre_query2bin2tresh.shape[2]

		part2bin2pre = np.zeros((part_num, bin_num, thresh_num), dtype=np.float64)
		part2bin2rec = np.zeros((part_num, bin_num, thresh_num), dtype=np.float64)
		part2bin2tp_pre_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)
		part2bin2tp_rec_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)
		part2bin2fn_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)
		part2bin2fp_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)

		for p in range(part_num):
		    part = partitions[p]
		    for b in range(bin_num):
		        for t in range(thresh_num):
		            tp_pre_nr = np.sum(tp_pre_query2bin2tresh[:, b, t][part])
		            tp_rec_nr = np.sum(tp_rec_query2bin2tresh[:, b, t][part])
		            fn_nr = np.sum(fn_query2bin2tresh[:, b, t][part])
		            fp_nr = np.sum(fp_query2bin2tresh[:, b, t][part])
		            part2bin2pre[p, b, t] = (tp_pre_nr/(tp_pre_nr + fp_nr)) if tp_pre_nr or fp_nr else 0
		            part2bin2rec[p, b, t] = (tp_rec_nr/(tp_rec_nr + fn_nr)) if tp_rec_nr or fn_nr else 0
		            part2bin2tp_pre_nr[p, b, t] = tp_pre_nr
		            part2bin2tp_rec_nr[p, b, t] = tp_rec_nr
		            part2bin2fn_nr[p, b, t] = fn_nr
		            part2bin2fp_nr[p, b, t] = fp_nr

		part2bin2query_nr = np.zeros((part_num, bin_num), dtype=np.uint64)

		for p in range(part_num):
		    for b in range(bin_num):
		        part2bin2query_nr[p, b] = part2bin2tp_rec_nr[p, b, 0] + part2bin2fn_nr[p, b, 0]
		        
		return part2bin2pre, part2bin2rec, part2bin2query_nr

	def F1(self, part2bin2pre, part2bin2rec):
		n = part2bin2pre * part2bin2rec
		d = part2bin2pre + part2bin2rec
		part2bin2f1 = 2 * np.divide(n, d, out=np.zeros_like(n), where=d!=0)
		part2bin2f1_max = np.max(part2bin2f1, axis=2)

		part2bin2f1_tval = np.zeros(part2bin2f1_max.shape)
		part2bin2f1_toff = np.zeros(part2bin2f1_max.shape, dtype=np.uint64)

		for p in range(part2bin2f1.shape[0]):
			for b in range(part2bin2f1.shape[1]):
				toff = np.where(part2bin2f1[p, b]==part2bin2f1_max[p, b])[0][0]
				part2bin2f1_tval[p, b] = self._thresholds[:][toff]
				part2bin2f1_toff[p, b] = toff
		return part2bin2f1_max, part2bin2f1_tval, part2bin2f1_toff

	def PREpro(self, part2bin2pre, part2bin2rec):
		n = part2bin2pre * part2bin2pre * part2bin2rec
		d = part2bin2pre + part2bin2pre + part2bin2rec
		part2bin2f1 = 3 * np.divide(n, d, out=np.zeros_like(n), where=d!=0)
		part2bin2f1_max = np.max(part2bin2f1, axis=2)

		part2bin2f1_tval = np.zeros(part2bin2f1_max.shape)
		part2bin2f1_toff = np.zeros(part2bin2f1_max.shape, dtype=np.uint64)

		for p in range(part2bin2f1.shape[0]):
			for b in range(part2bin2f1.shape[1]):
				toff = np.where(part2bin2f1[p, b]==part2bin2f1_max[p, b])[0][0]
				part2bin2f1_tval[p, b] = self._thresholds[:][toff]
				part2bin2f1_toff[p, b] = toff
		return part2bin2f1_max, part2bin2f1_tval, part2bin2f1_toff


class ValidationCS(Validation):
	'''
	For DIAMOND and Smith-Waterman
	'''
	def __init__(self, se, name=None, path=None, val_mode='golike'):
		super().__init__(se, name, path, val_mode)
		self.fs = se
		self.se = se

	def validate(self, thresholds, hog2x=None, x_num=None, hf_lca=True, hf_lcasis=False, prob=True):

		assert (self.mode in {'w', 'a'}), 'Validation must be opened in write mode.'

		# store thresholds
		self.va.create_carray('/', 'Threshold', obj=np.array(thresholds, dtype=np.float64), filters=self.db._compr)

		# run validation by loading once in memory the required args
		tp_pre_query2tax2tresh, tp_rec_query2tax2tresh, fn_query2tax2tresh, fp_query2tax2tresh = self._validate(self._thresholds[:], self.se._res_tab[:], 
			self.db._prot_tab[:], self.db._fam_tab[:], self.db._hog_tab.col('ParentOff'), get_root_leaf_hog_offsets, 
			prob, hog2x, x_num, self.ki.hog_filter_lca if hf_lca else np.array([]), self.ki.hog_filter_lcasis if hf_lcasis else np.array([]), self.val_mode)

		# store results
		self.va.create_carray('/', 'TP_pre', obj=tp_pre_query2tax2tresh, filters=self.db._compr)
		self.va.create_carray('/', 'TP_rec', obj=tp_rec_query2tax2tresh, filters=self.db._compr)
		self.va.create_carray('/', 'FN', obj=fn_query2tax2tresh, filters=self.db._compr)
		self.va.create_carray('/', 'FP', obj=fp_query2tax2tresh, filters=self.db._compr)

		# close and re-open in read mode
		self.va.close()
		self.mode = 'r'
		self.va = tables.open_file(self.file, self.mode)

		# delete cache
		if hf_lca:
			del self.ki.tax_filter, self.ki.hog_filter_lca

	@staticmethod
	def _validate(thresholds, res_tab, prot_tab, fam_tab, hog2parent, fun_root_leaf, prob, hog2x, x_num, hog_filter_lca, hog_filter_lcasis, val_mode):
		'''
		args:
		 - pv: whether p-value type of score (the lower the better)
		 - hog2x: mapper between HOGs and any HOG grouping such as taxonomy or amount of duplications in query sister
		 - x_num: number of group in hog2x mapper
		'''
		def _compute_tp_fp_fn(
			tp_hogs, fn_hogs, fp_hogs, hog2x, x_num, tp_pre_query2x2tresh, tp_rec_query2x2tresh, fn_query2x2tresh, fp_query2x2tresh, val_mode):
			'''
			3 options to define TPs, FNs and FPs
			'''
			# map to bins and keep track of HOG numbers
			tp_x, tp_nr = np.unique(hog2x[tp_hogs], return_counts=True) if x_num else tp_hogs
			fn_x, fn_nr = np.unique(hog2x[fn_hogs], return_counts=True) if x_num else fn_hogs
			fp_x, fp_nr = np.unique(hog2x[fp_hogs], return_counts=True) if x_num else fp_hogs

			# my custom approach: split TPs in to recall and precision TPs
			if val_mode == 'custom':
			    tp_x_pre = np.setdiff1d(tp_x, fp_x)
			    tp_x_rec = np.setdiff1d(tp_x, fn_x)

			    # and counts TPs, FPs and FNs by query
			    tp_pre_query2x2tresh[q, tp_x_pre, t_off] = 1
			    tp_rec_query2x2tresh[q, tp_x_rec, t_off] = 1
			    fn_query2x2tresh[q, fn_x, t_off] = 1
			    fp_query2x2tresh[q, fp_x, t_off] = 1

			# stringent approach ignoring hierarchy
			elif val_mode == 'stringent':   
			    tp_x = np.setdiff1d(tp_x, np.union1d(fn_x, fp_x))

			    #also counts TPs, FPs and FNs by query
			    tp_pre_query2x2tresh[q, tp_x, t_off] = 1
			    tp_rec_query2x2tresh[q, tp_x, t_off] = 1
			    fn_query2x2tresh[q, fn_x, t_off] = 1
			    fp_query2x2tresh[q, fp_x, t_off] = 1

			# approach where TPs, FPs and FNs are counted by HOG
			elif val_mode == 'golike':
			    tp_pre_query2x2tresh[q, tp_x, t_off] = tp_nr
			    tp_rec_query2x2tresh[q, tp_x, t_off] = tp_nr
			    fn_query2x2tresh[q, fn_x, t_off] = fn_nr
			    fp_query2x2tresh[q, fp_x, t_off] = fp_nr

		thresholds=np.array(thresholds, dtype=np.float64)

		# store validation results
		tp_pre_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)
		tp_rec_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)
		fn_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)
		fp_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)

		# iterage over queries
		for q in tqdm(range(res_tab.size)):

			# true data
			prot_off = res_tab[q]['QueryOff']
			true_fam = prot_tab[prot_off]['FamOff']
			true_leafhog = prot_tab[prot_off]['HOGoff']
			true_hogs = fun_root_leaf(true_leafhog, hog2parent)

			# remove hogs specific to hidden taxa from true hogs
			#true_hogs = true_hogs[~tax_filter[hog2lcatax[true_hogs]]]
			if hog_filter_lca.size > 0:
				true_hogs = true_hogs[~hog_filter_lca[true_hogs]]

			# pred fam
			pred_fam = res_tab[q]['FamOff']

			# # hogs of pred fam
			# hog_off = fam_tab[pred_fam]['HOGoff']
			# hog_offsets = np.arange(hog_off, hog_off + fam_tab[pred_fam]['HOGnum'], dtype=np.uint64)

			# # best path hogs and score
			# hogs_mask = bestpath_mask[q][hog_offsets]
			# hogs_bestpath = hog_offsets[hogs_mask]
			# hogs_bestpath_score = hog_score[q][hogs_bestpath]

			# elif idx_type == 'cs': --> used for orlov

			pred_leafhog = res_tab[q]['HOGoff']
			score = res_tab[q]['Score']

			# CS can have NA result (-1) with BLAST
			if pred_leafhog != -1:
				hogs_bestpath = fun_root_leaf(pred_leafhog, hog2parent)
			else:
				hogs_bestpath = np.array([pred_leafhog], dtype=np.uint64)
			# some broadcasting to use same function later
			hogs_bestpath_score = np.full(hogs_bestpath.shape, score)

			# iterate over thresholds
			for t_off in range(thresholds.size):
				t_val = thresholds[t_off]

				# pred hogs
				pred_hogs = hogs_bestpath[(hogs_bestpath_score < t_val) if prob else (hogs_bestpath_score >= t_val)]
				
				if hog_filter_lcasis.size > 0:
					# remove HOGs with only sister HOGs that are specific to hidden taxa
					pred_hogs = pred_hogs[~hog_filter_lcasis[pred_hogs]]

				# confront true classes against predicted classes to get benchmark results
				# not supported by numba ...
				tp_hogs = np.intersect1d(true_hogs, pred_hogs, assume_unique=True)
				fn_hogs = np.setdiff1d(true_hogs, pred_hogs, assume_unique=True)
				fp_hogs = np.setdiff1d(pred_hogs, true_hogs, assume_unique=True)

				# map to bins after benchmark if bins
				tp_x = np.unique(hog2x[tp_hogs]) if x_num else tp_hogs
				fn_x = np.unique(hog2x[fn_hogs]) if x_num else fn_hogs
				fp_x = np.unique(hog2x[fp_hogs]) if x_num else fp_hogs

				_compute_tp_fp_fn(tp_hogs, fn_hogs, fp_hogs, hog2x, x_num, tp_pre_query2x2tresh, tp_rec_query2x2tresh,
					fn_query2x2tresh, fp_query2x2tresh, val_mode)

		return tp_pre_query2x2tresh, tp_rec_query2x2tresh, fn_query2x2tresh, fp_query2x2tresh


class ValidationFamily():

	def __init__(self, se_pos, neg_root_taxon):

		assert se_pos.mode == 'r', 'Search must be opened in read mode.'

		self.db = se_pos.db
		self.ki = se_pos.ki
		self.fs = se_pos.fs
		self.se_pos = se_pos

		# hdf5 file
		self.neg_root_taxon = neg_root_taxon 
		self.name = "{}_{}".format(self.se_pos.name, neg_root_taxon if neg_root_taxon else 'Random')
		self.path = self.se_pos.path
		self.file = "{}{}.h5".format(self.path, self.name)

		if os.path.isfile(self.file):
			self.mode = 'r'
			self.va = tables.open_file(self.file, self.mode)
		else:
			self.mode = 'w'
			self.va = tables.open_file(self.file, self.mode)

		# negative queries
		self.neg_query_path = '{}neg_query/'.format(self.db.path)
		if not os.path.exists(self.neg_query_path):
			os.mkdir(self.neg_query_path)
	
	def __enter__(self):
		return self

	def __exit__(self, *_):
		self.va.close()

	def clean_vf(self):
		'''
		clean only validation family, not the negative search files
		'''
		self.__exit__()
		try:
			os.remove(self.file)
		except FileNotFoundError:
			print("{} already cleaned".format(self.file))

	def clean(self):
		'''
		clean all
		'''
		self.clean_vf()
		try:
			self.fs_neg.fs.close()
			os.remove(self.fs_neg.file)
		except:
			pass
		try:
			self.se_neg.se.close()
			os.remove(self.se_neg.file)
		except:
			pass
		try:
			self.se_neg.fs.close()
			os.remove(self.se_neg.file)
		except:
			pass
		# removes also negative files
		try:
			os.remove('{}_fs.h5'.format(self.file[:-3]))
			os.remove('{}_fs_se.h5'.format(self.file[:-3]))
		except:
			pass

	### easy access to data via properties ###
	@property
	def _thresholds(self):
	    if '/Threshold' in self.va:
	        return self.va.root.Threshold
	    else:
	        return None

	@property
	def _tp(self):
	    if '/TP' in self.va:
	        return self.va.root.TP
	    else:
	        return None

	@property
	def _fn(self):
	    if '/FN' in self.va:
	        return self.va.root.FN
	    else:
	        return None

	@property
	def _tn(self):
	    if '/TN' in self.va:
	        return self.va.root.TN
	    else:
	        return None

	@property
	def _fp_pos(self):
	    if '/FP_pos' in self.va:
	        return self.va.root.FP_pos
	    else:
	        return None

	@property
	def _fp_neg(self):
	    if '/FP_neg' in self.va:
	        return self.va.root.FP_neg
	    else:
	        return None

	def _validate(self, thresholds, prob):

		assert (self.mode in {'w', 'a'}), 'ValidationFamily must be opened in write mode.'

		# store thresholds
		self.va.create_carray('/', 'Threshold', obj=np.array(thresholds, dtype=np.float64), filters=self.db._compr)
		
		### positive validation
		print('validate positives')
		tp_query2tresh, fn_query2tresh, fp_pos_query2tresh = self._validate_positive(self.se_pos._res_tab[:], self._thresholds[:], self.db._prot_tab[:], prob)

		print('validate negatives')
		### negative validation
		tn_query2tresh, fp_neg_query2tresh = self._validate_negative(self.se_neg._res_tab[:], self._thresholds[:], prob)

		# store results
		self.va.create_carray('/', 'TP', obj=tp_query2tresh, filters=self.db._compr)
		self.va.create_carray('/', 'FN', obj=fn_query2tresh, filters=self.db._compr)
		self.va.create_carray('/', 'TN', obj=tn_query2tresh, filters=self.db._compr)
		self.va.create_carray('/', 'FP_pos', obj=fp_pos_query2tresh, filters=self.db._compr)
		self.va.create_carray('/', 'FP_neg', obj=fp_neg_query2tresh, filters=self.db._compr)

		# close and re-open in read mode
		self.va.close()
		self.mode = 'r'
		self.va = tables.open_file(self.file, self.mode)

	@staticmethod
	def _validate_positive(res_tab, thresholds, prot_tab, prob):
		'''
		results from the positive query set
		'''
		tp_query2tresh = np.zeros((res_tab.size, thresholds.size), dtype=np.bool)
		fn_query2tresh = np.zeros((res_tab.size, thresholds.size), dtype=np.bool)
		fp_query2tresh = np.zeros((res_tab.size, thresholds.size), dtype=np.bool)

		for q in range(res_tab.size):

			# true data
			prot_off = res_tab[q]['QueryOff']
			true_fam = prot_tab[prot_off]['FamOff']

			# pred fam
			pred_fam = res_tab[q]['FamOff']

			# score
			score = res_tab[q]['Score']

			# iterate over thresholds
			for t_off in range(thresholds.size):
			    t_val = thresholds[t_off]
			    
			    is_pred = (True if score < t_val else False) if prob else (True if score >= t_val else False)
			    
			    # TP
			    if is_pred:
			        if pred_fam==true_fam:
			            tp_query2tresh[q, t_off] = True
			        # FP
			        else:
			            fp_query2tresh[q, t_off] = True
			    # FN
			    else:
			        fn_query2tresh[q, t_off] = True

		return tp_query2tresh, fn_query2tresh, fp_query2tresh

	@staticmethod
	def _validate_negative(res_tab, thresholds, prob):
		'''
		results from the negative query set
		'''
		tn_query2tresh = np.zeros((res_tab.size, thresholds.size), dtype=np.bool)
		fp_query2tresh = np.zeros((res_tab.size, thresholds.size), dtype=np.bool)

		for q in range(res_tab.size):

			# pred fam
			pred_fam = res_tab[q]['FamOff']

			# score
			score = res_tab[q]['Score']

			# iterate over thresholds
			for t_off in range(thresholds.size):
			    t_val = thresholds[t_off]
			    
			    is_pred = (True if score < t_val else False) if prob else (True if score >= t_val else False)
			    
			    # FP
			    if is_pred:
			        fp_query2tresh[q, t_off] = True
			    # TN
			    else:
			        tn_query2tresh[q, t_off] = True

		return tn_query2tresh, fp_query2tresh
	
	@staticmethod
	def compute_precision_recall_specificity(tp_query2tresh, fn_query2tresh, tn_query2tresh, fp_pos_query2tresh, fp_neg_query2tresh, partitions=np.array([])):

		if partitions.size == 0:
			partitions = np.array([np.full(tp_query2tresh.shape[0], True)])

		part_num = partitions.shape[0]
		thresh_num = tp_query2tresh.shape[1]

		part2pre = np.zeros((part_num, thresh_num), dtype=np.float64)
		part2rec = np.zeros((part_num, thresh_num), dtype=np.float64)
		part2spe = np.zeros((part_num, thresh_num), dtype=np.float64)
		part2tp_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
		part2fn_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
		part2tn_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
		pos_part2fp_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
		neg_part2fp_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
		part2fp_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
		for p in range(part_num):
		    part = partitions[p]
		    for t in range(thresh_num):
		        tp_nr = np.sum(tp_query2tresh[:, t][part])
		        fn_nr = np.sum(fn_query2tresh[:, t][part])
		        tn_nr = np.sum(tn_query2tresh[:, t][part])
		        # FP can come from positive and negative queries
		        pos_fp_nr = np.sum(fp_pos_query2tresh[:, t][part])
		        neg_fp_nr = np.sum(fp_neg_query2tresh[:, t][part])
		        fp_nr = pos_fp_nr + neg_fp_nr
		        # compute precision, recall and specificity
		        part2pre[p, t] = (tp_nr/(tp_nr + fp_nr)) if tp_nr or fp_nr else 0
		        part2rec[p, t] = (tp_nr/(tp_nr + fn_nr)) if tp_nr or fn_nr else 0
		        part2spe[p, t] = (tn_nr/(tn_nr + neg_fp_nr)) if tn_nr or neg_fp_nr else 0
		        # store numbers        
		        part2tp_nr[p, t] = tp_nr
		        part2fn_nr[p, t] = fn_nr
		        part2tn_nr[p, t] = tn_nr
		        pos_part2fp_nr[p, t] = pos_fp_nr
		        neg_part2fp_nr[p, t] = neg_fp_nr
		        part2fp_nr[p, t] = fp_nr

		# compute number of queries
		part2query_nr = np.zeros((part_num), dtype=np.uint64)
		for p in range(part_num):
		    part2query_nr[p] = part2tp_nr[p, 0] + part2fn_nr[p, 0] + pos_part2fp_nr[p, 0]

		return part2pre, part2rec, part2spe, part2query_nr

	@staticmethod
	def _F1max(part2pre, part2rec, thresholds):
		n = part2pre * part2rec
		d = part2pre + part2rec
		part2f1 = 2 * np.divide(n, d, out=np.zeros_like(n), where=d!=0)
		part2f1_max = np.max(part2f1, axis=1)

		part2f1_tval = np.zeros(part2f1_max.shape)
		part2f1_toff = np.zeros(part2f1_max.shape, dtype=np.uint64)

		for p in range(part2f1.shape[0]):
		    toff = np.where(part2f1[p]==part2f1_max[p])[0][0]
		    part2f1_tval[p] = thresholds[toff]
		    part2f1_toff[p] = toff

		return part2f1_max, part2f1_tval, part2f1_toff

	def F1max(self, partitions=np.array([])):
		part2pre, part2rec, part2spe, part2query_nr = self.compute_precision_recall_specificity(
			self._tp[:], self._fn[:], self._tn[:], self._fp_pos[:], self._fp_neg[:], partitions)

		return self._F1max(part2pre, part2rec, self._thresholds[:])

	# precision favorable metric
	@staticmethod
	def _precision_pro_metric(part2pre, part2rec, thresholds):
		'''
		simply make precision weight twice as recall
		'''
		n = part2pre * part2pre * part2rec
		d = part2pre + part2pre + part2rec
		part2f1 = 3 * np.divide(n, d, out=np.zeros_like(n), where=d!=0)
		part2f1_max = np.max(part2f1, axis=1)

		part2f1_tval = np.zeros(part2f1_max.shape)
		part2f1_toff = np.zeros(part2f1_max.shape, dtype=np.uint64)

		for p in range(part2f1.shape[0]):
		    toff = np.where(part2f1[p]==part2f1_max[p])[0][0]
		    part2f1_tval[p] = thresholds[toff]
		    part2f1_toff[p] = toff

		return part2f1_max, part2f1_tval, part2f1_toff


class OMAmerValidationFamily(ValidationFamily):

	def __init__(self, se_pos, neg_root_taxon):
		super().__init__(se_pos, neg_root_taxon)
		self.fs = se_pos.fs
		self.neg_query_name = '{}_wo_{}_{}_{}'.format(self.db.root_taxon, self.ki.hidden_taxon, self.fs.query_species, ''.join(neg_root_taxon.split()) if neg_root_taxon else 'Random')
		self.neg_query_file = '{}{}.fa'.format(self.neg_query_path, self.neg_query_name)

	def validate(self, thresholds, prob, stree_path=None, oma_h5_path=None):

		self.search_negative(stree_path, oma_h5_path)

		self._validate(thresholds, prob)

	def search_negative(self, stree_path=None, oma_h5_path=None):

		# if negative do not exist, get them
		if not os.path.exists(self.neg_query_file):

			# clade-specific negatives
			if self.neg_root_taxon:
				seqs, ids = self.get_clade_specific_negatives(stree_path, oma_h5_path)

			# random negatives
			else:
				seqs, ids = self.get_random_negatives()

			# store negatives
			with open(self.neg_query_file, 'w') as ff:
				for i, s in enumerate(seqs):
				    ff.write(">{}\n{}\n".format(ids[i], s))
		else:
			sb = SequenceBuffer(fasta_file=self.neg_query_file)
			seqs = list(sb)
			ids = list(sb.ids)

		# flat search of negatives
		name = '{}_fs'.format('_'.join(self.name.split('_')[len(self.ki.name.split('_')):]))
		fs_neg = FlatSearch(self.ki, name=name)
		if fs_neg.mode != 'r':
		    fs_neg.flat_search(seqs=seqs, ids=ids)

		# search of negatives
		se_neg = Search(fs_neg, name='se', score=self.se_pos.score, cum_mode=self.se_pos.cum_mode)
		if se_neg.mode != 'r':
		    se_neg.search()

		self.fs_neg = fs_neg
		self.se_neg = se_neg

	def get_clade_specific_negatives(self, stree_path, oma_h5_path):
		'''
		1.	precompute clade taxonomic levels
		2.	iterate over hog tab and store families with all taxa within clade of interest and with >6 members
		3.	iterate over entry tab and store candidate negatives inside fam2protoffsets (mem?)
		4.	reduce to fam2protoff by randomly selecting one protein per family
		5.	randomize fam2protoff values and select the appropriate number (enough?)
		6.	parse the sequences and ids of these proteins
		'''
		random.seed(123)
		np.random.seed(123)

		print(' - gather clade specific taxa to restrict OMA families')
		clade_taxa, clade_species = IndexValidation.get_clade_specific_taxa(stree_path, self.neg_root_taxon)

		# load OMA h5 file and create pointers
		h5file = open_file(oma_h5_path, mode="r")
		hog_tab = h5file.root.HogLevel
		gen_tab = h5file.root.Genome[:]
		ent_tab = h5file.root.Protein.Entries
		seq_buffer = h5file.root.Protein.SequenceBuffer

		# load families in memory (not sure if necessary actually)
		families = hog_tab.col('Fam')

		print(' - gather all OMA families from the clade')
		clade_families = set()

		curr_fam = families[0]
		i = 0
		j = 0

		for fam in tqdm(families):
		    if fam != curr_fam:
		        
		        fam_taxa = hog_tab[i:j]['Level']
		        idx = np.searchsorted(clade_taxa, fam_taxa)
		        
		        # check if all fam_taxa are neg_root_taxon specific
		        if np.sum(clade_taxa[idx]==fam_taxa)==fam_taxa.size:
		            clade_families.add(fam)
		            
		        # move pointer and update current family
		        i = j
		        curr_fam = fam
		    j += 1

		fam_taxa = hog_tab[i:j]['Level']
		idx = np.searchsorted(clade_taxa, fam_taxa)
		if np.sum(clade_taxa[idx]==fam_taxa)==fam_taxa.size:
		    clade_families.add(fam)

		clade_families = np.array(sorted(clade_families))

		print(' - gather proteins of these families')
		fam2ent_offsets = collections.defaultdict(list)

		for r in tqdm(gen_tab):
		    sp = r['SciName']
		    
		    # load a slice if species in taxonomy
		    if sp in clade_species:
		        entry_off = r['EntryOff']
		        entry_num = r['TotEntries']
		        
		        # sub entry table for species
		        sp_ent_tab = ent_tab[entry_off: entry_off + entry_num]
		        
		        for ent in sp_ent_tab:
		            hog = ent['OmaHOG']
		            
		            if hog:
		                fam = int(ent['OmaHOG'].split(b'.')[0][4:])
		                
		                if fam in clade_families:
		                    fam2ent_offsets[fam].append(ent['EntryNr'] - 1)

		print(' - select randomly one protein per family')
		fam2ent_off = dict()

		for fam, ent_offsets in tqdm(fam2ent_offsets.items()):
		    
		    # same filter than positive queries
		    if len(ent_offsets) >= self.db.min_prot_nr and len(ent_offsets) <= self.db.max_prot_nr:
		        
		        fam2ent_off[fam] = random.choice(ent_offsets)

		print(' - select randomly one negative per positive query')
		neg_ent_offsets = np.random.permutation(list(fam2ent_off.values()))[:self.se_pos.fs._query_id.nrows]

		print(' - collect negative sequences and identifiers')
		seqs = []
		ids = []
		for ent_off in tqdm(neg_ent_offsets):
		    seq_off = ent_tab[ent_off]['SeqBufferOffset']
		    seqs.append(b''.join(seq_buffer[seq_off: seq_off + ent_tab[ent_off]['SeqBufferLength'] - 1]).decode('ascii'))
		    ids.append(ent_off + 1)

		return seqs, ids

	def get_random_negatives(self):

		def _simulate_query(ql, a_freq):
		    return "".join(np.random.choice(list(a_freq.keys()), ql, p=list(a_freq_scaled.values())))

		np.random.seed(123)

		# load in memory
		prot_tab = self.db._prot_tab[:]
		pos_qoffsets = self.fs._query_id[:].flatten()

		# amino acid frequencies from uniprot
		a_freq = {"A":0.0825,"C":0.0137,"D":0.0545,"E":0.0675,"F":0.0386,"G":0.0707,"H":0.0227,"I":0.0596,"K":0.0584,"L":0.0966,
		          "M":0.0242,"N":0.0406,"P":0.0470,"Q":0.0393,"R":0.0553,"S":0.0656,"T":0.0534,"V":0.0687,"W":0.0108,"Y":0.0292}

		sum_a_freq = sum(a_freq.values())

		# they do not sum to one. scale them to one
		a_freq_scaled = {k:v/sum_a_freq for k, v in a_freq.items()}

		seqs = [_simulate_query(prot_tab[q]['SeqLen'], a_freq_scaled) for q in self.fs._query_id[:].flatten()]

		return seqs, list(range(len(seqs)))


class DIAMONDvalidationFamily(ValidationFamily):

	def __init__(self, se_pos, neg_root_taxon):
		super().__init__(se_pos, neg_root_taxon)
		self.fs = se_pos
		self.neg_query_name = '{}_wo_{}_{}_{}'.format(self.db.root_taxon, self.ki.hidden_taxon, self.fs.query_species, ''.join(neg_root_taxon.split()) if neg_root_taxon else 'Random')
		self.neg_query_file = '{}{}.fa'.format(self.neg_query_path, self.neg_query_name)

	def initiate_negative(self):

		assert os.path.exists(self.neg_query_file), 'Negative queries missing'

		qr_name = '_'.join(self.name.split('_')[len(self.ki.name.split('_')):])
		name = '{}_fs'.format(qr_name)

		fs_neg = DIAMONDsearch(self.ki, name=name, fasta=self.neg_query_file, qr_name=qr_name)
		if fs_neg.mode != 'r':
			fs_neg.export_query_fasta()

		self.fs_neg = fs_neg
		self.fs_neg.fs.close()

	def validate(self, thresholds):

		self.import_negative()

		self._validate(thresholds, True)

	def import_negative(self):

		qr_name = '_'.join(self.name.split('_')[len(self.ki.name.split('_')):])
		name = '{}_fs'.format(qr_name)

		fs_neg = DIAMONDsearch(self.ki, name=name, fasta=self.neg_query_file, qr_name=qr_name, mode='a')
		fs_neg.import_blast_result('evalue', True)

		self.se_neg = fs_neg


class SWvalidationFamily(ValidationFamily):
    '''
    only positives queries
    '''
    def __init__(self, se_pos):

        assert se_pos.mode == 'r', 'Search must be opened in read mode.'

        self.db = se_pos.db
        self.ki = se_pos.ki
        self.se_pos = se_pos
        
        self.name = "{}_Family".format(self.se_pos.name)
        self.path = self.se_pos.path
        self.file = "{}{}.h5".format(self.path, self.name)
        
        if os.path.isfile(self.file):
            self.mode = 'r'
            self.va = tables.open_file(self.file, self.mode)
        else:
            self.mode = 'w'
            self.va = tables.open_file(self.file, self.mode)
            
    def validate(self, thresholds, prob):

        assert (self.mode in {'w', 'a'}), 'ValidationFamily must be opened in write mode.'

        # store thresholds
        self.va.create_carray('/', 'Threshold', obj=np.array(thresholds, dtype=np.float64), filters=self.db._compr)

        ### positive validation
        tp_query2tresh, fn_query2tresh, fp_pos_query2tresh = self._validate_positive(self.se_pos._res_tab[:], self._thresholds[:], self.db._prot_tab[:], prob)

        # store results
        self.va.create_carray('/', 'TP', obj=tp_query2tresh, filters=self.db._compr)
        self.va.create_carray('/', 'FN', obj=fn_query2tresh, filters=self.db._compr)
        self.va.create_carray('/', 'FP_pos', obj=fp_pos_query2tresh, filters=self.db._compr)

        # close and re-open in read mode
        self.va.close()
        self.mode = 'r'
        self.va = tables.open_file(self.file, self.mode)


# class Validation():

# 	def __init__(self, se, name=None, path=None, val_mode='custom'):
# 		'''this class gives the framework for the validation of SPHOG.'''

# 		assert se.db.mode == 'r', 'Database must be opened in read mode.'
# 		assert se.ki.mode == 'r', 'Index must be opened in read mode.'
# 		assert se.fs.mode == 'r', 'FlatSearch must be opened in read mode.'
# 		assert se.mode == 'r', 'Search must be opened in read mode.'
# 		assert se.fs.query_species, 'The Search object must include a FlatSearchValidation object with a query species'

# 		self.db = se.db
# 		self.ki = se.ki
# 		self.fs = se.fs
# 		self.se = se

# 		# hdf5 file
# 		self.name = "{}_{}".format(self.se.name, name if name else '{}val'.format(val_mode))
# 		self.path = path if path else self.se.path
# 		self.file = "{}{}.h5".format(self.path, self.name)

# 		if os.path.isfile(self.file):
# 			self.mode = 'r'
# 			self.va = tables.open_file(self.file, self.mode)

# 		else:
# 			self.mode = 'w'
# 			self.va = tables.open_file(self.file, self.mode)

# 		# mode of validation
# 		self.val_mode = val_mode

# 	def __enter__(self):
# 	    return self

# 	def __exit__(self, *_):
# 	    self.va.close()
	    
# 	def clean(self):
# 	    '''
# 	    close and remove hdf5 file
# 	    '''
# 	    self.__exit__()
# 	    try:
# 	        os.remove(self.file)
# 	    except FileNotFoundError:
# 	        print("{} already cleaned".format(self.file))

# 	# # tax and HOG filters --> moved to IndexValidation

# 	### easy access to data via properties ###
# 	@property
# 	def _thresholds(self):
# 	    if '/Threshold' in self.va:
# 	        return self.va.root.Threshold
# 	    else:
# 	        return None

# 	@property
# 	def _tp_pre(self):
# 	    if '/TP_pre' in self.va:
# 	        return self.va.root.TP_pre
# 	    else:
# 	        return None

# 	@property
# 	def _tp_rec(self):
# 	    if '/TP_rec' in self.va:
# 	        return self.va.root.TP_rec
# 	    else:
# 	        return None

# 	@property
# 	def _fp(self):
# 	    if '/FP' in self.va:
# 	        return self.va.root.FP
# 	    else:
# 	        return None

# 	@property
# 	def _fn(self):
# 	    if '/FN' in self.va:
# 	        return self.va.root.FN
# 	    else:
# 	        return None

# 	####################################################################################################################################
# 	def validate(self, thresholds, pv=False, hog2x=None, x_num=None, hf_lca=True, hf_lcasis=False):

# 		assert (self.mode in {'w', 'a'}), 'Validation must be opened in write mode.'

# 		# store thresholds
# 		self.va.create_carray('/', 'Threshold', obj=np.array(thresholds, dtype=np.float64), filters=self.db._compr)

# 		bestpath_mask = self.se._bestpath_mask[:] 
# 		hog_score = self.se._hog_score[:]

# 		# run validation by loading once in memory the required args
# 		tp_pre_query2tax2tresh, tp_rec_query2tax2tresh, fn_query2tax2tresh, fp_query2tax2tresh = self._validate(self._thresholds[:], self.se._res_tab[:], 
# 			self.db._prot_tab[:], self.db._fam_tab[:], self.db._hog_tab.col('ParentOff'), bestpath_mask, hog_score, get_root_leaf_hog_offsets, 
# 			self.ki.idx_type, pv, hog2x, x_num, self.ki.hog_filter_lca if hf_lca else np.array([]), self.ki.hog_filter_lcasis if hf_lcasis else np.array([]), self.val_mode)

# 		# store results
# 		self.va.create_carray('/', 'TP_pre', obj=tp_pre_query2tax2tresh, filters=self.db._compr)
# 		self.va.create_carray('/', 'TP_rec', obj=tp_rec_query2tax2tresh, filters=self.db._compr)
# 		self.va.create_carray('/', 'FN', obj=fn_query2tax2tresh, filters=self.db._compr)
# 		self.va.create_carray('/', 'FP', obj=fp_query2tax2tresh, filters=self.db._compr)

# 		# close and re-open in read mode
# 		self.va.close()
# 		self.mode = 'r'
# 		self.va = tables.open_file(self.file, self.mode)

# 		# delete cache
# 		del self.ki.tax_filter, self.ki.hog_filter_lca, self.ki.hog_filter_lcasis

# 	@staticmethod
# 	def _validate(thresholds, res_tab, prot_tab, fam_tab, hog2parent, bestpath_mask, hog_score, fun_root_leaf, idx_type, pv, hog2x, x_num, hog_filter_lca, hog_filter_lcasis, val_mode):
# 		'''
# 		args:
# 		 - pv: whether p-value type of score (the lower the better)
# 		 - hog2x: mapper between HOGs and any HOG grouping such as taxonomy or amount of duplications in query sister
# 		 - x_num: number of group in hog2x mapper
# 		'''
# 		def _compute_tp_fp_fn(tp_hogs, fn_hogs, fp_hogs, hog2x, x_num, tp_pre_query2x2tresh, tp_rec_query2x2tresh, 
# 			fn_query2x2tresh, fp_query2x2tresh, val_mode):
# 			'''
# 			3 options to define TPs, FNs and FPs
# 			'''
# 			# map to bins and keep track of HOG numbers
# 			tp_x, tp_nr = np.unique(hog2x[tp_hogs], return_counts=True) if x_num else tp_hogs
# 			fn_x, fn_nr = np.unique(hog2x[fn_hogs], return_counts=True) if x_num else fn_hogs
# 			fp_x, fp_nr = np.unique(hog2x[fp_hogs], return_counts=True) if x_num else fp_hogs

# 			# my custom approach: split TPs in to recall and precision TPs
# 			if val_mode == 'custom':
# 			    tp_x_pre = np.setdiff1d(tp_x, fp_x)
# 			    tp_x_rec = np.setdiff1d(tp_x, fn_x)

# 			    # and counts TPs, FPs and FNs by query
# 			    tp_pre_query2x2tresh[q, tp_x_pre, t_off] = 1
# 			    tp_rec_query2x2tresh[q, tp_x_rec, t_off] = 1
# 			    fn_query2x2tresh[q, fn_x, t_off] = 1
# 			    fp_query2x2tresh[q, fp_x, t_off] = 1

# 			# stringent approach ignoring hierarchy
# 			elif val_mode == 'stringent':   
# 			    tp_x = np.setdiff1d(tp_x, np.union1d(fn_x, fp_x))

# 			    #also counts TPs, FPs and FNs by query
# 			    tp_pre_query2x2tresh[q, tp_x, t_off] = 1
# 			    tp_rec_query2x2tresh[q, tp_x, t_off] = 1
# 			    fn_query2x2tresh[q, fn_x, t_off] = 1
# 			    fp_query2x2tresh[q, fp_x, t_off] = 1

# 			# approach where TPs, FPs and FNs are counted by HOG
# 			elif val_mode == 'golike':
# 			    tp_pre_query2x2tresh[q, tp_x, t_off] = tp_nr
# 			    tp_rec_query2x2tresh[q, tp_x, t_off] = tp_nr
# 			    fn_query2x2tresh[q, fn_x, t_off] = fn_nr
# 			    fp_query2x2tresh[q, fp_x, t_off] = fp_nr

# 		thresholds=np.array(thresholds, dtype=np.float64)

# 		# store validation results
# 		tp_pre_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)
# 		tp_rec_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)
# 		fn_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)
# 		fp_query2x2tresh = np.zeros((res_tab.size, x_num if x_num else hog2parent.size, thresholds.size), dtype=np.uint16)

# 		# iterage over queries
# 		for q in tqdm(range(res_tab.size)):

# 			# true data
# 			prot_off = res_tab[q]['QueryOff']
# 			true_fam = prot_tab[prot_off]['FamOff']
# 			true_leafhog = prot_tab[prot_off]['HOGoff']
# 			true_hogs = fun_root_leaf(true_leafhog, hog2parent)

# 			# remove hogs specific to hidden taxa from true hogs
# 			#true_hogs = true_hogs[~tax_filter[hog2lcatax[true_hogs]]]
# 			if hog_filter_lca.size > 0:
# 				true_hogs = true_hogs[~hog_filter_lca[true_hogs]]

# 			# pred fam
# 			pred_fam = res_tab[q]['FamOff']

# 			if idx_type == 'lca':

# 			    # hogs of pred fam
# 			    hog_off = fam_tab[pred_fam]['HOGoff']
# 			    hog_offsets = np.arange(hog_off, hog_off + fam_tab[pred_fam]['HOGnum'], dtype=np.uint64)
			    
# 			    # best path hogs and score
# 			    hogs_mask = bestpath_mask[q][hog_offsets]
# 			    hogs_bestpath = hog_offsets[hogs_mask]
# 			    hogs_bestpath_score = hog_score[q][hogs_bestpath]

# 			elif idx_type == 'cs':

# 			    pred_leafhog = res_tab[q]['HOGoff']
# 			    score = res_tab[q]['Score']

# 			    # cs can have NA result (-1) with BLAST
# 			    if pred_leafhog != -1:
# 			    	hogs_bestpath = fun_root_leaf(pred_leafhog, hog2parent)
# 			    else:
# 			    	hogs_bestpath = np.array([pred_leafhog], dtype=np.uint64)
			    
# 			    # some broadcasting to use same function later
# 			    hogs_bestpath_score = np.full(hogs_bestpath.shape, score)

# 			# iterate over thresholds
# 			for t_off in range(thresholds.size):
# 				t_val = thresholds[t_off]

# 				# pred hogs
# 				pred_hogs = hogs_bestpath[(hogs_bestpath_score < t_val) if pv else (hogs_bestpath_score >= t_val)]
				
# 				if hog_filter_lcasis.size > 0:
# 					# remove HOGs with only sister HOGs that are specific to hidden taxa
# 					pred_hogs = pred_hogs[~hog_filter_lcasis[pred_hogs]]

# 				# confront true classes against predicted classes to get benchmark results
# 				# not supported by numba ...
# 				tp_hogs = np.intersect1d(true_hogs, pred_hogs, assume_unique=True)
# 				fn_hogs = np.setdiff1d(true_hogs, pred_hogs, assume_unique=True)
# 				fp_hogs = np.setdiff1d(pred_hogs, true_hogs, assume_unique=True)

# 				# map to bins after benchmark if bins
# 				tp_x = np.unique(hog2x[tp_hogs]) if x_num else tp_hogs
# 				fn_x = np.unique(hog2x[fn_hogs]) if x_num else fn_hogs
# 				fp_x = np.unique(hog2x[fp_hogs]) if x_num else fp_hogs

# 				_compute_tp_fp_fn(tp_hogs, fn_hogs, fp_hogs, hog2x, x_num, tp_pre_query2x2tresh, tp_rec_query2x2tresh,
# 					fn_query2x2tresh, fp_query2x2tresh, val_mode)

# 				# # after binning, TP are different for precision and recall calculation
# 				# # for precision, TP are TP if they are not also FP
# 				# tp_x_pre = np.setdiff1d(tp_x, fp_x)
# 				# # for recall, TP are TP if not also FN
# 				# tp_x_rec = np.setdiff1d(tp_x, fn_x)     

# 				# tp_pre_query2x2tresh[q, tp_x_pre, t_off] = True
# 				# tp_rec_query2x2tresh[q, tp_x_rec, t_off] = True
# 				# fn_query2x2tresh[q, fn_x, t_off] = True
# 				# fp_query2x2tresh[q, fp_x, t_off] = True

# 		return tp_pre_query2x2tresh, tp_rec_query2x2tresh, fn_query2x2tresh, fp_query2x2tresh

# 	### bin taxa #################################################################################################################################
# 	@staticmethod
# 	def bin_taxa(species_tree_path, root_tax_level, taxoff2tax, tax_tab, query_species, hog2taxoff, bin_num, root=False, merge_post_lca_taxa=True):

# 		def _bin_taxa(bin_num, tax2dist, lca_tax, root, merge_post_lca_taxa):
		    
# 		    # remove one bin if not merging taxa descending from lca taxon
# 		    bin_num = bin_num if merge_post_lca_taxa else bin_num - 1
		    
# 		    # grab distance of lca taxon
# 		    lca_dist = tax2dist[lca_tax] if lca_tax else max(tax2dist.values())
		    
# 		    if root:
# 		        dist_range_size = lca_dist / (bin_num - 1)
# 		        dist_ranges = [-1] + [dist_range_size*n for n in range(0, bin_num)]
# 		    else:
# 		        dist_range_size = lca_dist / bin_num
# 		        dist_ranges = [-1] + [dist_range_size*n for n in range(1, bin_num + 1)]
# 		    print(dist_ranges)    
		    
# 		    tax2taxbin = {}

# 		    # fill bins with taxa within distance ranges
# 		    for bn in range(bin_num):
# 		        bin_taxa = {k for k,v in tax2dist.items() if v > dist_ranges[bn] and v <= dist_ranges[bn + 1]}
# 		        tax2taxbin.update(dict(zip(bin_taxa, repeat(bn))))
		    
# 		    # deal with taxa descending from lca taxon
# 		    max_dist = max(tax2dist.values())
# 		    post_lca_taxa = {k for k,v in tax2dist.items() if v > lca_dist and v <= max_dist}
# 		    if merge_post_lca_taxa:
# 		        tax2taxbin.update(dict(zip(post_lca_taxa, repeat(bin_num - 1))))
# 		    else:
# 		        tax2taxbin.update(dict(zip(post_lca_taxa, repeat(bin_num))))

# 		    return tax2taxbin
		
# 		# select subtree
# 		stree = Tree(species_tree_path, format=1, quoted_node_names=True)
# 		tl_stree = [x for x in stree.traverse() if x.name == root_tax_level][0]
# 		tax2dist = {''.join(x.name.split()):x.get_distance(tl_stree) for x in tl_stree.traverse()}

# 		# grab lca taxon between query and reference
# 		lca_tax = taxoff2tax[tax_tab[np.searchsorted(taxoff2tax, query_species.encode('ascii'))]['ParentOff']].decode('ascii')

# 		tax2taxbin = _bin_taxa(bin_num, tax2dist, lca_tax, root, merge_post_lca_taxa)

# 		taxoff2taxbin = {np.searchsorted(taxoff2tax, tax.encode('ascii')):b for tax, b in tax2taxbin.items()}

# 		# because opistokonta error, add an exception with -1
# 		return np.array([taxoff2taxbin.get(tax, -1) for tax in hog2taxoff], np.int64), tax2taxbin

	
# 	def compute_inparalog_coverage(self, htax):

# 		tax_off = np.searchsorted(self.db._tax_tab.col('ID'), htax)
# 		hidden_taxa = [np.searchsorted(self.db._tax_tab.col('ID'), x) for x in self.ki.hidden_species]

# 		# compute inparalog coverage for all queries of htax
# 		sispecies_cands = get_sispecies_candidates(tax_off, self.db._tax_tab[:], self.db._ctax_arr[:], hidden_taxa)

# 		query_ids = self.fs._query_id[:].flatten()
# 		prot_tab = self.db._prot_tab[:]
# 		cprot_buff = self.db._cprot_arr[:]
# 		hog_tab = self.db._hog_tab[:]
# 		chog_buff = self.db._chog_arr[:]

# 		ip_covs = np.array([compute_inparalog_coverage_new(q, query_ids, prot_tab, 
# 			cprot_buff, hog_tab, chog_buff, hidden_taxa, sispecies_cands) for q in range(query_ids.size)])

# 		return ip_covs

# 	@staticmethod
# 	def partition_queries(thresholds, query_values, parameter_name):
# 		part_names = []
# 		partitions = []
# 		curr_thresh = -1
# 		for thresh in thresholds:
# 		    part = np.full(query_values.size, False)
# 		    part[(query_values > curr_thresh) & (query_values <= thresh)] = True
# 		    partitions.append(part)
# 		    part_names.append('{} < {} <= {}'.format(curr_thresh if curr_thresh != -1 else 0, parameter_name, thresh))
# 		    curr_thresh = thresh

# 		# last one
# 		part = np.full(query_values.size, False)
# 		part[query_values > curr_thresh] = True
# 		partitions.append(part)
# 		part_names.append('{} < {} <= {}'.format(curr_thresh if curr_thresh != -1 else 0, parameter_name, 1))

# 		return np.array(partitions), part_names


# 	@staticmethod
# 	def compute_precision_recall(tp_pre_query2bin2tresh, tp_rec_query2bin2tresh, fn_query2bin2tresh, fp_query2bin2tresh, partitions=np.array([])):

# 		if partitions.size == 0:
# 		    partitions = np.array([np.full(tp_pre_query2bin2tresh.shape[0], True)])

# 		part_num = partitions.shape[0]
# 		bin_num = tp_pre_query2bin2tresh.shape[1]
# 		thresh_num = tp_pre_query2bin2tresh.shape[2]

# 		part2bin2pre = np.zeros((part_num, bin_num, thresh_num), dtype=np.float64)
# 		part2bin2rec = np.zeros((part_num, bin_num, thresh_num), dtype=np.float64)
# 		part2bin2tp_pre_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)
# 		part2bin2tp_rec_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)
# 		part2bin2fn_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)
# 		part2bin2fp_nr = np.zeros((part_num, bin_num, thresh_num), dtype=np.uint64)

# 		for p in range(part_num):
# 		    part = partitions[p]
# 		    for b in range(bin_num):
# 		        for t in range(thresh_num):
# 		            tp_pre_nr = np.sum(tp_pre_query2bin2tresh[:, b, t][part])
# 		            tp_rec_nr = np.sum(tp_rec_query2bin2tresh[:, b, t][part])
# 		            fn_nr = np.sum(fn_query2bin2tresh[:, b, t][part])
# 		            fp_nr = np.sum(fp_query2bin2tresh[:, b, t][part])
# 		            part2bin2pre[p, b, t] = (tp_pre_nr/(tp_pre_nr + fp_nr)) if tp_pre_nr or fp_nr else 0
# 		            part2bin2rec[p, b, t] = (tp_rec_nr/(tp_rec_nr + fn_nr)) if tp_rec_nr or fn_nr else 0
# 		            part2bin2tp_pre_nr[p, b, t] = tp_pre_nr
# 		            part2bin2tp_rec_nr[p, b, t] = tp_rec_nr
# 		            part2bin2fn_nr[p, b, t] = fn_nr
# 		            part2bin2fp_nr[p, b, t] = fp_nr

# 		part2bin2query_nr = np.zeros((part_num, bin_num), dtype=np.uint64)

# 		for p in range(part_num):
# 		    for b in range(bin_num):
# 		        part2bin2query_nr[p, b] = part2bin2tp_rec_nr[p, b, 0] + part2bin2fn_nr[p, b, 0]
		        
# 		return part2bin2pre, part2bin2rec, part2bin2query_nr

# 	def F1(self, part2bin2pre, part2bin2rec):
# 		n = part2bin2pre * part2bin2rec
# 		d = part2bin2pre + part2bin2rec
# 		part2bin2f1 = 2 * np.divide(n, d, out=np.zeros_like(n), where=d!=0)
# 		part2bin2f1_max = np.max(part2bin2f1, axis=2)

# 		part2bin2f1_tval = np.zeros(part2bin2f1_max.shape)
# 		part2bin2f1_toff = np.zeros(part2bin2f1_max.shape, dtype=np.uint64)

# 		for p in range(part2bin2f1.shape[0]):
# 			for b in range(part2bin2f1.shape[1]):
# 				toff = np.where(part2bin2f1[p, b]==part2bin2f1_max[p, b])[0][0]
# 				part2bin2f1_tval[p, b] = self._thresholds[:][toff]
# 				part2bin2f1_toff[p, b] = toff
# 		return part2bin2f1_max, part2bin2f1_tval, part2bin2f1_toff


# 	### plot ###############################################################################################################################################
# 	@staticmethod
# 	def plot_precision_recall(precisions_list, recalls_list, name_list, n_pos_list, colors, line_styles, marker_styles, title):
# 		plt.rcParams.update({'font.size': 20})
# 		fig=plt.figure(figsize=(10, 6))
# 		for i, name in enumerate(name_list):   
# 		    plt.plot(recalls_list[i], precisions_list[i], label='{} n={}'.format(name, n_pos_list[i]) if n_pos_list[i] else '',markeredgewidth=1.5, marker=marker_styles[i],markersize=20, color=colors[i], linewidth = 6, linestyle = line_styles[i])
# 		#plt.plot([-0.1, 1], [-0.1, 1], 'k-')
# 		#marker=marker_styles[i],markersize=1
# 		axes=plt.gca()
# 		axes.set_ylim([0,1])
# 		axes.set_xlim([0,1])
# 		plt.xlabel("sensitivity = TP / (TP + FN)")
# 		plt.ylabel("precison = TP / (TP + FP)")
# 		plt.title(title, fontsize=24)
# 		plt.legend(loc=3)
# 		plt.tight_layout()

# 		plt.show()
# 		return fig


# class ValidationFamily():

# 	def __init__(self, se_pos, neg_root_taxon):

# 		assert se_pos.mode == 'r', 'Search must be opened in read mode.'

# 		self.db = se_pos.db
# 		self.ki = se_pos.ki
# 		self.fs = se_pos.fs
# 		self.se_pos = se_pos

# 		# hdf5 file
# 		self.neg_root_taxon = neg_root_taxon 
# 		self.name = "{}_{}".format(self.se_pos.name, neg_root_taxon if neg_root_taxon else 'Random')
# 		self.path = self.se_pos.path
# 		self.file = "{}{}.h5".format(self.path, self.name)

# 		if os.path.isfile(self.file):
# 			self.mode = 'r'
# 			self.va = tables.open_file(self.file, self.mode)
# 		else:
# 			self.mode = 'w'
# 			self.va = tables.open_file(self.file, self.mode)

# 		# negative queries
# 		self.neg_query_path = '{}neg_query/'.format(self.db.path)
# 		self.neg_query_name = '{}_wo_{}_{}_{}'.format(self.db.root_taxon, self.ki.hidden_taxon, self.fs.query_species, ''.join(neg_root_taxon.split()) if neg_root_taxon else 'Random')
# 		self.neg_query_file = '{}{}.fa'.format(self.neg_query_path, self.neg_query_name)
	
# 	def __enter__(self):
# 		return self

# 	def __exit__(self, *_):
# 		self.va.close()

# 	def clean(self):
# 		'''
# 		close and remove hdf5 file
# 		TO DO: close statements should be in __exit__
# 		'''
# 		self.__exit__()
# 		try:
# 			os.remove(self.file)
# 		except FileNotFoundError:
# 			print("{} already cleaned".format(self.file))
# 		try:
# 			self.fs_neg.fs.close()
# 			os.remove(self.fs_neg.file)
# 		except:
# 			pass
# 		try:
# 			self.se_neg.se.close()
# 			os.remove(self.se_neg.file)
# 		except:
# 			pass
# 	### easy access to data via properties ###
# 	@property
# 	def _thresholds(self):
# 	    if '/Threshold' in self.va:
# 	        return self.va.root.Threshold
# 	    else:
# 	        return None

# 	@property
# 	def _tp(self):
# 	    if '/TP' in self.va:
# 	        return self.va.root.TP
# 	    else:
# 	        return None

# 	@property
# 	def _fn(self):
# 	    if '/FN' in self.va:
# 	        return self.va.root.FN
# 	    else:
# 	        return None

# 	@property
# 	def _tn(self):
# 	    if '/TN' in self.va:
# 	        return self.va.root.TN
# 	    else:
# 	        return None

# 	@property
# 	def _fp_pos(self):
# 	    if '/FP_pos' in self.va:
# 	        return self.va.root.FP_pos
# 	    else:
# 	        return None

# 	@property
# 	def _fp_neg(self):
# 	    if '/FP_neg' in self.va:
# 	        return self.va.root.FP_neg
# 	    else:
# 	        return None

# 	def _validate(self, thresholds, prob):

# 		assert (self.mode in {'w', 'a'}), 'ValidationFamily must be opened in write mode.'

# 		# store thresholds
# 		self.va.create_carray('/', 'Threshold', obj=np.array(thresholds, dtype=np.float64), filters=self.db._compr)
		
# 		### positive validation
# 		tp_query2tresh, fn_query2tresh, fp_pos_query2tresh = self._validate_positive(self.se_pos._res_tab[:], self._thresholds[:], self.db._prot_tab[:], self.ki.idx_type, prob)

# 		### negative validation
# 		tn_query2tresh, fp_neg_query2tresh = self._validate_negative(self.se_neg._res_tab[:], self._thresholds[:], self.ki.idx_type, prob)

# 		# store results
# 		self.va.create_carray('/', 'TP', obj=tp_query2tresh, filters=self.db._compr)
# 		self.va.create_carray('/', 'FN', obj=fn_query2tresh, filters=self.db._compr)
# 		self.va.create_carray('/', 'TN', obj=tn_query2tresh, filters=self.db._compr)
# 		self.va.create_carray('/', 'FP_pos', obj=fp_pos_query2tresh, filters=self.db._compr)
# 		self.va.create_carray('/', 'FP_neg', obj=fp_neg_query2tresh, filters=self.db._compr)

# 		# close and re-open in read mode
# 		self.va.close()
# 		self.mode = 'r'
# 		self.va = tables.open_file(self.file, self.mode)

# 	@staticmethod
# 	def _validate_positive(res_tab, thresholds, prot_tab, idx_type, prob):
# 		'''
# 		results from the positive query set
# 		'''
# 		tp_query2tresh = np.zeros((res_tab.size, thresholds.size), dtype=np.bool)
# 		fn_query2tresh = np.zeros((res_tab.size, thresholds.size), dtype=np.bool)
# 		fp_query2tresh = np.zeros((res_tab.size, thresholds.size), dtype=np.bool)

# 		for q in range(res_tab.size):

# 			# true data
# 			prot_off = res_tab[q]['QueryOff']
# 			true_fam = prot_tab[prot_off]['FamOff']

# 			# pred fam
# 			pred_fam = res_tab[q]['FamOff']

# 			# score
# 			score = (res_tab[q]['FamScore'] if idx_type == 'lca' else res_tab[q]['Score'])

# 			# iterate over thresholds
# 			for t_off in range(thresholds.size):
# 			    t_val = thresholds[t_off]
			    
# 			    is_pred = (True if score < t_val else False) if prob else (True if score >= t_val else False)
			    
# 			    # TP
# 			    if is_pred:
# 			        if pred_fam==true_fam:
# 			            tp_query2tresh[q, t_off] = True
# 			        # FP
# 			        else:
# 			            fp_query2tresh[q, t_off] = True
# 			    # FN
# 			    else:
# 			        fn_query2tresh[q, t_off] = True

# 		return tp_query2tresh, fn_query2tresh, fp_query2tresh

# 	@staticmethod
# 	def _validate_negative(res_tab, thresholds, idx_type, prob):
# 		'''
# 		results from the negative query set
# 		'''
# 		tn_query2tresh = np.zeros((res_tab.size, thresholds.size), dtype=np.bool)
# 		fp_query2tresh = np.zeros((res_tab.size, thresholds.size), dtype=np.bool)

# 		for q in range(res_tab.size):

# 			# pred fam
# 			pred_fam = res_tab[q]['FamOff']

# 			# score
# 			score = (res_tab[q]['FamScore'] if idx_type == 'lca' else res_tab[q]['Score'])

# 			# iterate over thresholds
# 			for t_off in range(thresholds.size):
# 			    t_val = thresholds[t_off]
			    
# 			    is_pred = (True if score < t_val else False) if prob else (True if score >= t_val else False)
			    
# 			    # FP
# 			    if is_pred:
# 			        fp_query2tresh[q, t_off] = True
# 			    # TN
# 			    else:
# 			        tn_query2tresh[q, t_off] = True

# 		return tn_query2tresh, fp_query2tresh
	
# 	@staticmethod
# 	def compute_precision_recall_specificity(tp_query2tresh, fn_query2tresh, tn_query2tresh, fp_pos_query2tresh, fp_neg_query2tresh, partitions=np.array([])):

# 		if partitions.size == 0:
# 			partitions = np.array([np.full(tp_query2tresh.shape[0], True)])

# 		part_num = partitions.shape[0]
# 		thresh_num = tp_query2tresh.shape[1]

# 		part2pre = np.zeros((part_num, thresh_num), dtype=np.float64)
# 		part2rec = np.zeros((part_num, thresh_num), dtype=np.float64)
# 		part2spe = np.zeros((part_num, thresh_num), dtype=np.float64)
# 		part2tp_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
# 		part2fn_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
# 		part2tn_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
# 		pos_part2fp_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
# 		neg_part2fp_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
# 		part2fp_nr = np.zeros((part_num, thresh_num), dtype=np.uint64)
# 		for p in range(part_num):
# 		    part = partitions[p]
# 		    for t in range(thresh_num):
# 		        tp_nr = np.sum(tp_query2tresh[:, t][part])
# 		        fn_nr = np.sum(fn_query2tresh[:, t][part])
# 		        tn_nr = np.sum(tn_query2tresh[:, t][part])
# 		        # FP can come from positive and negative queries
# 		        pos_fp_nr = np.sum(fp_pos_query2tresh[:, t][part])
# 		        neg_fp_nr = np.sum(fp_neg_query2tresh[:, t][part])
# 		        fp_nr = pos_fp_nr + neg_fp_nr
# 		        # compute precision, recall and specificity
# 		        part2pre[p, t] = (tp_nr/(tp_nr + fp_nr)) if tp_nr or fp_nr else 0
# 		        part2rec[p, t] = (tp_nr/(tp_nr + fn_nr)) if tp_nr or fn_nr else 0
# 		        part2spe[p, t] = (tn_nr/(tn_nr + neg_fp_nr)) if tn_nr or neg_fp_nr else 0
# 		        # store numbers        
# 		        part2tp_nr[p, t] = tp_nr
# 		        part2fn_nr[p, t] = fn_nr
# 		        part2tn_nr[p, t] = tn_nr
# 		        pos_part2fp_nr[p, t] = pos_fp_nr
# 		        neg_part2fp_nr[p, t] = neg_fp_nr
# 		        part2fp_nr[p, t] = fp_nr

# 		# compute number of queries
# 		part2query_nr = np.zeros((part_num), dtype=np.uint64)
# 		for p in range(part_num):
# 		    part2query_nr[p] = part2tp_nr[p, 0] + part2fn_nr[p, 0] + pos_part2fp_nr[p, 0]

# 		return part2pre, part2rec, part2spe, part2query_nr

# 	@staticmethod
# 	def _F1max(part2pre, part2rec, thresholds):
# 		n = part2pre * part2rec
# 		d = part2pre + part2rec
# 		part2f1 = 2 * np.divide(n, d, out=np.zeros_like(n), where=d!=0)
# 		part2f1_max = np.max(part2f1, axis=1)

# 		part2f1_tval = np.zeros(part2f1_max.shape)
# 		part2f1_toff = np.zeros(part2f1_max.shape, dtype=np.uint64)

# 		for p in range(part2f1.shape[0]):
# 		    toff = np.where(part2f1[p]==part2f1_max[p])[0][0]
# 		    part2f1_tval[p] = thresholds[toff]
# 		    part2f1_toff[p] = toff

# 		return part2f1_max, part2f1_tval, part2f1_toff

# 	def F1max(self, partitions=np.array([])):
# 		part2pre, part2rec, part2spe, part2query_nr = self.compute_precision_recall_specificity(
# 			self._tp[:], self._fn[:], self._tn[:], self._fp_pos[:], self._fp_neg[:], partitions)

# 		return self._F1max(part2pre, part2rec, self._thresholds[:])


# class SPHOGvalidationFamily(ValidationFamily):

# 	def __init__(self, se_pos, neg_root_taxon):
# 		super().__init__(se_pos, neg_root_taxon)

# 	def validate(self, align, thresholds, prob, stree_path=None, oma_h5_path=None):

# 		self.search_negative(align, stree_path, oma_h5_path)

# 		self._validate(thresholds, prob)

# 	def search_negative(self, align, stree_path=None, oma_h5_path=None):

# 		# if negative do not exist, get them
# 		if not os.path.exists(self.neg_query_file):

# 			# clade-specific negatives
# 			if self.neg_root_taxon:
# 				seqs, ids = self.get_clade_specific_negatives(stree_path, oma_h5_path)

# 			# random negatives
# 			else:
# 				seqs, ids = self.get_random_negatives()

# 			# store negatives
# 			with open(self.neg_query_file, 'w') as ff:
# 				for i, s in enumerate(seqs):
# 				    ff.write(">{}\n{}\n".format(ids[i], s))
# 		else:
# 			sb = SequenceBuffer(fasta_file=self.neg_query_file)
# 			seqs = list(sb)
# 			ids = list(sb.ids)

# 		# flat search of negatives
# 		name = '{}_fs'.format('_'.join(self.name.split('_')[len(self.ki.name.split('_')):]))
# 		fs_neg = FlatSearch(self.ki, name=name)
# 		if fs_neg.mode != 'r':
# 		    fs_neg.flat_search(seqs=seqs, ids=ids, align=align)

# 		# search of negatives
# 		se_neg = Search(fs_neg, name='se')
# 		if se_neg.mode != 'r':
# 		    se_neg.search(se_neg.norm_fam_query_size, 1, se_neg._max, se_neg.norm_hog_query_size)

# 		self.fs_neg = fs_neg
# 		self.se_neg = se_neg

# 	def get_clade_specific_negatives(self, stree_path, oma_h5_path):
# 		'''
# 		1.	precompute clade taxonomic levels
# 		2.	iterate over hog tab and store families with all taxa within clade of interest and with >6 members
# 		3.	iterate over entry tab and store candidate negatives inside fam2protoffsets (mem?)
# 		4.	reduce to fam2protoff by randomly selecting one protein per family
# 		5.	randomize fam2protoff values and select the appropriate number (enough?)
# 		6.	parse the sequences and ids of these proteins
# 		'''
# 		random.seed(123)
# 		np.random.seed(123)

# 		print(' - gather clade specific taxa to restrict OMA families')
# 		clade_taxa, clade_species = self.get_clade_specific_taxa(stree_path, self.neg_root_taxon)

# 		# load OMA h5 file and create pointers
# 		h5file = open_file(oma_h5_path, mode="r")
# 		hog_tab = h5file.root.HogLevel
# 		gen_tab = h5file.root.Genome[:]
# 		ent_tab = h5file.root.Protein.Entries
# 		seq_buffer = h5file.root.Protein.SequenceBuffer

# 		# load families in memory (not sure if necessary actually)
# 		families = hog_tab.col('Fam')

# 		print(' - gather all OMA families from the clade')
# 		clade_families = set()

# 		curr_fam = families[0]
# 		i = 0
# 		j = 0

# 		for fam in tqdm(families):
# 		    if fam != curr_fam:
		        
# 		        fam_taxa = hog_tab[i:j]['Level']
# 		        idx = np.searchsorted(clade_taxa, fam_taxa)
		        
# 		        # check if all fam_taxa are neg_root_taxon specific
# 		        if np.sum(clade_taxa[idx]==fam_taxa)==fam_taxa.size:
# 		            clade_families.add(fam)
		            
# 		        # move pointer and update current family
# 		        i = j
# 		        curr_fam = fam
# 		    j += 1

# 		fam_taxa = hog_tab[i:j]['Level']
# 		idx = np.searchsorted(clade_taxa, fam_taxa)
# 		if np.sum(clade_taxa[idx]==fam_taxa)==fam_taxa.size:
# 		    clade_families.add(fam)

# 		clade_families = np.array(sorted(clade_families))

# 		print(' - gather proteins of these families')
# 		fam2ent_offsets = collections.defaultdict(list)

# 		for r in tqdm(gen_tab):
# 		    sp = r['SciName']
		    
# 		    # load a slice if species in taxonomy
# 		    if sp in clade_species:
# 		        entry_off = r['EntryOff']
# 		        entry_num = r['TotEntries']
		        
# 		        # sub entry table for species
# 		        sp_ent_tab = ent_tab[entry_off: entry_off + entry_num]
		        
# 		        for ent in sp_ent_tab:
# 		            hog = ent['OmaHOG']
		            
# 		            if hog:
# 		                fam = int(ent['OmaHOG'].split(b'.')[0][4:])
		                
# 		                if fam in clade_families:
# 		                    fam2ent_offsets[fam].append(ent['EntryNr'] - 1)

# 		print(' - select randomly one protein per family')
# 		fam2ent_off = dict()

# 		for fam, ent_offsets in tqdm(fam2ent_offsets.items()):
		    
# 		    # same filter than positive queries
# 		    if len(ent_offsets) >= self.db.min_prot_nr and len(ent_offsets) <= self.db.max_prot_nr:
		        
# 		        fam2ent_off[fam] = random.choice(ent_offsets)

# 		print(' - select randomly one negative per positive query')
# 		neg_ent_offsets = np.random.permutation(list(fam2ent_off.values()))[:self.se_pos.fs._query_id.nrows]

# 		print(' - collect negative sequences and identifiers')
# 		seqs = []
# 		ids = []
# 		for ent_off in tqdm(neg_ent_offsets):
# 		    seq_off = ent_tab[ent_off]['SeqBufferOffset']
# 		    seqs.append(b''.join(seq_buffer[seq_off: seq_off + ent_tab[ent_off]['SeqBufferLength'] - 1]).decode('ascii'))
# 		    ids.append(ent_off + 1)

# 		return seqs, ids

# 	def get_random_negatives(self):

# 		def _simulate_query(ql, a_freq):
# 		    return "".join(np.random.choice(list(a_freq.keys()), ql, p=list(a_freq_scaled.values())))

# 		np.random.seed(123)

# 		# load in memory
# 		prot_tab = self.db._prot_tab[:]
# 		pos_qoffsets = self.fs._query_id[:].flatten()

# 		# amino acid frequencies from uniprot
# 		a_freq = {"A":0.0825,"C":0.0137,"D":0.0545,"E":0.0675,"F":0.0386,"G":0.0707,"H":0.0227,"I":0.0596,"K":0.0584,"L":0.0966,
# 		          "M":0.0242,"N":0.0406,"P":0.0470,"Q":0.0393,"R":0.0553,"S":0.0656,"T":0.0534,"V":0.0687,"W":0.0108,"Y":0.0292}

# 		sum_a_freq = sum(a_freq.values())

# 		# they do not sum to one. scale them to one
# 		a_freq_scaled = {k:v/sum_a_freq for k, v in a_freq.items()}

# 		seqs = [_simulate_query(prot_tab[q]['SeqLen'], a_freq_scaled) for q in self.fs._query_id[:].flatten()]

# 		return seqs, list(range(len(seqs)))


# class BLASTvalidationFamily(ValidationFamily):

# 	def __init__(self, se_pos, neg_root_taxon):
# 		super().__init__(se_pos, neg_root_taxon)

# 	def initiate_negative(self):

# 		assert os.path.exists(self.neg_query_file), 'Negative queries missing'

# 		qr_name = '_'.join(self.name.split('_')[len(self.ki.name.split('_')):])
# 		name = '{}_fs'.format(qr_name)

# 		fs_neg = BLASTsearch(self.ki, name=name, fasta=self.neg_query_file, qr_name=qr_name)
# 		if fs_neg.mode != 'r':
# 			fs_neg.export_query_fasta()

# 		self.fs_neg = fs_neg
# 		self.fs_neg.fs.close()

# 	def validate(self, thresholds):

# 		self.import_negative()

# 		self._validate(thresholds, True)

# 	def import_negative(self):

# 		qr_name = '_'.join(self.name.split('_')[len(self.ki.name.split('_')):])
# 		name = '{}_fs'.format(qr_name)

# 		fs_neg = DIAMONDsearch(self.ki, name=name, fasta=self.neg_query_file, qr_name=qr_name, mode='a')
# 		fs_neg.import_blast_result('evalue', True)

# 		se_neg = Search(fs_neg)
# 		self.se_neg = se_neg


# class SWvalidationFamily(ValidationFamily):
#     '''
#     only positives for now
#     '''
#     def __init__(self, se_pos):

#         assert se_pos.mode == 'r', 'Search must be opened in read mode.'

#         self.db = se_pos.db
#         self.ki = se_pos.ki
#         self.fs = se_pos.fs
#         self.se_pos = se_pos
        
#         self.name = "{}_Family".format(self.se_pos.name)
#         self.path = self.se_pos.path
#         self.file = "{}{}.h5".format(self.path, self.name)
        
#         if os.path.isfile(self.file):
#             self.mode = 'r'
#             self.va = tables.open_file(self.file, self.mode)
#         else:
#             self.mode = 'w'
#             self.va = tables.open_file(self.file, self.mode)
            
#     def validate(self, thresholds, prob):

#         assert (self.mode in {'w', 'a'}), 'ValidationFamily must be opened in write mode.'

#         # store thresholds
#         self.va.create_carray('/', 'Threshold', obj=np.array(thresholds, dtype=np.float64), filters=self.db._compr)

#         ### positive validation
#         tp_query2tresh, fn_query2tresh, fp_pos_query2tresh = self._validate_positive(self.se_pos._res_tab[:], self._thresholds[:], self.db._prot_tab[:], self.ki.idx_type, prob)

#         # store results
#         self.va.create_carray('/', 'TP', obj=tp_query2tresh, filters=self.db._compr)
#         self.va.create_carray('/', 'FN', obj=fn_query2tresh, filters=self.db._compr)
#         self.va.create_carray('/', 'FP_pos', obj=fp_pos_query2tresh, filters=self.db._compr)

#         # close and re-open in read mode
#         self.va.close()
#         self.mode = 'r'
#         self.va = tables.open_file(self.file, self.mode)



