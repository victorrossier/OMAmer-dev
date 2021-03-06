import os
import sys
import numba
import scipy
import tables
import numpy as np
from property_manager import lazy_property, cached_property

from .hierarchy import _children_hog

class CumSearch():

    class ResultFormat(tables.IsDescription):
        QueryOff = tables.UInt64Col(pos=1)
        FamOff = tables.UInt64Col(pos=2)
        Score = tables.Float64Col(pos=3)

    def __init__(self, fs, name=None, path=None, nthreads=None):
        
        assert fs.mode == 'r', 'FlatSearch must be opened in read mode.'
        assert fs.db.mode == 'r', 'Database must be opened in read mode.'
        assert fs.ki.mode == 'r', 'Index must be opened in read mode.'
        
        # load fs, ki and db
        self.db = fs.db
        self.ki = fs.ki
        self.fs = fs
        
        # performance features
        self.nthreads = nthreads if nthreads is not None else os.cpu_count()

        # hdf5 file
        self.name = "{}_{}".format(self.fs.name, name if name else 'kmerfreq_prob_max')
        self.path = path if path else self.ki.path
        self.file = "{}{}.h5".format(self.path, self.name)
        
        if os.path.isfile(self.file):
            self.mode = 'r'
            self.se = tables.open_file(self.file, self.mode)

        else:
            self.mode = 'w'
            self.se = tables.open_file(self.file, self.mode)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.se.close()
        
    def clean(self):
        '''
        close and remove hdf5 file
        '''
        self.__exit__()
        try:
            os.remove(self.file)
        except FileNotFoundError:
            print("{} already cleaned".format(self.file))
    
    def reset_cache(self):
        '''
            Reset caches.
        '''
        # del self.hog_cum_count
        del self.nr_kmer

    @cached_property
    def nr_kmer(self):
        return np.unique(self.ki._table_idx[:]).size -1

    ### easy access to data via properties ###
    @property
    def _res_tab(self):

        if '/Result' in self.se:
            return self.se.root.Result
        else:
            return self.se.create_table('/', 'Result', self.ResultFormat, filters=self.db._compr)

    @property
    def _hog_score(self):
        if '/HOGscore' in self.se:
            return self.se.root.HOGscore
        else:
            return self.se.create_earray('/', 'HOGscore', tables.Float64Atom(self.db._hog_tab.nrows), (0,), filters=self.db._compr)

    @property
    def _bestpath_mask(self):
        if '/BestPathMask' in self.se:
            return self.se.root.BestPathMask
        else:
            return self.se.create_earray('/', 'BestPathMask', tables.BoolAtom(self.db._hog_tab.nrows), (0,), filters=self.db._compr)

    def search(self, top_n=10):
        '''
        args
         - top_n: top n families on which compute the score and if cum_mode='max', also on which to cumulate counts.
         - cum_mode: how count are cumulated inside families. They can be summed or maxed at each node.
         - score: how counts are normalized: naive, querysize, theo_prob, kmerfreq_prob, etc.
        '''
        assert top_n <= self.db._fam_tab.nrows, 'top n is smaller than number of families'

        print('compute family scores')
        # filter top n families
        fam_ranked_n = self.fs._fam_ranked[:][:,:top_n]

        # cumulated queryHOG counts for the top n families
        queryHog_cum_counts = self.cumulate_counts_nfams_nqueries(
            fam_ranked_n, self.fs._queryHog_count[:], self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), 
            self.fs._query_count.nrows, self.cumulate_counts_1fam, self._sum, self._max)

        # update queryFam counts to have maxed counts instead of summed counts (using root-HOG maxed counts)
        fam_rh_offsets = self.db._fam_tab.col('HOGoff')[fam_ranked_n]
        queryFam_cum_counts = queryHog_cum_counts[np.arange(fam_rh_offsets.shape[0])[:,None], fam_rh_offsets]

        # cumulate HOG counts
        hog_cum_counts = self.cumulate_counts_nfams(
            self.ki._hog_count[:], self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), self.cumulate_counts_1fam, self._sum, self._max)            

        # and get family cumulated counts
        fam_cum_counts = hog_cum_counts[self.db._fam_tab.col('HOGoff')]

        # normalize family cum counts
        queryFam_scores = self.compute_family_kmerfreq_probs(
            fam_ranked_n, self.fs._query_count[:], self.fs._query_occur[:], queryFam_cum_counts, fam_cum_counts, self.nr_kmer)

        # ranked families
        fam_reranked_1, queryFam_scores = self.reranked_families_and_scores(queryFam_scores, fam_ranked_n, True)

        print('compute subfamily scores')
        ### Subfamilies
        queryHog_scores, queryHog_bestpaths = self.compute_subfamily_kmerfreq_probs(
            fam_reranked_1, self.fs._query_count[:], self.fs._query_occur[:], self.db._fam_tab[:], queryFam_scores, self.db._hog_tab[:], 
            self.db._chog_arr[:], queryHog_cum_counts, hog_cum_counts, self.fs._queryHog_count[:], self.fs._queryHog_occur[:], self.nr_kmer)

        print('store results')
        # store results for hogs
        self._hog_score.append(queryHog_scores)
        self._hog_score.flush()
        self._bestpath_mask.append(queryHog_bestpaths)
        self._bestpath_mask.flush()

        # store results for families
        self._res_tab.append(list(zip(*[self.fs._query_id[:].flatten(), fam_reranked_1[:,0], queryFam_scores[:, 0]])))
        self._res_tab.flush()

        # reset cache
        self.reset_cache()

        # close and re-open in read mode
        self.se.close()
        self.mode = 'r'
        self.se = tables.open_file(self.file, self.mode)

    def compute_family_kmerfreq_probs(self, fam_ranked_n, query_counts, query_occurs, queryFam_cum_count, fam_cum_counts, nr_kmer):

        queryFam_score = np.zeros(fam_ranked_n.shape, np.float64)

        for q in range(fam_ranked_n.shape[0]):

            # number of unique k-mer in the query
            query_size = query_counts[q]

            # sum of k-mer family occurence of the query
            query_occur = query_occurs[q]
            
            for i in range(fam_ranked_n.shape[1]):
                
                # family == root-HOG
                qf_ccount = queryFam_cum_count[q, i]
                f_ccount = fam_cum_counts[fam_ranked_n[q, i]]
                
                queryFam_score[q, i] = self.compute_prob_score(
                    query_size, qf_ccount, self._bernoulli_true_kmerfreq, 
                    query_occur=query_occur, query_count=query_size, nr_kmer=nr_kmer, hog_count=f_ccount)
        
        return queryFam_score

    def compute_prob_score(self, query_size, qh_ccount, bernoulli_fun, **kwargs):

        # compute the binomial bernoulli probability (P(draw a k-mer in the HOG))
        bernoulli = bernoulli_fun(**kwargs)

        # probabilities for tail x values
        tail_size = query_size - qh_ccount 
        tail_log_probs = self.poisson_log_pmf(np.arange(qh_ccount, query_size + 1), np.full(tail_size + 1, bernoulli * query_size))

        # sum of these tail probabilities
        return scipy.special.logsumexp(tail_log_probs)
    
    @staticmethod
    def poisson_log_pmf(k, lda):
        return k*scipy.log(lda) - lda - scipy.special.gammaln(k + 1)

    @staticmethod
    def reranked_families_and_scores(queryFam_scores, fam_ranked_n, prob=True):
        if prob:
            idx = queryFam_scores.argsort()
        else:
            idx = (-queryFam_scores).argsort()
        return fam_ranked_n[np.arange(fam_ranked_n.shape[0])[:,None], idx][:,:1], queryFam_scores[np.arange(fam_ranked_n.shape[0])[:,None], idx][:,:1]

    @staticmethod
    def _bernoulli_true_kmerfreq(query_occur, query_count, nr_kmer, hog_count):
        '''
        true k-mer frequencies are used for probability to get a k-mer at one location
        Bernoulli is the joint probability of not having the k-mer in the family
        '''
        kmer_prob = query_occur / query_count / nr_kmer

        return 1 - (1 - kmer_prob) ** hog_count

    ### subfamily ######################################################################################################################
    def compute_subfamily_kmerfreq_probs(
        self, fam_reranked_1, query_counts, query_occurs, fam_tab, queryFam_scores, hog_tab, chog_buff, queryHog_cum_counts, hog_cum_counts, queryHog_counts, queryHog_occurs, nr_kmer):
        
        def _top_down_best_path(
            self, hog_off, hog_tab, chog_buff, queryHog_cum_counts, query_off, hog_cum_counts, query_occur, query_size, nr_kmer, queryHog_counts, queryHog_occurs,
            queryHog_score, queryHog_bestpath):
        
            if hog_off and hog_tab[hog_off]['ChildrenHOGoff'] != -1:

                # get HOG children
                children = _children_hog(hog_off, hog_tab, chog_buff)

                # intiate scores; either raw cumulative counts or probabilities
                child_scores = np.zeros(children.shape)

                # compute scores
                for i, h in enumerate(children):

                    # query-HOG cumulative k-mer counts
                    qh_ccount = queryHog_cum_counts[query_off, h]

                    if qh_ccount > 0:
                        h_ccount = hog_cum_counts[h]
                        
                        child_scores[i] = self.compute_prob_score(
                            query_size, qh_ccount, self._bernoulli_true_kmerfreq, 
                            query_occur=query_occur, query_count=query_size, nr_kmer=nr_kmer, hog_count=h_ccount)

                    elif qh_ccount == 0:
                        child_scores[i] = 0.0  # because log(1) = 0

                # find best child
                cand_offsets = np.where((child_scores < 0) & (child_scores==np.min(child_scores)))[0]

                # deals with ties
                if cand_offsets.size == 1:
                    best_child = children[cand_offsets][0] 

                    # store best path and score --> latter use buff for that
                    queryHog_bestpath[query_off, best_child] = True
                    queryHog_score[query_off, best_child] = child_scores[cand_offsets][0]

                else:
                    best_child = None       

                # remove queryHOG count from query size (not the cumulated one!)
                query_size = query_size - queryHog_counts[query_off, best_child]

                # remove queryHOG k-mer occurences (sum of k-mer occurences for the intersecting k-mers) from query occur
                query_occur = query_occur - queryHog_occurs[query_off, best_child]
                
                _top_down_best_path(
                    self, best_child, hog_tab, chog_buff, queryHog_cum_counts, query_off, hog_cum_counts, query_occur, query_size, nr_kmer, queryHog_counts, queryHog_occurs, 
                    queryHog_score, queryHog_bestpath)
        
        queryHog_scores = np.zeros(queryHog_cum_counts.shape, dtype=np.float64)
        queryHog_bestpaths = np.full(queryHog_cum_counts.shape, False)
        
        for q in range(fam_reranked_1.shape[0]):

            # number of unique k-mer in the query
            query_size = query_counts[q]

            # sum of k-mer family occurence of the query
            query_occur = query_occurs[q]

            rh = fam_tab[fam_reranked_1[q, 0]]['HOGoff']

            # add root-HOG score and best path
            queryHog_scores[q, rh] = queryFam_scores[q, 0]
            queryHog_bestpaths[q, rh] = True

            _top_down_best_path(
                self, rh, hog_tab, chog_buff, queryHog_cum_counts, q, hog_cum_counts, query_occur, query_size, nr_kmer, queryHog_counts, queryHog_occurs, 
                queryHog_scores, queryHog_bestpaths)
            
        return queryHog_scores, queryHog_bestpaths

    ### cumulate counts #####################################################################################################################
    @staticmethod
    @numba.njit
    def _max(x, y):
        return max(x, y)

    @staticmethod
    @numba.njit
    def _sum(x, y):
            return x + y 

    @staticmethod
    @numba.njit
    def cumulate_counts_1fam(hog_cum_counts, fam_level_offsets, hog2parent, cum_fun, prop_fun):
        
        current_best_child_count = np.zeros(hog_cum_counts.shape, dtype=np.uint64)
        
        # iterate over level offsets backward
        for i in range(fam_level_offsets.size - 2):
            x = fam_level_offsets[-i - 3: -i - 1]

            # when reaching level, sum all hog counts with their best child count
            hog_cum_counts[x[0]:x[1]] = cum_fun(hog_cum_counts[x[0]:x[1]], current_best_child_count[x[0]:x[1]])   

            # update current_best_child_count of the parents of the current hogs
            for i in range(x[0], x[1]):
                parent_off = hog2parent[i]
                
                # only if parent exists
                if parent_off != -1:
                    c = current_best_child_count[hog2parent[i]]
                    current_best_child_count[hog2parent[i]] = prop_fun(c, hog_cum_counts[i])
    
    @staticmethod
    @numba.njit(parallel=True, nogil=True)
    def cumulate_counts_nfams(hog_counts, fam_tab, level_arr, hog2parent, main_fun, cum_fun, prop_fun):

        hog_cum_counts = hog_counts.copy()

        for fam_off in numba.prange(fam_tab.size):
            entry = fam_tab[fam_off]
            level_off = entry['LevelOff']
            level_num = entry['LevelNum']
            fam_level_offsets = level_arr[level_off:np.int64(level_off + level_num + 2)]

            main_fun(hog_cum_counts, fam_level_offsets, hog2parent, cum_fun, prop_fun)

        return hog_cum_counts

    @staticmethod
    @numba.njit(parallel=True, nogil=True)
    def cumulate_counts_nfams_nqueries(fam_results, hog_counts, fam_tab, level_arr, hog2parent, q_num, main_fun, cum_fun, prop_fun):
        
        hog_cum_counts = hog_counts.copy()
        
        # iterate queries
        for q in numba.prange(q_num):
            
            # iterate families
            for fam_off in fam_results[q]:
                
                entry = fam_tab[fam_off]
                level_off = entry['LevelOff']
                level_num = entry['LevelNum']
                fam_level_offsets = level_arr[level_off:np.int64(level_off + level_num + 2)]
                main_fun(hog_cum_counts[q], fam_level_offsets, hog2parent, cum_fun, prop_fun)
            
        return hog_cum_counts