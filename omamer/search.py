import os
import sys
import numba
import scipy
import tables
import numpy as np
from property_manager import lazy_property, cached_property

from .hierarchy import _children_hog

class Search():

    class ResultFormat(tables.IsDescription):
        QueryOff = tables.UInt64Col(pos=1)
        FamOff = tables.UInt64Col(pos=2)
        Score = tables.Float64Col(pos=3)

    def __init__(self, fs, name=None, path=None, score='querysize', cum_mode='max', nthreads=None):
        
        assert fs.mode == 'r', 'FlatSearch must be opened in read mode.'
        assert fs.db.mode == 'r', 'Database must be opened in read mode.'
        assert fs.ki.mode == 'r', 'Index must be opened in read mode.'
        
        # load fs, ki and db
        self.db = fs.db
        self.ki = fs.ki
        self.fs = fs

        # score features
        self.score = score
        self.cum_mode = cum_mode
        
        # performance features
        self.nthreads = nthreads if nthreads is not None else os.cpu_count()

        # hdf5 file
        self.name = "{}_{}".format(self.fs.name, name if name else '{}_{}'.format(score, cum_mode))
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

    # @cached_property
    # def hog_cum_count(self):
    #     return self.cumulate_counts_nfams(self.ki._hog_count[:], self.db._fam_tab[:], self.db._level_arr[:], 
    #         self.db._hog_tab.col('ParentOff'), self.cumulate_counts_1fam, self._sum, self._max)

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

    def search(self, top_n=10, chunksize=500):
        '''
        args
         - top_n: top n families on which compute the score and if cum_mode='max', also on which to cumulate counts.
         - cum_mode: how count are cumulated inside families. They can be summed or maxed at each node.
         - score: how counts are normalized: naive, querysize, theo_prob, kmerfreq_prob, etc.
        '''
        assert top_n <= self.db._fam_tab.nrows, 'top n is smaller than number of families'

        # reference HOG cum counts
        if self.cum_mode == 'max':
            # cumulate HOG counts
            hog_cum_counts = self.cumulate_counts_nfams(
                self.ki._hog_count[:], self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), self.cumulate_counts_1fam, self._sum, self._max)            
        
        elif self.cum_mode == 'sum':
            # cumulate HOG counts
            hog_cum_counts = self.cumulate_counts_nfams(
                self.ki._hog_count[:], self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), self.cumulate_counts_1fam, self._sum, self._sum)
        
        # get family cumulated counts
        fam_cum_counts = hog_cum_counts[self.db._fam_tab.col('HOGoff')]
        
        # process by chunk to save memory
        i = 0
        while (i + chunksize) <= self.fs._fam_ranked.nrows:

            print('compute family scores')
            # filter top n families
            fam_ranked_n = self.fs._fam_ranked[i:i + chunksize][:,:top_n]

            # cumulated queryHOG counts for the top n families
            if self.cum_mode == 'max':
                queryHog_cum_counts = self.cumulate_counts_nfams_nqueries(
                    fam_ranked_n, self.fs._queryHog_count[i:i + chunksize], self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), 
                    chunksize, self.cumulate_counts_1fam, self._sum, self._max)

                # update queryFam counts to have maxed counts instead of summed counts (using root-HOG maxed counts)
                fam_rh_offsets = self.db._fam_tab.col('HOGoff')[fam_ranked_n]
                queryFam_cum_counts = queryHog_cum_counts[np.arange(fam_rh_offsets.shape[0])[:,None], fam_rh_offsets]

            # if sum, already cumulated during flat search
            elif self.cum_mode == 'sum':
                queryFam_cum_counts = self.fs._queryFam_count[i:i + chunksize][np.arange(fam_ranked_n.shape[0])[:,None], fam_ranked_n]

            # normalize family cum counts
            if self.score == 'naive':
                queryFam_scores = queryFam_cum_counts
                fam_reranked_1, queryFam_scores = self.reranked_families_and_scores(queryFam_scores, fam_ranked_n, False)

            elif self.score in {'querysize', 'correct_querysize'}:
                queryFam_scores = queryFam_cum_counts / self.fs._query_count[i:i + chunksize]
                fam_reranked_1, queryFam_scores = self.reranked_families_and_scores(queryFam_scores, fam_ranked_n, False)

            elif self.score == 'theo_prob':
                queryFam_scores = self.compute_family_theo_probs(
                    fam_ranked_n, self.fs._query_count[i:i + chunksize], queryFam_cum_counts, fam_cum_counts, self.ki.k, self.ki.alphabet.n)
                fam_reranked_1, queryFam_scores = self.reranked_families_and_scores(queryFam_scores, fam_ranked_n, True)

            elif self.score == 'kmerfreq_prob':
                queryFam_scores = self.compute_family_kmerfreq_probs(
                    fam_ranked_n, self.fs._query_count[i:i + chunksize], self.fs._query_occur[i:i + chunksize], queryFam_cum_counts, fam_cum_counts, self.nr_kmer)
                fam_reranked_1, queryFam_scores = self.reranked_families_and_scores(queryFam_scores, fam_ranked_n, True)

            elif self.score == 'kmerfreq_probdiv6':
                queryFam_scores = self.compute_family_kmerfreq_probs(
                    fam_ranked_n, self.fs._query_count[i:i + chunksize], self.fs._query_occur[i:i + chunksize], queryFam_cum_counts, fam_cum_counts, self.nr_kmer, True)
                fam_reranked_1, queryFam_scores = self.reranked_families_and_scores(queryFam_scores, fam_ranked_n, True)

            print('compute subfamily scores')
            ### Subfamilies
            # compute queryHog_cum_counts if cum_mode == sum
            if self.cum_mode == 'sum':
                queryHog_cum_counts = self.cumulate_counts_nfams_nqueries(
                    fam_reranked_1, self.fs._queryHog_count[i:i + chunksize], self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), 
                    self.fs._query_count.nrows, self.cumulate_counts_1fam, self._sum, self._sum)

            if self.score == 'naive':
                queryHog_scores, queryHog_bestpaths = self.norm_hog_naive(fam_reranked_1, queryHog_cum_counts)

            elif self.score == 'querysize':
                queryHog_scores, queryHog_bestpaths = self.norm_hog_query_size(fam_reranked_1, queryHog_cum_counts)

            # not sure the other are update for chunksize
            elif self.score == 'correct_querysize':
                queryHog_scores, queryHog_bestpaths = self.norm_hog_correct_query_size(
                    fam_reranked_1, queryHog_cum_counts, self.fs._query_count[i:i + chunksize], self.fs._queryHog_count[i:i + chunksize])

            elif self.score == 'theo_prob':
                queryHog_scores, queryHog_bestpaths = self.compute_subfamily_theo_probs(
                    fam_reranked_1, self.fs._query_count[i:i + chunksize], self.db._fam_tab[:], queryFam_scores, self.db._hog_tab[:], self.db._chog_arr[:], 
                    queryHog_cum_counts, hog_cum_counts, self.ki.k, self.ki.alphabet.n, self.fs._queryHog_count[i:i + chunksize])

            elif self.score == 'kmerfreq_prob':
                queryHog_scores, queryHog_bestpaths = self.compute_subfamily_kmerfreq_probs(
                    fam_reranked_1, self.fs._query_count[i:i + chunksize], self.fs._query_occur[i:i + chunksize], self.db._fam_tab[:], queryFam_scores, self.db._hog_tab[:], 
                    self.db._chog_arr[:], queryHog_cum_counts, hog_cum_counts, self.fs._queryHog_count[i:i + chunksize], self.fs._queryHog_occur[i:i + chunksize], self.nr_kmer)

            elif self.score == 'kmerfreq_probdiv6':
                queryHog_scores, queryHog_bestpaths = self.compute_subfamily_kmerfreq_probs(
                    fam_reranked_1, self.fs._query_count[i:i + chunksize], self.fs._query_occur[i:i + chunksize], self.db._fam_tab[:], queryFam_scores, self.db._hog_tab[:], 
                    self.db._chog_arr[:], queryHog_cum_counts, hog_cum_counts, self.fs._queryHog_count[i:i + chunksize], self.fs._queryHog_occur[i:i + chunksize], self.nr_kmer, True)
          
            print('store results')
            # store results for hogs
            self._hog_score.append(queryHog_scores)
            self._hog_score.flush()
            self._bestpath_mask.append(queryHog_bestpaths)
            self._bestpath_mask.flush()

            # store results for families
            self._res_tab.append(list(zip(*[self.fs._query_id[i:i + chunksize].flatten(), fam_reranked_1[:,0], queryFam_scores[:, 0]])))
            self._res_tab.flush()

            i += chunksize

        ### ---> end
        chunksize = self.fs._fam_ranked.nrows - i

        print('compute family scores')
        # filter top n families
        fam_ranked_n = self.fs._fam_ranked[i:i + chunksize][:,:top_n]

        # cumulated queryHOG counts for the top n families
        if self.cum_mode == 'max':
            queryHog_cum_counts = self.cumulate_counts_nfams_nqueries(
                fam_ranked_n, self.fs._queryHog_count[i:i + chunksize], self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), 
                chunksize, self.cumulate_counts_1fam, self._sum, self._max)

            # update queryFam counts to have maxed counts instead of summed counts (using root-HOG maxed counts)
            fam_rh_offsets = self.db._fam_tab.col('HOGoff')[fam_ranked_n]
            queryFam_cum_counts = queryHog_cum_counts[np.arange(fam_rh_offsets.shape[0])[:,None], fam_rh_offsets]

        # if sum, already cumulated during flat search
        elif self.cum_mode == 'sum':
            queryFam_cum_counts = self.fs._queryFam_count[i:i + chunksize][np.arange(fam_ranked_n.shape[0])[:,None], fam_ranked_n]

        # normalize family cum counts
        if self.score == 'naive':
            queryFam_scores = queryFam_cum_counts
            fam_reranked_1, queryFam_scores = self.reranked_families_and_scores(queryFam_scores, fam_ranked_n, False)

        elif self.score in {'querysize', 'correct_querysize'}:
            queryFam_scores = queryFam_cum_counts / self.fs._query_count[i:i + chunksize]
            fam_reranked_1, queryFam_scores = self.reranked_families_and_scores(queryFam_scores, fam_ranked_n, False)

        elif self.score == 'theo_prob':
            queryFam_scores = self.compute_family_theo_probs(
                fam_ranked_n, self.fs._query_count[i:i + chunksize], queryFam_cum_counts, fam_cum_counts, self.ki.k, self.ki.alphabet.n)
            fam_reranked_1, queryFam_scores = self.reranked_families_and_scores(queryFam_scores, fam_ranked_n, True)

        elif self.score == 'kmerfreq_prob':
            queryFam_scores = self.compute_family_kmerfreq_probs(
                fam_ranked_n, self.fs._query_count[i:i + chunksize], self.fs._query_occur[i:i + chunksize], queryFam_cum_counts, fam_cum_counts, self.nr_kmer)
            fam_reranked_1, queryFam_scores = self.reranked_families_and_scores(queryFam_scores, fam_ranked_n, True)

        elif self.score == 'kmerfreq_probdiv6':
            queryFam_scores = self.compute_family_kmerfreq_probs(
                fam_ranked_n, self.fs._query_count[i:i + chunksize], self.fs._query_occur[i:i + chunksize], queryFam_cum_counts, fam_cum_counts, self.nr_kmer, True)
            fam_reranked_1, queryFam_scores = self.reranked_families_and_scores(queryFam_scores, fam_ranked_n, True)

        print('compute subfamily scores')
        ### Subfamilies
        # compute queryHog_cum_counts if cum_mode == sum
        if self.cum_mode == 'sum':
            queryHog_cum_counts = self.cumulate_counts_nfams_nqueries(
                fam_reranked_1, self.fs._queryHog_count[i:i + chunksize], self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), 
                self.fs._query_count.nrows, self.cumulate_counts_1fam, self._sum, self._sum)

        if self.score == 'naive':
            queryHog_scores, queryHog_bestpaths = self.norm_hog_naive(fam_reranked_1, queryHog_cum_counts)

        elif self.score == 'querysize':
            queryHog_scores, queryHog_bestpaths = self.norm_hog_query_size(fam_reranked_1, queryHog_cum_counts)

        # not sure the other are update for chunksize
        elif self.score == 'correct_querysize':
            queryHog_scores, queryHog_bestpaths = self.norm_hog_correct_query_size(
                fam_reranked_1, queryHog_cum_counts, self.fs._query_count[i:i + chunksize], self.fs._queryHog_count[i:i + chunksize])

        elif self.score == 'theo_prob':
            queryHog_scores, queryHog_bestpaths = self.compute_subfamily_theo_probs(
                fam_reranked_1, self.fs._query_count[i:i + chunksize], self.db._fam_tab[:], queryFam_scores, self.db._hog_tab[:], self.db._chog_arr[:], 
                queryHog_cum_counts, hog_cum_counts, self.ki.k, self.ki.alphabet.n, self.fs._queryHog_count[i:i + chunksize])

        elif self.score == 'kmerfreq_prob':
            queryHog_scores, queryHog_bestpaths = self.compute_subfamily_kmerfreq_probs(
                fam_reranked_1, self.fs._query_count[i:i + chunksize], self.fs._query_occur[i:i + chunksize], self.db._fam_tab[:], queryFam_scores, self.db._hog_tab[:], 
                self.db._chog_arr[:], queryHog_cum_counts, hog_cum_counts, self.fs._queryHog_count[i:i + chunksize], self.fs._queryHog_occur[i:i + chunksize], self.nr_kmer)

        elif self.score == 'kmerfreq_probdiv6':
            queryHog_scores, queryHog_bestpaths = self.compute_subfamily_kmerfreq_probs(
                fam_reranked_1, self.fs._query_count[i:i + chunksize], self.fs._query_occur[i:i + chunksize], self.db._fam_tab[:], queryFam_scores, self.db._hog_tab[:], 
                self.db._chog_arr[:], queryHog_cum_counts, hog_cum_counts, self.fs._queryHog_count[i:i + chunksize], self.fs._queryHog_occur[i:i + chunksize], self.nr_kmer, True)
      
        print('store results')
        # store results for hogs
        self._hog_score.append(queryHog_scores)
        self._hog_score.flush()
        self._bestpath_mask.append(queryHog_bestpaths)
        self._bestpath_mask.flush()

        # store results for families
        self._res_tab.append(list(zip(*[self.fs._query_id[i:i + chunksize].flatten(), fam_reranked_1[:,0], queryFam_scores[:, 0]])))
        self._res_tab.flush()

        # reset cache
        # self.reset_cache()

        # close and re-open in read mode
        self.se.close()
        self.mode = 'r'
        self.se = tables.open_file(self.file, self.mode)

    def compute_family_theo_probs(self, fam_ranked_n, query_counts, queryFam_cum_count, fam_cum_counts, k, aa_n):
        
        queryFam_score = np.zeros(fam_ranked_n.shape, np.float64)
        
        for q in range(fam_ranked_n.shape[0]):
            query_size = query_counts[q]
            
            for i in range(fam_ranked_n.shape[1]):
                
                # family == root-HOG
                qf_ccount = queryFam_cum_count[q, i]
                f_ccount = fam_cum_counts[fam_ranked_n[q, i]]
                
                queryFam_score[q, i] = self.compute_prob_score(
                    query_size, qf_ccount, self._bernoulli_theo_kmerprob, k=k, 
                    digit_num=aa_n, hog_count=f_ccount)
        
        return queryFam_score

    def compute_family_kmerfreq_probs(self, fam_ranked_n, query_counts, query_occurs, queryFam_cum_count, fam_cum_counts, nr_kmer, div6=False):

        queryFam_score = np.zeros(fam_ranked_n.shape, np.float64)

        if div6:
            fam_cum_counts = np.array(np.ceil(fam_cum_counts/6), dtype=np.uint64)

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

    # ### functions to compute p-values
    # @staticmethod
    # def _bernoulli_fam_kmerfreq(query_occur, query_count, fam_num):
    #     '''
    #     frequencies of families with k-mer
    #     can be used as bernoulli assuming all families have equal size
    #     '''
    #     # return np.divide(query_occur, query_count, out=np.zeros_like(query_occur), where=query_count!=0) / fam_num

    #     return query_occur / query_count / fam_num
    
    @staticmethod
    def _bernoulli_theo_kmerprob(k, digit_num, hog_count):
        '''
        prob of random k-mer is used for probability to get a k-mer at one location
        Bernoulli is the joint probability of not having the k-mer in the family
        '''
        kmer_prob = (1 / digit_num) ** k

        return 1 - (1 - kmer_prob) ** hog_count

    @staticmethod
    def _bernoulli_true_kmerfreq(query_occur, query_count, nr_kmer, hog_count):
        '''
        true k-mer frequencies are used for probability to get a k-mer at one location
        Bernoulli is the joint probability of not having the k-mer in the family
        '''
        kmer_prob = query_occur / query_count / nr_kmer

        return 1 - (1 - kmer_prob) ** hog_count

    ### subfamily ######################################################################################################################
    def compute_subfamily_theo_probs(
        self, fam_reranked_1, query_counts, fam_tab, queryFam_scores, hog_tab, chog_buff, queryHog_cum_counts, hog_cum_counts, k, aa_n, queryHog_counts):
        
        def _top_down_best_path(
            self, hog_off, hog_tab, chog_buff, queryHog_cum_counts, query_off, hog_cum_counts, query_size, k, aa_n, queryHog_counts, 
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
                            query_size, qh_ccount, self._bernoulli_theo_kmerprob, 
                            k=k, digit_num=aa_n, hog_count=h_ccount)

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
                
                _top_down_best_path(
                    self, best_child, hog_tab, chog_buff, queryHog_cum_counts, query_off, hog_cum_counts, query_size, k, aa_n, 
                    queryHog_counts, queryHog_score, queryHog_bestpath)
        
        queryHog_scores = np.zeros(queryHog_cum_counts.shape, dtype=np.float64)
        queryHog_bestpaths = np.full(queryHog_cum_counts.shape, False)
        
        for q in range(fam_reranked_1.shape[0]):
            query_size = query_counts[q]
            rh = fam_tab[fam_reranked_1[q, 0]]['HOGoff']

            # add root-HOG score and best path
            queryHog_scores[q, rh] = queryFam_scores[q, 0]
            queryHog_bestpaths[q, rh] = True

            _top_down_best_path(
                self, rh, hog_tab, chog_buff, queryHog_cum_counts, q, hog_cum_counts, query_size, k, aa_n, 
                queryHog_counts, queryHog_scores, queryHog_bestpaths)
            
        return queryHog_scores, queryHog_bestpaths

    def compute_subfamily_kmerfreq_probs(
        self, fam_reranked_1, query_counts, query_occurs, fam_tab, queryFam_scores, hog_tab, chog_buff, queryHog_cum_counts, hog_cum_counts, queryHog_counts, queryHog_occurs, nr_kmer, div6):
        
        def _top_down_best_path(
            self, hog_off, hog_tab, chog_buff, queryHog_cum_counts, query_off, hog_cum_counts, query_occur, query_size, nr_kmer, queryHog_counts, queryHog_occurs,
            queryHog_score, queryHog_bestpath):
        
            if hog_off and hog_tab[hog_off]['ChildrenHOGoff'] != -1:

                # UPDATE query size and occurs here!!! because we are already in a sub-HOG!!!
                # remove queryHOG count from query size (not the cumulated one!)
                query_size = query_size - queryHog_counts[query_off, hog_off]

                # remove queryHOG k-mer occurences (sum of k-mer occurences for the intersecting k-mers) from query occur
                query_occur = query_occur - queryHog_occurs[query_off, hog_off]

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
                
                _top_down_best_path(
                    self, best_child, hog_tab, chog_buff, queryHog_cum_counts, query_off, hog_cum_counts, query_occur, query_size, nr_kmer, queryHog_counts, queryHog_occurs, 
                    queryHog_score, queryHog_bestpath)
        
        queryHog_scores = np.zeros(queryHog_cum_counts.shape, dtype=np.float64)
        queryHog_bestpaths = np.full(queryHog_cum_counts.shape, False)

        if div6:
            hog_cum_counts = np.array(np.ceil(hog_cum_counts/6), dtype=np.uint64)
        
        for q in range(fam_reranked_1.shape[0]):

            # number of unique k-mer in the query  # !! I should already subtract there (or directly in function)
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

    def norm_hog_naive(self, fam_ranked, queryHog_cum_count):
        '''
        no normalization there
        '''
        queryHog_bestpath = self.compute_bestpath_nqueries_nfams(self.fs._query_count.nrows, fam_ranked, queryHog_cum_count, 
            self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'))
        return queryHog_cum_count, queryHog_bestpath

    def norm_hog_query_size(self, fam_ranked, queryHog_cum_count):
        '''
        same but normalize first the queryHog_cum_count
        could be slightly more clever and remove part conserved in parent 
        '''
        # query_count_bc = np.full((queryHog_cum_count.shape[1], queryHog_cum_count.shape[0]), self.fs._query_count[:]).T
        
        queryHog_score = queryHog_cum_count / self.fs._query_count[:]

        queryHog_bestpath = self.compute_bestpath_nqueries_nfams(self.fs._query_count.nrows, fam_ranked, queryHog_score, 
            self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'))
        
        return queryHog_score, queryHog_bestpath

    def norm_hog_correct_query_size(self, fam_ranked, queryHog_cum_count, query_count, queryHog_count):
        return self._norm_hog_correct_query_size(queryHog_cum_count, fam_ranked, query_count, 
            self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), queryHog_count)

    def _norm_hog_correct_query_size(self, queryHog_cum_count, fam_ranked, query_count, fam_tab, level_arr, hog2parent, queryHog_count):

        queryHog_score = np.zeros(queryHog_cum_count.shape, dtype=np.float64)
        queryHog_bestpath = np.full(queryHog_cum_count.shape, False)
        queryHog_revcum_count = np.full((queryHog_cum_count.shape[1], query_count.size), query_count.flatten(), dtype=np.uint16).T

        for q in range(queryHog_cum_count.shape[0]):
            for f in fam_ranked[q]:

                entry = fam_tab[f]
                level_off = entry['LevelOff']
                level_num = entry['LevelNum']
                fam_level_offsets = level_arr[level_off:np.int64(level_off + level_num + 2)]
                
                # compute and store root-HOG score
                rh = fam_level_offsets[0]
                queryHog_score[q][rh] = np.array(queryHog_cum_count[q][rh], dtype=np.float64) / query_count[q]
                
                # set the root-HOG as best path
                queryHog_bestpath[q][rh] = True
                
                for i in range(1, fam_level_offsets.size - 2):
                    x = fam_level_offsets[i: i + 2]
                    hog_offsets = np.arange(x[0], x[1])
                    
                    # grab parents
                    parent_offsets = hog2parent[hog_offsets]
                    
                    # update query revcumcount, basically substracting queryParent counts from query count
                    qh_count = queryHog_revcum_count[q][parent_offsets] - queryHog_count[q][parent_offsets]
                    queryHog_revcum_count[q][hog_offsets] = qh_count
                    
                    # compute and store score
                    queryHog_score[q][hog_offsets] = np.array(queryHog_cum_count[q][hog_offsets], dtype=np.float64) / qh_count
                    
                    # store best path
                    self.store_bestpath(hog_offsets, parent_offsets, queryHog_bestpath[q], queryHog_score[q], False)

        return queryHog_score, queryHog_bestpath

    @staticmethod
    def store_bestpath(hog_offsets, parent_offsets, queryHog_bestpath, queryHog_score, pv=False):

        # keep HOGs descending from the best path
        cands = hog_offsets[queryHog_bestpath[parent_offsets]]

        # get the score of these candidates
        cands_scores = queryHog_score[cands]

        # find the candidate HOGs offsets with the higher count (>0) at this level
        if cands_scores.size > 0:
            
            # need smallest
            if pv:
                cands_offsets = np.where(cands_scores==np.min(cands_scores))[0]
            # need > 0 and max
            else:
                cands_offsets = np.where((cands_scores > 0) & (cands_scores==np.max(cands_scores)))[0]

            # if a single candidate, update the best path. Else, stop because of tie
            if cands_offsets.size == 1:
                queryHog_bestpath[cands[cands_offsets]] = True

    def compute_bestpath_nqueries_nfams(self, q_num, fam_ranked, queryHog_cum_count, fam_tab, level_arr, hog2parent):
        '''
        this one is to compute the best path w/o computing the score simultaneously
        '''
        queryHog_bestpath = np.full(queryHog_cum_count.shape, False)
        
        # loop through queries and families
        for q in range(q_num):
            for f in fam_ranked[q]:
                
                entry = fam_tab[f]
                level_off = entry['LevelOff']
                level_num = entry['LevelNum']
                fam_level_offsets = level_arr[level_off:np.int64(level_off + level_num + 2)]
                
                # use the cumulative counts to define the best path
                rh = fam_level_offsets[0]
                qh_score = queryHog_cum_count[q][rh]
                if qh_score > 0: queryHog_bestpath[q][rh] = True
                    
                # loop through hog levels
                for i in range(1, fam_level_offsets.size - 2):
                    x = fam_level_offsets[i: i + 2]
                    hog_offsets = np.arange(x[0], x[1])
                    
                    # grab parents
                    parent_offsets = hog2parent[hog_offsets]
                    
                    # store best path
                    self.store_bestpath(hog_offsets, parent_offsets, queryHog_bestpath[q], queryHog_cum_count[q])                
        
        return queryHog_bestpath

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

# class Search():

#     class ResultFormat(tables.IsDescription):
#         QueryOff = tables.UInt64Col(pos=1)
#         FamOff = tables.UInt64Col(pos=2)
#         FamScore = tables.Float64Col(pos=3)

#     def __init__(self, fs, name=None, path=None, nthreads=None):
        
#         assert fs.mode == 'r', 'FlatSearch must be opened in read mode.'
#         assert fs.db.mode == 'r', 'Database must be opened in read mode.'
#         assert fs.ki.mode == 'r', 'Index must be opened in read mode.'
        
#         # load fs, ki and db
#         self.db = fs.db
#         self.ki = fs.ki
#         self.fs = fs
        
#         # performance features
#         self.nthreads = nthreads if nthreads is not None else os.cpu_count()

#         # hdf5 file
#         self.name = "{}_{}".format(self.fs.name, name if name else 'search')
#         self.path = path if path else self.ki.path
#         self.file = "{}{}.h5".format(self.path, self.name)
        
#         if os.path.isfile(self.file):
#             self.mode = 'r'
#             self.se = tables.open_file(self.file, self.mode)

#         else:
#             self.mode = 'w'
#             self.se = tables.open_file(self.file, self.mode)
    
#     def __enter__(self):
#         return self
    
#     def __exit__(self, *_):
#         self.se.close()
        
#     def clean(self):
#         '''
#         close and remove hdf5 file
#         '''
#         self.__exit__()
#         try:
#             os.remove(self.file)
#         except FileNotFoundError:
#             print("{} already cleaned".format(self.file))
    
#     def reset_cache(self):
#         '''
#             Reset caches.
#         '''
#         del self.hog_cum_count
#         del self.total_occur

#     @cached_property
#     def hog_cum_count(self):
#         return self.cumulate_counts_nfams(self.ki._hog_count[:], self.db._fam_tab[:], self.db._level_arr[:], 
#             self.db._hog_tab.col('ParentOff'), self.cumulate_counts_1fam, self._sum, self._max)

#     @cached_property
#     def total_occur(self):
#         return self.ki._idx_arr[:].size

#     ### easy access to data via properties ###
#     @property
#     def _res_tab(self):

#         if '/Result' in self.se:
#             return self.se.root.Result
#         else:
#             return self.se.create_table('/', 'Result', self.ResultFormat, filters=self.db._compr)

#     @property
#     def _hog_score(self):
#         if '/HOGscore' in self.se:
#             return self.se.root.HOGscore
#         else:
#             return self.se.create_earray('/', 'HOGscore', tables.Float64Atom(self.db._hog_tab.nrows), (0,), filters=self.db._compr)

#     @property
#     def _bestpath_mask(self):
#         if '/BestPathMask' in self.se:
#             return self.se.root.BestPathMask
#         else:
#             return self.se.create_earray('/', 'BestPathMask', tables.BoolAtom(self.db._hog_tab.nrows), (0,), filters=self.db._compr)

    # def search(self, norm_fam_fun, top_n, prop_fun, norm_hog_fun, norm_fam_args=[], norm_hog_args=[]):
    #     '''
    #     args
    #      - norm_fam_fun: choice of function to normalize k-mer count of families
    #      - top_n: number of families to proceed with cumulation of HOG k-mer counts
    #      - prop_fun: function to decide how counts are cumulated (_max, _sum)
    #      - norm_hog_fun: choice of function to normalize k-mer count of HOGs

    #     additional args can be passed to norm_fam_fun and norm_hog_fun
    #     '''
    #     assert top_n <= self.db._fam_tab.nrows, 'top n is smaller than number of families'

        # # normalize at family level
        # print('compute family scores')
        # queryFam_score = norm_fam_fun(*norm_fam_args)

        # # resort families after normalization and keep top n
        # fam_ranked = self.resort_ranked_fam(self.fs._fam_ranked[:], queryFam_score)[:,:top_n]
        
        # print('propagate HOG counts bottom-up')
        # queryHog_cum_count = self.cumulate_counts_nfams_nqueries(fam_ranked, self.fs._queryHog_count[:], self.db._fam_tab[:], 
        #     self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), self.fs._query_count.nrows, self.cumulate_counts_1fam, self._sum, self._max)

        # print('compute HOG scores')
        # queryHog_score, queryHog_bestpath = norm_hog_fun(fam_ranked, queryHog_cum_count, *norm_hog_args)

        # print('store results')
        # # store results for hogs
        # self._hog_score.append(queryHog_score)
        # self._hog_score.flush()
        # self._bestpath_mask.append(queryHog_bestpath)
        # self._bestpath_mask.flush()

        # # store results for families
        # self._res_tab.append(list(zip(*[self.fs._query_id[:].flatten(), fam_ranked[:,0], 
        #     queryFam_score[np.arange(fam_ranked.shape[0])[:,None], fam_ranked[:,:1]]])))
        # self._res_tab.flush()

        # # reset cache
        # self.reset_cache()

        # # close and re-open in read mode
        # self.se.close()
        # self.mode = 'r'
        # self.se = tables.open_file(self.file, self.mode)


#     ### normalize family ####################################################################################################################
#     def norm_fam_naive(self):
#         '''
#         no normalization there
#         '''
#         # move to int because resort function requires -
#         return np.array(self.fs._queryFam_count, dtype=np.int16)


#     def norm_fam_query_size(self):
#         '''
#         simple division by the length of query
#         '''
#         # no need to broadcast because query_count is a vertical vector
#         # query_count_bc = np.full((self.fs._queryFam_count[:].shape[1], self.fs._queryFam_count[:].shape[0]), self.fs._query_count[:]).T
#         return self.fs._queryFam_count[:] / self.fs._query_count[:]

#     def norm_fam_null_model(self, bernoulli_fun, sf_fun):
#         '''
#         '''
#         # compute bernoulli p
#         if bernoulli_fun == self._bernoulli_fam_kmerfreq:
#             p = bernoulli_fun(self.fs._query_occur[:], self.fs._query_count[:], self.db._fam_tab.nrows)

#         elif bernoulli_fun == self._bernoulli_theo_kmerprob:
#             p = bernoulli_fun(self.ki.k, self.ki.DIGITS_AA.size, self.ki._fam_count[:])

#         elif bernoulli_fun == self._bernoulli_true_kmerfreq:
#             p = bernoulli_fun(self.fs._query_occur[:], self.fs._query_count[:], self.total_occur, self.ki._fam_count[:])

#         else: 
#             raise ValueError('This Bernoulli function does not exist')

#         return sf_fun(self.fs._queryFam_count[:] - 1, self.fs._query_count[:], p)

#     ### functions to compute p-values
#     @staticmethod
#     def _bernoulli_fam_kmerfreq(query_occur, query_count, fam_num):
#         '''
#         frequencies of families with k-mer
#         can be used as bernoulli assuming all families have equal size
#         '''
#         # return np.divide(query_occur, query_count, out=np.zeros_like(query_occur), where=query_count!=0) / fam_num

#         return query_occur / query_count / fam_num
    
#     @staticmethod
#     def _bernoulli_theo_kmerprob(k, digit_num, hog_count):
#         '''
#         prob of random k-mer is used for probability to get a k-mer at one location
#         Bernoulli is the joint probability of not having the k-mer in the family
#         '''
#         kmer_prob = (1 / digit_num) ** k

#         return 1 - (1 - kmer_prob) ** hog_count

#     @staticmethod
#     def _bernoulli_true_kmerfreq(query_occur, query_count, total_occur, hog_count):
#         '''
#         true k-mer frequencies are used for probability to get a k-mer at one location
#         Bernoulli is the joint probability of not having the k-mer in the family
#         '''
#         kmer_prob = query_occur / query_count / total_occur

#         return 1 - (1 - kmer_prob) ** hog_count

#     @staticmethod
#     def _sf_poisson(k, n, p):
#         return scipy.stats.poisson.sf(k, n * p)


#     ### sort families #######################################################################################################################
#     @staticmethod
#     def resort_ranked_fam(fam_ranked, queryFam_score, top_n=None):
        
#         # ranked scores following fam_ranked until top_n
#         ranked_score = queryFam_score[np.arange(fam_ranked.shape[0])[:,None], fam_ranked[:,:top_n]]
        
#         # get indices of the reverse sorted ranked scores
#         idx = (-ranked_score).argsort()
        
#         # back to the fam offsets by indexing fam_ranked with our indices
#         return fam_ranked[np.arange(fam_ranked.shape[0])[:,None], idx]


#     ### cumulate counts #####################################################################################################################
#     @staticmethod
#     @numba.njit
#     def _max(x, y):
#         return max(x, y)

#     @staticmethod
#     @numba.njit
#     def _sum(x, y):
#             return x + y 

#     @staticmethod
#     @numba.njit
#     def cumulate_counts_1fam(hog_cum_counts, fam_level_offsets, hog2parent, cum_fun, prop_fun):
        
#         current_best_child_count = np.zeros(hog_cum_counts.shape, dtype=np.uint64)
        
#         # iterate over level offsets backward
#         for i in range(fam_level_offsets.size - 2):
#             x = fam_level_offsets[-i - 3: -i - 1]

#             # when reaching level, sum all hog counts with their best child count
#             hog_cum_counts[x[0]:x[1]] = cum_fun(hog_cum_counts[x[0]:x[1]], current_best_child_count[x[0]:x[1]])   

#             # update current_best_child_count of the parents of the current hogs
#             for i in range(x[0], x[1]):
#                 parent_off = hog2parent[i]
                
#                 # only if parent exists
#                 if parent_off != -1:
#                     c = current_best_child_count[hog2parent[i]]
#                     current_best_child_count[hog2parent[i]] = prop_fun(c, hog_cum_counts[i])
    
#     @staticmethod
#     @numba.njit(parallel=True, nogil=True)
#     def cumulate_counts_nfams(hog_counts, fam_tab, level_arr, hog2parent, main_fun, cum_fun, prop_fun):

#         hog_cum_counts = hog_counts.copy()

#         for fam_off in numba.prange(fam_tab.size):
#             entry = fam_tab[fam_off]
#             level_off = entry['LevelOff']
#             level_num = entry['LevelNum']
#             fam_level_offsets = level_arr[level_off:np.int64(level_off + level_num + 2)]

#             main_fun(hog_cum_counts, fam_level_offsets, hog2parent, cum_fun, prop_fun)

#         return hog_cum_counts

#     @staticmethod
#     @numba.njit(parallel=True, nogil=True)
#     def cumulate_counts_nfams_nqueries(fam_results, hog_counts, fam_tab, level_arr, hog2parent, q_num, main_fun, cum_fun, prop_fun):
        
#         hog_cum_counts = hog_counts.copy()
        
#         # iterate queries
#         for q in numba.prange(q_num):
            
#             # iterate families
#             for fam_off in fam_results[q]:
                
#                 entry = fam_tab[fam_off]
#                 level_off = entry['LevelOff']
#                 level_num = entry['LevelNum']
#                 fam_level_offsets = level_arr[level_off:np.int64(level_off + level_num + 2)]
#                 main_fun(hog_cum_counts[q], fam_level_offsets, hog2parent, cum_fun, prop_fun)
            
#         return hog_cum_counts

    # ### normalize hogs ######################################################################################################################
    # def norm_hog_naive(self, fam_ranked, queryHog_cum_count):
    #     '''
    #     no normalization there
    #     '''
    #     queryHog_bestpath = self.compute_bestpath_nqueries_nfams(self.fs._query_count.nrows, fam_ranked, queryHog_cum_count, 
    #         self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'))
    #     return queryHog_cum_count, queryHog_bestpath

    # # def norm_hog_query_size(self, fam_ranked, queryHog_cum_count):
    # #     '''
    #     same but normalize first the queryHog_cum_count
    #     could be slightly more clever and remove part conserved in parent 
    #     '''
    #     # query_count_bc = np.full((queryHog_cum_count.shape[1], queryHog_cum_count.shape[0]), self.fs._query_count[:]).T
        
    #     queryHog_score = queryHog_cum_count / self.fs._query_count[:]

    #     queryHog_bestpath = self.compute_bestpath_nqueries_nfams(self.fs._query_count.nrows, fam_ranked, queryHog_score, 
    #         self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'))
        
    #     return queryHog_score, queryHog_bestpath

    # def norm_hog_correct_query_size(self, fam_ranked, queryHog_cum_count):
    #     return self._norm_hog_correct_query_size(queryHog_cum_count, fam_ranked, self.fs._query_count[:], 
    #         self.db._fam_tab[:], self.db._level_arr[:], self.db._hog_tab.col('ParentOff'), self.fs._queryHog_count[:])

#     def _norm_hog_correct_query_size(self, queryHog_cum_count, fam_ranked, query_count, fam_tab, level_arr, hog2parent, queryHog_count):

#         queryHog_score = np.zeros(queryHog_cum_count.shape, dtype=np.float64)
#         queryHog_bestpath = np.full(queryHog_cum_count.shape, False)
#         queryHog_revcum_count = np.full((queryHog_cum_count.shape[1], query_count.size), query_count.flatten(), dtype=np.uint16).T

#         for q in range(queryHog_cum_count.shape[0]):
#             for f in fam_ranked[q]:

#                 entry = fam_tab[f]
#                 level_off = entry['LevelOff']
#                 level_num = entry['LevelNum']
#                 fam_level_offsets = level_arr[level_off:np.int64(level_off + level_num + 2)]
                
#                 # compute and store root-HOG score
#                 rh = fam_level_offsets[0]
#                 queryHog_score[q][rh] = np.array(queryHog_cum_count[q][rh], dtype=np.float64) / query_count[q]
                
#                 # set the root-HOG as best path
#                 queryHog_bestpath[q][rh] = True
                
#                 for i in range(1, fam_level_offsets.size - 2):
#                     x = fam_level_offsets[i: i + 2]
#                     hog_offsets = np.arange(x[0], x[1])
                    
#                     # grab parents
#                     parent_offsets = hog2parent[hog_offsets]
                    
#                     # update query revcumcount, basically substracting queryParent counts from query count
#                     qh_count = queryHog_revcum_count[q][parent_offsets] - queryHog_count[q][parent_offsets]
#                     queryHog_revcum_count[q][hog_offsets] = qh_count
                    
#                     # compute and store score
#                     queryHog_score[q][hog_offsets] = np.array(queryHog_cum_count[q][hog_offsets], dtype=np.float64) / qh_count
                    
#                     # store best path
#                     self.store_bestpath(hog_offsets, parent_offsets, queryHog_bestpath[q], queryHog_score[q], False)

#         return queryHog_score, queryHog_bestpath

#     def norm_hog_null_model(self, fam_ranked, queryHog_cum_count, bernoulli_fun, sf_fun):

#         return  self._norm_hog_null_model(
#             queryHog_cum_count, self.hog_cum_count, self.fs._query_count[:], fam_ranked, self.db._fam_tab[:], self.db._level_arr[:], 
#             bernoulli_fun, self.fs._query_occur[:], sf_fun, self.db._hog_tab.col('ParentOff'), self.fs._queryHog_count[:], self.fs._queryHog_occur[:])

#     # new null model where division by k is applied first
#     def norm_hog_null_model_div6(self, fam_ranked, queryHog_cum_count, bernoulli_fun, sf_fun):
        
#         # ceil rounds up
#         queryHog_cum_count =  np.array(np.ceil(queryHog_cum_count/self.ki.k), dtype=np.uint64)
#         query_count = np.array(np.ceil(self.fs._query_count[:]/self.ki.k), dtype=np.uint64)
#         query_occur = np.array(np.ceil(self.fs._query_occur[:]/self.ki.k), dtype=np.uint64)
#         queryHog_count = np.array(np.ceil(self.fs._queryHog_count[:]/self.ki.k), dtype=np.uint64)
#         queryHog_occur = np.array(np.ceil(self.fs._queryHog_occur[:]/self.ki.k), dtype=np.uint64)

#         return  self._norm_hog_null_model(
#             queryHog_cum_count , self.hog_cum_count, query_count, fam_ranked, self.db._fam_tab[:], self.db._level_arr[:], 
#             bernoulli_fun, query_occur, sf_fun, self.db._hog_tab.col('ParentOff'), queryHog_count, queryHog_occur)


#     def _norm_hog_null_model(self, queryHog_cum_count, hog_cum_count, query_count, fam_ranked, fam_tab, 
#         level_arr, bernoulli_fun, query_occur, sf_fun, hog2parent, queryHog_count, queryHog_occur):

#         # initiate with 1's because p-value style
#         queryHog_score = np.ones(queryHog_cum_count.shape, dtype=np.float64)

#         queryHog_bestpath = np.full(queryHog_cum_count.shape, False)
#         queryHog_revcum_count = np.full((hog_cum_count.size, query_count.size), query_count.flatten(), dtype=np.uint16).T
#         queryHog_revcum_occur = np.full((hog_cum_count.size, query_occur.size), query_occur.flatten(), dtype=np.uint32).T
        
#         # loop through queries and families
#         for q in range(queryHog_cum_count.shape[0]):
#             for f in fam_ranked[q]:
                
#                 entry = fam_tab[f]
#                 level_off = entry['LevelOff']
#                 level_num = entry['LevelNum']
#                 fam_level_offsets = level_arr[level_off:np.int64(level_off + level_num + 2)]
                
#                 # compute the score for the root-HOG (level 0)
#                 rh = fam_level_offsets[0]

#                 # compute bernoulli p
#                 if bernoulli_fun == self._bernoulli_fam_kmerfreq:
#                     p = bernoulli_fun(query_occur[q], query_count[q], fam_tab.size)
#                 elif bernoulli_fun == self._bernoulli_theo_kmerprob:
#                     p = bernoulli_fun(self.ki.k, self.ki.DIGITS_AA.size, hog_cum_count[rh])
#                 elif bernoulli_fun == self._bernoulli_true_kmerfreq:
#                     p = bernoulli_fun(query_occur[q], query_count[q], self.total_occur, hog_cum_count[rh])
#                 else: 
#                     raise ValueError('This Bernoulli function does not exist')

#                 # compute p-value
#                 # gives output of 1 if not converted to float. weird
#                 queryHog_score[q][rh] = sf_fun(np.array(queryHog_cum_count[q][rh], dtype=np.float64) - 1, np.array(query_count[q], dtype=np.float64), p)

#                 # set the root-HOG as best path
#                 queryHog_bestpath[q][rh] = True

#                 # compute the score for child-HOGs (level 1 - n)
#                 for i in range(1, fam_level_offsets.size - 2):
#                     x = fam_level_offsets[i: i + 2]
#                     hog_offsets = np.arange(x[0], x[1])

#                     # grab parents
#                     parent_offsets = hog2parent[hog_offsets]

#                     # update query count and occur
#                     qh_count = queryHog_revcum_count[q][parent_offsets] - queryHog_count[q][parent_offsets]
#                     qh_occur = queryHog_revcum_occur[q][parent_offsets] - queryHog_occur[q][parent_offsets]
                    
#                     queryHog_revcum_count[q][hog_offsets] = qh_count    
#                     queryHog_revcum_occur[q][hog_offsets] = qh_occur

#                     # compute bernoulli p
#                     if bernoulli_fun == self._bernoulli_fam_kmerfreq:
#                         p = bernoulli_fun(qh_occur, qh_count, fam_tab.size)

#                     elif bernoulli_fun == self._bernoulli_theo_kmerprob:
#                         p = bernoulli_fun(self.ki.k, self.ki.DIGITS_AA.size, hog_cum_count[hog_offsets])

#                     elif bernoulli_fun == self._bernoulli_true_kmerfreq:
#                         p = bernoulli_fun(qh_occur, qh_count, self.total_occur, hog_cum_count[hog_offsets])

#                     else: 
#                         raise ValueError('This Bernoulli function does not exist')

#                     # compute p-value
#                     # gives output of 1 if not converted to float. weird
#                     queryHog_score[q][hog_offsets] = sf_fun(np.array(queryHog_cum_count[q][hog_offsets], dtype=np.float64) - 1, np.array(qh_count, dtype=np.float64), p)
                    
#                     # store best path
#                     self.store_bestpath(hog_offsets, parent_offsets, queryHog_bestpath[q], queryHog_score[q], pv=True)
    
#         return queryHog_score, queryHog_bestpath

# #     @staticmethod
# #     def store_bestpath(hog_offsets, parent_offsets, queryHog_bestpath, queryHog_score, pv=False):

#         # keep HOGs descending from the best path
#         cands = hog_offsets[queryHog_bestpath[parent_offsets]]

#         # get the score of these candidates
#         cands_scores = queryHog_score[cands]

#         # find the candidate HOGs offsets with the higher count (>0) at this level
#         if cands_scores.size > 0:
            
#             # need smallest
#             if pv:
#                 cands_offsets = np.where(cands_scores==np.min(cands_scores))[0]
#             # need > 0 and max
#             else:
#                 cands_offsets = np.where((cands_scores > 0) & (cands_scores==np.max(cands_scores)))[0]

#             # if a single candidate, update the best path. Else, stop because of tie
#             if cands_offsets.size == 1:
#                 queryHog_bestpath[cands[cands_offsets]] = True

#     def compute_bestpath_nqueries_nfams(self, q_num, fam_ranked, queryHog_cum_count, fam_tab, level_arr, hog2parent):
#         '''
#         this one is to compute the best path w/o computing the score simultaneously
#         '''
#         queryHog_bestpath = np.full(queryHog_cum_count.shape, False)
        
#         # loop through queries and families
#         for q in range(q_num):
#             for f in fam_ranked[q]:
                
#                 entry = fam_tab[f]
#                 level_off = entry['LevelOff']
#                 level_num = entry['LevelNum']
#                 fam_level_offsets = level_arr[level_off:np.int64(level_off + level_num + 2)]
                
#                 # use the cumulative counts to define the best path
#                 rh = fam_level_offsets[0]
#                 qh_score = queryHog_cum_count[q][rh]
#                 if qh_score > 0: queryHog_bestpath[q][rh] = True
                    
#                 # loop through hog levels
#                 for i in range(1, fam_level_offsets.size - 2):
#                     x = fam_level_offsets[i: i + 2]
#                     hog_offsets = np.arange(x[0], x[1])
                    
#                     # grab parents
#                     parent_offsets = hog2parent[hog_offsets]
                    
#                     # store best path
#                     self.store_bestpath(hog_offsets, parent_offsets, queryHog_bestpath[q], queryHog_cum_count[q])                
        
#         return queryHog_bestpath
