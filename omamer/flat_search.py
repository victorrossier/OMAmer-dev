import os
import sys
import numba
import tables
import numpy as np
from tqdm import tqdm
from property_manager import lazy_property, cached_property

from .index import get_transform, SequenceBuffer, QuerySequenceBuffer

# FlatSearchValidation
from .hierarchy import get_sispecies_candidates, compute_inparalog_coverage_new

# DIAMOND
from .index import DIAMONDindexValidation
from Bio import SeqIO, SearchIO
from shutil import copyfile

# SW
from pyoma.browser.db import Database
from scipy.stats import rankdata
from tqdm import tqdm
import random

# # Orlov
# sys.path.append("/Users/Victor/Documents/UNIL/PhD/03_projects/01_OMAmer/code/orlov_hog_dev")
# from skbio.alignment import StripedSmithWaterman
# from orlov.matrices import load_matrix
# from multiprocessing import sharedctypes
# import multiprocessing as mp
# import ctypes

'''
TO DO:
 - check parallelization OMAmer

 Nice to have:
 - store query k-mer locations
 - merge/unify QueryID and result table
'''

class FlatSearch():

    def __init__(
        self, ki, name=None, path=None, nthreads=None, low_mem=False):
        
        assert ki.mode == 'r', 'Index must be opened in read mode.'
        assert ki.db.mode == 'r', 'Database must be opened in read mode.'
        
        # load ki and db
        self.db = ki.db
        self.ki = ki
        
        # performance features
        self.nthreads = nthreads if nthreads is not None else os.cpu_count()
        self.low_mem = low_mem    

        # hdf5 file
        self.name = "{}_{}".format(self.ki.name, name)
        self.path = path if path else self.ki.path
        self.file = "{}{}.h5".format(self.path, self.name)
        
        if os.path.isfile(self.file):
            self.mode = 'r'
            self.fs = tables.open_file(self.file, self.mode)
        else:
            self.mode = 'w'
            self.fs = tables.open_file(self.file, self.mode)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.fs.close()
        
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
        del self.trans
        del self.table_idx
        del self.table_buff
        del self.prot2hog
        del self.hog2fam

    # cached properties from Index
    @cached_property
    def trans(self):
        return get_transform(self.ki.k, self.ki.alphabet.DIGITS_AA)

    @cached_property
    def table_idx(self):
        x = self.ki._table_idx
        return x[:] if not self.low_mem else x

    @cached_property
    def table_buff(self):
        x = self.ki._table_buff
        return x[:] if not self.low_mem else x

    @cached_property
    def prot2hog(self):
        return self.db._prot_tab.col('HOGoff')

    @cached_property
    def hog2fam(self):
        return self.db._hog_tab.col('FamOff')

    @property
    def _query_id(self):
        if '/QueryID' in self.fs:
            return self.fs.root.QueryID
        else:
            return self.fs.create_earray('/', 'QueryID', tables.StringAtom(255), (0,), filters=self.db._compr)

    # data for flat search
    @property
    def _fam_ranked(self):
        if '/FamRanked' in self.fs:
            return self.fs.root.FamRanked
        else:
            return self.fs.create_earray('/', 'FamRanked', tables.UInt32Atom(self.db._fam_tab.nrows), (0,), filters=self.db._compr)

    @property
    def _queryFam_count(self):
        if '/QueryFamCount' in self.fs:
            return self.fs.root.QueryFamCount
        else:
            return self.fs.create_earray('/', 'QueryFamCount', tables.UInt16Atom(self.db._fam_tab.nrows), (0,), filters=self.db._compr)

    @property
    def _queryFam_occur(self):
        if '/QueryFamOccur' in self.fs:
            return self.fs.root.QueryFamOccur
        else:
            return self.fs.create_earray('/', 'QueryFamOccur', tables.UInt32Atom(self.db._fam_tab.nrows), (0,), filters=self.db._compr)

    @property
    def _queryHog_count(self):
        if '/QueryHogCount' in self.fs:
            return self.fs.root.QueryHogCount
        else:
            return self.fs.create_earray('/', 'QueryHogCount', tables.UInt16Atom(self.db._hog_tab.nrows), (0,), filters=self.db._compr)

    @property
    def _queryHog_occur(self):
        if '/QueryHogOccur' in self.fs:
            return self.fs.root.QueryHogOccur
        else:
            return self.fs.create_earray('/', 'QueryHogOccur', tables.UInt32Atom(self.db._hog_tab.nrows), (0,), filters=self.db._compr)

    @property
    def _query_count(self):
        if '/QueryCount' in self.fs:
            return self.fs.root.QueryCount
        else:
            return self.fs.create_earray('/', 'QueryCount', tables.UInt16Atom(1), (0,), filters=self.db._compr)

    @property
    def _query_occur(self):
        if '/QueryOccur' in self.fs:
            return self.fs.root.QueryOccur
        else:
            return self.fs.create_earray('/', 'QueryOccur', tables.UInt32Atom(1), (0,), filters=self.db._compr)

    # flat search function
    def flat_search(self, seqs=None, ids=None, fasta_file=None, query_sp=None, chunksize=500):
        '''
        to limit code duplication in FlatSearchValidation 
        '''
        assert (self.mode in {'w', 'a'}), 'Search must be opened in write mode.'
        
        # load query sequences
        if seqs:
            sb = SequenceBuffer(seqs=seqs, ids=ids)
        elif fasta_file:
            sb = SequenceBuffer(fasta_file=fasta_file)
        elif query_sp:
            self.query_species = ''.join(query_sp.split())
            sb = QuerySequenceBuffer(db=self.db, query_sp=self.query_species)

        self._flat_search(sb, chunksize)

    def _flat_search(self, sbuff, chunksize):

        # to reduce memory footprint, process in chunk
        i = 0
        while i <= sbuff.prot_nr:
            print('look-up k-mer table')
            seqs_idx = sbuff.idx[i:i + chunksize + 1]
            seqs = sbuff.buff[seqs_idx[0]:seqs_idx[-1]]
            
            # important to reinitializae seqs_idx
            seqs_idx -= seqs_idx[0]
            seq_ids = sbuff.ids[i:i + chunksize]
            
            # search
            (fam_ranked, queryFam_count, queryFam_occur, queryHog_count, queryHog_occur, query_count, query_occur) = self._lookup(
                        seqs, seqs_idx, self.trans, self.table_idx, self.table_buff, self.db._hog_tab.col('FamOff'), self.db._fam_tab.nrows, 
                        self.db._hog_tab.nrows, self.ki.k, self.ki.alphabet.DIGITS_AA_LOOKUP)
            
            print('store results for flat search')
            self._fam_ranked.append(fam_ranked)
            self._fam_ranked.flush()
            self._queryFam_count.append(queryFam_count)
            self._queryFam_count.flush()
            self._queryFam_occur.append(queryFam_occur)
            self._queryFam_occur.flush()
            self._queryHog_count.append(queryHog_count)
            self._queryHog_count.flush()
            self._queryHog_occur.append(queryHog_occur)
            self._queryHog_occur.flush()

            # store as vertical vectors
            self._query_count.append(query_count[:,None])
            self._query_count.flush()
            self._query_occur.append(query_occur[:,None])
            self._query_occur.flush()

            # store ids of sbuff
            self.append_queries(seq_ids)
            
            i += chunksize

        # self.reset_cache() --> not sure if necessary

        # close and re-open in read mode
        self.fs.close()
        self.mode = 'r'
        self.fs = tables.open_file(self.file, self.mode)

    def append_queries(self, ids):
        self._query_id.append(ids.flatten())

    @lazy_property
    def _lookup(self):
        def func(seqs, seqs_idx, trans, table_idx, table_buff, hog2fam, n_fams, n_hogs, k, DIGITS_AA_LOOKUP):

            # updated to deal with Xs
            fam_results = np.zeros((len(seqs_idx) - 1, n_fams), dtype=np.uint32)
            fam_counts = np.zeros((len(seqs_idx) - 1, n_fams), dtype=np.uint16)
            fam_occur = np.zeros((len(seqs_idx) - 1, n_fams), dtype=np.uint32)  # ()
            hog_counts = np.zeros((len(seqs_idx) - 1, n_hogs), dtype=np.uint16)
            hog_occur = np.zeros((len(seqs_idx) - 1, n_hogs), dtype=np.uint32)  # ()
            query_counts = np.zeros((len(seqs_idx) - 1), dtype=np.uint16)  # ()
            query_occur = np.zeros((len(seqs_idx) - 1), dtype=np.uint32)  # ()
            x_char = DIGITS_AA_LOOKUP[88]  # 88 == b'X'
            x_kmer = 0
            for j in range(k):
                x_kmer += trans[j] * x_char
            x_kmer += 1
            for zz in numba.prange(len(seqs_idx) - 1):
                # grab query sequence
                s = seqs[seqs_idx[zz] : int(seqs_idx[zz + 1] - 1)]
                n_kmers = s.shape[0] - (k - 1)
                # double check we don't have short peptides
                # note: written in this order to provide loop-optimisation hint
                if n_kmers > 0:
                    pass
                else:
                    continue
                # parse into k-mers
                # TO DO: get locations simultaneously
                s_norm = DIGITS_AA_LOOKUP[s]
                r = np.zeros(n_kmers, dtype=np.uint32)  # max kmer 7
                for i in numba.prange(n_kmers):
                    # numba can't do np.dot with non-float
                    for j in range(k):
                        r[i] += trans[j] * s_norm[i + j]
                    x_seen = np.any(s_norm[i:i+k] == x_char)
                    r[i] = r[i] if not x_seen else x_kmer
                r1 = np.unique(r)
                if len(r1) > 1:
                    pass
                elif r1[0] == x_kmer:
                    continue
                
                # search hogs and fams
                hog_res = np.zeros(n_hogs, dtype=np.uint16)
                fam_res = np.zeros(n_fams, dtype=np.uint16)
                hog_occ = np.zeros(n_hogs, dtype=np.uint32)
                fam_occ = np.zeros(n_fams, dtype=np.uint32)
                query_occ = 0
                for m in numba.prange(r1.shape[0]):
                    kmer = r1[m]
                    if kmer < x_kmer:
                        pass
                    else:
                        continue
                    x = table_idx[kmer : kmer + 2]
                    hogs = table_buff[x[0]:x[1]]
                    fams = hog2fam[hogs]
                    hog_res[hogs] += np.uint16(1)
                    fam_res[fams] += np.uint16(1)
                    # kmer_occ is the nr of hogs/fams with the given kmer, used to compute its frequency
                    kmer_occ = hogs.size # the built-in function len made a type error only when parallel was ON
                    # store for the set of query k-mers 
                    query_occ += kmer_occ
                    # and for the set of intersecting k-mers between the query and every hogs and fams
                    hog_occ[hogs] += np.uint32(kmer_occ)
                    fam_occ[fams] += np.uint32(kmer_occ)  # I think here it should be len(fams) instead of len(hogs) ... # actually fine because one hog for one fam...

                # TO DO: store only top_n families and corresponding HOGs
                # report results for families sorted by k-mer count
                t = np.argsort(fam_res)[::-1]  
                fam_results[zz, :n_fams] = t
                # not sorting anymore the corresponding fam counts and occur
                fam_counts[zz, :n_fams] = fam_res
                fam_occur[zz, :n_fams] = fam_occ
                # fam_counts[zz, :n_fams] = fam_res[t]
                # fam_occur[zz, :n_fams] = fam_occ[t]
                # report raw results for hogs 
                hog_counts[zz, :n_hogs] = hog_res
                hog_occur[zz, :n_hogs] = hog_occ
                # report results for the query
                query_counts[zz] = len(r1)
                query_occur[zz] = query_occ

            return fam_results, fam_counts, fam_occur, hog_counts, hog_occur, query_counts, query_occur

        if not self.low_mem:
            # Set nthreads, note: this only works before numba called first time!
            numba.config.NUMBA_NUM_THREADS = self.nthreads
            return numba.jit(func, parallel=True, nopython=True, nogil=True)
        else:
            return func


class FlatSearchValidation(FlatSearch):

    def __init__(self, ki, name=None, path=None, nthreads=None, low_mem=False, query_species=None):

        # set query species
        assert query_species, 'A query species must be defined.'
        self.query_species = ''.join(query_species.split())

        # create name from query species
        name = '_'.join([self.query_species, name]) if name else self.query_species

        # call init of parent class
        super().__init__(ki, name, path, nthreads, low_mem)        

    # query ids are integer now 
    @property
    def _query_id(self):
        if '/QueryID' in self.fs:
            return self.fs.root.QueryID
        else:
            return self.fs.create_earray('/', 'QueryID', tables.UInt64Atom(1), (0,), filters=self.db._compr)

    def flat_search(self, chunksize=500):

        assert (self.mode in {'w', 'a'}), 'Search must be opened in write mode.'

        # load query sequences
        qbuff = QuerySequenceBuffer(db=self.db, query_sp=self.query_species)

        self._flat_search(qbuff, chunksize)

    def append_queries(self, ids):
        self._query_id.append(ids)

    def compute_subfamily_coverage(self):

        # query taxon offsets
        tax_off = np.searchsorted(self.db._tax_tab.col('ID'), self.query_species)

        # hidden taxa offsets
        hidden_taxa = [np.searchsorted(self.db._tax_tab.col('ID'), x) for x in self.ki.hidden_taxa]

        # find sister species
        sispecies_cands = hierarchy.get_sispecies_candidates(tax_off, self.db._tax_tab[:], self.db._ctax_arr[:], hidden_taxa)

        query_ids = self._query_id[:].flatten()
        prot_tab = self.db._prot_tab[:]
        cprot_buff = self.db._cprot_arr[:]
        hog_tab = self.db._hog_tab[:]
        chog_buff = self.db._chog_arr[:]

        return np.array([hierarchy.compute_inparalog_coverage_new(
            q, query_ids, prot_tab, cprot_buff, hog_tab, chog_buff, hidden_taxa, sispecies_cands) for q in range(query_ids.size)])

# END of OMAmer
################################################################################################################################################
# baseline classes mimicking IndexValidation structure

class DIAMONDsearch():
    '''
    to export query fasta for BLAST/DIAMOND, parse BLAST output and mimic FlatSearch
    Note, BLAST/DIAMOND have to be ran independly from the outputed fasta of ReferenceExport and BLASTsearch
    '''
    # result format for CS flat search
    class CSresultFormat(tables.IsDescription):
        QueryOff = tables.Int64Col(pos=1)
        FamOff = tables.Int64Col(pos=2)
        HOGoff = tables.Int64Col(pos=3)
        Score = tables.Float64Col(pos=4)

    def __init__(self, ki, name, fasta=None, qr_name=None, mode=None):
        
        assert ki.db.mode == 'r', 'Database must be opened in read mode.'
        
        # load ki and db
        self.db = ki.db
        self.ki = ki
        
        # name of the search
        self.name = "{}_{}".format(self.ki.name, name)
        self.fasta = fasta
        self.query_path = '{}query/{}/'.format(self.ki.path, qr_name if qr_name else name)
        self.result_path = '{}result/{}/'.format(self.ki.path, qr_name if qr_name else name)

        if not os.path.exists(self.query_path):
            os.makedirs(self.query_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # hdf5 file to store CS results and query id table
        self.path = self.ki.path
        self.file = "{}{}.h5".format(self.path, self.name)
        
        if os.path.isfile(self.file):
            # to support three steps : initiation + validation + read
            # 1. mode 'w', 2. mode 'w', 3. mode 'r'
            self.mode = mode if mode else 'r' 
            self.fs = tables.open_file(self.file, self.mode)
        else:
            self.mode = 'w'
            self.fs = tables.open_file(self.file, self.mode)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.fs.close()
        
    def clean(self):
        '''
        close and remove hdf5 file
        '''
        self.__exit__()
        try:
            os.remove(self.file)
        except FileNotFoundError:
            print("{} already cleaned".format(self.file))

    ### query IDs ###
    @property
    def _query_id(self):
        if '/QueryID' in self.fs:
            return self.fs.root.QueryID
        else:
            return self.fs.create_earray('/', 'QueryID', tables.StringAtom(255), (0,), filters=self.db._compr)

    ### data for CS flat search ###
    @property
    def _res_tab(self):
        if '/Result' in self.fs:
            return self.fs.root.Result
        else:
            return self.fs.create_table('/', 'Result', self.CSresultFormat, filters=self.db._compr)

    def export_query_fasta(self):

        sb = SequenceBuffer(fasta_file=self.fasta)

        # TO DO: option to split fasta for parallel BLAST
        copyfile(self.fasta, '{}0.fa'.format(self.query_path))

        self._query_id.append(sb.ids)

    @staticmethod
    def parse_blast_result(result_path, prot_tab, score):

        result_files = sorted(list(os.listdir(result_path)))

        assert result_files, 'No BLAST results'

        qi_rows = []
        res_rows = []

        for rf in result_files:
            rf = '{}{}'.format(result_path, rf)
            result = SearchIO.parse(rf, 'blast-tab')

            for res in result:
                qi = res.id
                qi_rows.append(qi)
                ti = int(res[0].id)
                s = (res[0][0].bitscore if score=='bitscore' else res[0][0].evalue)
                res_rows.append((int(qi), prot_tab[ti]['FamOff'], prot_tab[ti]['HOGoff'], s))

        return qi_rows, res_rows

    @staticmethod
    def interleave_na_results(query_ids, tmp_res_rows, prob):
        '''
        some query might not have any hit with BLAST/DIAMOND but should still be present in the result table
        '''
        qi2res = dict([ (k, (x, y, z)) for k, x, y, z in tmp_res_rows])
        return [(int(qi), *qi2res.get(int(qi), (-1, -1, 1000000 if prob else -1))) for qi in query_ids.flatten()]

    def import_blast_result(self, score='bitscore', prob=False):

        qi_rows, tmp_res_rows = self.parse_blast_result(self.result_path, self.db._prot_tab[:], score)

        if len(self._query_id) != len(tmp_res_rows):
            res_rows = self.interleave_na_results(self._query_id[:], tmp_res_rows, prob)
        else:
            res_rows = tmp_res_rows

        self._res_tab.append(res_rows)
        self._res_tab.flush()

        assert len(self._res_tab) == len(self._query_id)

        # close and re-open in read mode
        self.fs.close()
        self.mode = 'r'
        self.fs = tables.open_file(self.file, self.mode)


class DIAMONDsearchValidation(DIAMONDsearch):
    '''
    to export query fasta for BLAST/DIAMOND, parse BLAST output and mimic FlatSearchValidation
    Note, BLAST/DIAMOND have to be ran independly from the outputed fasta of ReferenceExport and BLASTsearch
    '''
    def __init__(self, ki, name=None, query_species=None, mode=None):

        # set query species
        assert query_species, 'A query species must be defined.'
        self.query_species = ''.join(query_species.split())

        # create name from query species
        name = self.query_species

        # call init of parent class
        super().__init__(ki, name, mode)

    @property
    def _query_id(self):
        if '/QueryID' in self.fs:
            return self.fs.root.QueryID
        else:
            return self.fs.create_earray('/', 'QueryID', tables.UInt64Atom(1), (0,), filters=self.db._compr)

    def export_query_fasta(self):
        '''
        export query from SPHOG database
        '''
        sb = QuerySequenceBuffer(db=self.db, query_sp=self.query_species)
        ids = sb.ids.flatten()

        # TO DO: option to split fasta for parallel BLAST
        with open("{}0.fa".format(self.query_path), 'w') as ff:
            for idx, seq in enumerate(sb):
                ff.write(">{}\n{}\n".format(ids[idx], seq))

        self._query_id.append(sb.ids[:])
        #self.fs.close()


class SWsearchValidation():
    '''
    get SW closest sequences precomputed in OMA
    '''
    class CSresultFormat(tables.IsDescription):
        QueryOff = tables.Int64Col(pos=1)
        FamOff = tables.Int64Col(pos=2)
        HOGoff = tables.Int64Col(pos=3)
        Score = tables.Float64Col(pos=4)

    def __init__(self, ki, query_species=None):

        assert ki.db.mode == 'r', 'Database must be opened in read mode.'
        
        # load ki and db
        self.db = ki.db
        self.ki = ki
        
        # set query species
        assert query_species, 'A query species must be defined.'
        self.query_species = ''.join(query_species.split())

        # create name from query species
        name = self.query_species
        self.name = "{}_{}".format(self.ki.name, name)

        # hdf5 file to store CS results and query id table
        self.path = self.ki.path
        self.file = "{}{}.h5".format(self.path, self.name)
        
        if os.path.isfile(self.file):
            self.mode = 'r'
            self.fs = tables.open_file(self.file, self.mode)
        else:
            self.mode = 'w'
            self.fs = tables.open_file(self.file, self.mode)

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.fs.close()
        
    def clean(self):
        '''
        close and remove hdf5 file
        '''
        self.__exit__()
        try:
            os.remove(self.file)
        except FileNotFoundError:
            print("{} already cleaned".format(self.file))
    
    @property
    def _query_id(self):
        if '/QueryID' in self.fs:
            return self.fs.root.QueryID
        else:
            return self.fs.create_earray('/', 'QueryID', tables.UInt64Atom(1), (0,), filters=self.db._compr)

    @property
    def _res_tab(self):
        if '/Result' in self.fs:
            return self.fs.root.Result
        else:
            return self.fs.create_table('/', 'Result', self.CSresultFormat, filters=self.db._compr)

    def get_closest(self, oma_h5_path):

        prob = False # bitscore

        prot_tab = self.db._prot_tab[:]
        sp_filter = self.ki.sp_filter

        # load queries
        sb = QuerySequenceBuffer(db=self.db, query_sp=self.query_species)
        ids = sb.ids.flatten()
        oma_entries = prot_tab['ID'][ids]

        # mapper oma entries to protein offsets
        id2off = dict(zip(prot_tab['ID'], range(len(prot_tab))))

        # parser of OMA Verified pairs .. Alex code
        closest = ClosestEntries(oma_h5_path)

        rows = []

        non_ref_count = 0
        i = 0
        for entry in tqdm(oma_entries):
            
            cs, score, non_ref_2 = self._get_closest(int(entry), sp_filter, prot_tab, id2off, closest)
            
            if cs:
                rows.append((ids[i], prot_tab['FamOff'][cs], prot_tab['HOGoff'][cs], score))
            else:
                rows.append((ids[i], -1, -1, 1000000 if prob else 0))

            if non_ref_2:
                non_ref_count += 1
            
            i+=1

        print('{} queries with closest sequence not in reference proteins'.format(non_ref_count))

        self._res_tab.append(rows)
        self._res_tab.flush()
        self._query_id.append(sb.ids[:])
        self._query_id.flush()

        del prot_tab, sp_filter

        # close and re-open in read mode
        self.fs.close()
        self.mode = 'r'
        self.fs = tables.open_file(self.file, self.mode)

    @staticmethod
    def _get_closest(entry, sp_filter, prot_tab, id2off, closest):
        '''
        find the closest sequence that is not hidden nor not-referenced by iterating over ranks
        '''

        def _filter_cand(cand, sp_filter, prot_tab, id2off):
            '''
            filter hidden species and non-referenced proteins
            '''
            f_cand = set()
            non_ref = False
            for x in cand:
                x_off = id2off.get(str(x).encode('ascii'), None)
                
                # filter out non-referenced proteins
                if x_off:
                    # filter out hidden proteins
                    if not sp_filter[prot_tab[x_off]['SpeOff']]:
                        
                        f_cand.add(x)

                # set non_ref flag to True
                else:
                    non_ref = True

            return f_cand, non_ref

        cs = None
        rank = 1
        cand, score = closest.by_score(entry, rank=rank)

        non_ref_2 = False

        while cand:
            f_cand, non_ref = _filter_cand(cand, sp_filter, prot_tab, id2off)
            if f_cand:
                cs = id2off[str(random.choice(list(f_cand))).encode('ascii')]
                break
            elif non_ref:
                non_ref_2 = True
                break
            else:
                rank += 1
                cand, score = closest.by_score(entry, rank=rank)

        return cs, score, non_ref_2

class ClosestEntries(object):
    def __init__(self, db_fn, as_enum=False):
        self.db = Database(db_fn)
        self.as_enum = as_enum
    
    def ensure_enum(self, entry):
        if isinstance(entry, int):
            return entry
        else:
            return self.db.id_mapper['Oma'].omaid_to_entry_nr(entry)
    
    def by_dist(self, entry, rank=1):
        x = self.db.get_vpairs(self.ensure_enum(entry))
        r = rankdata(x['Distance'], method='dense')

        return set(x[r == rank]['EntryNr2']), x[r == rank]['Distance'][0]

    def by_score(self, entry, rank=1):
        x = self.db.get_vpairs(self.ensure_enum(entry))
        r = rankdata(-x['Score'], method='dense')

        return set(x[r == rank]['EntryNr2']), x[r == rank]['Score'][0]

# #######################################################################################################################################
# # for Orlov
# glob = None
# def worker_init(matrix, gap_open, gap_extend, qbuff, sbuff, scores_shr, scores_shape):
#     global glob

#     glob = {}
#     glob['MATRIX'] = load_matrix(matrix)
#     glob['GAP_OPEN'] = gap_open
#     glob['GAP_EXTEND'] = gap_extend

#     glob['qbuff'] = qbuff
#     glob['sbuff'] = sbuff
#     glob['scores'] = np.frombuffer(scores_shr, dtype=np.float64).reshape(scores_shape)

# def worker_align(i, j, z):
#     global glob
#     aligner = StripedSmithWaterman(
#                   query_sequence=glob['qbuff'][i],
#                   gap_open_penalty=glob['GAP_OPEN'],
#                   gap_extend_penalty=glob['GAP_EXTEND'],
#                   protein=True,
#                   substitution_matrix=glob['MATRIX'],
#                   score_only=True,
#                   suppress_sequences=True)

#     glob['scores'][i][j] = aligner(glob['sbuff'][z])['optimal_alignment_score']


# class OrlovFlatSearchValidation():

#     class ResultFormat(tables.IsDescription):
#         QueryOff = tables.UInt64Col(pos=1)
#         FamOff = tables.UInt64Col(pos=2)
#         HOGoff = tables.UInt64Col(pos=3)
#         Score = tables.Float64Col(pos=4)

#     def __init__(
#         self, ki, name=None, path=None, n_top=50, matrix='BLOSUM62', gap_open=11, gap_extend=1, nthreads=None, low_mem=False, query_species=None):
        
#         assert ki.mode == 'r', 'Index must be opened in read mode.'
#         assert ki.db.mode == 'r', 'Database must be opened in read mode.'
#         assert query_species, 'A query species must be defined.'
#         self.query_species = ''.join(query_species.split())

#         # create name from query species
#         name = '_'.join([self.query_species, name]) if name else self.query_species
        
#         # load ki and db
#         self.db = ki.db
#         self.ki = ki
        
#         self.n_top = n_top if n_top < len(self.db._prot_tab) else len(self.db._prot_tab)  # because broadcasting issue when parallelized
#         self.matrix = matrix
#         self.gap_open = gap_open
#         self.gap_extend = gap_extend
        
#         # performance features
#         self.nthreads = nthreads if nthreads is not None else os.cpu_count()
#         self.low_mem = low_mem    

#         # hdf5 file
#         self.name = "{}_{}".format(self.ki.name, name)
#         self.path = path if path else self.ki.path
#         self.file = "{}{}.h5".format(self.path, self.name)
        
#         if os.path.isfile(self.file):
#             self.mode = 'r'
#             self.fs = tables.open_file(self.file, self.mode)
#         else:
#             self.mode = 'w'
#             self.fs = tables.open_file(self.file, self.mode)
    
#     def __enter__(self):
#         return self
    
#     def __exit__(self, *_):
#         self.fs.close()
        
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
#         del self.trans
#         del self.table_idx
#         del self.table_buff
#         del self.prot2hog
#         del self.hog2fam

#     ### cached properties from Index ###
#     @cached_property
#     def trans(self):
#         return get_transform(self.ki.k, self.ki.alphabet.DIGITS_AA)

#     @cached_property
#     def table_idx(self):
#         x = self.ki._table_idx
#         return x[:] if not self.low_mem else x

#     @cached_property
#     def table_buff(self):
#         x = self.ki._table_buff
#         return x[:] if not self.low_mem else x

#     @cached_property
#     def prot2hog(self):
#         return self.db._prot_tab.col('HOGoff')

#     @cached_property
#     def hog2fam(self):
#         return self.db._hog_tab.col('FamOff')

#     ### query IDs ###
#     @property
#     def _query_id(self):
#         if '/QueryID' in self.fs:
#             return self.fs.root.QueryID
#         else:
#             return self.fs.create_earray('/', 'QueryID', tables.UInt64Atom(1), (0,), filters=self.db._compr)

#     @property
#     def _res_tab(self):
#         if '/Result' in self.fs:
#             return self.fs.root.Result
#         else:
#             return self.fs.create_table('/', 'Result', self.ResultFormat, filters=self.db._compr)

#     def flat_search(self, align=True):

#         assert (self.mode in {'w', 'a'}), 'Search must be opened in write mode.'

#         # load query sequences
#         qbuff = QuerySequenceBuffer(db=self.db, query_sp=self.query_species)

#         self._flat_search(qbuff, align)

#     def _flat_search(self, qbuff, align):
        
#         fam_results, hog_results, scores = self.search(qbuff, align)

#         # store results (+ self._query_id.nrows is when adding multiple fastas)
#         self._res_tab.append(list(zip(*[qbuff.ids[:].flatten(), fam_results, hog_results, scores])))

#         # self._res_tab.append(list(zip(*[qbuff.prot_offsets + self._query_id.nrows, fam_results, hog_results, cs_scores])))
#         self._res_tab.flush()

#         # store ids of qbuff
#         self._query_id.append(qbuff.ids)

#         # here maybe it would make sense to reset cache!
#         self.reset_cache()

#         # close and re-open in read mode
#         self.fs.close()
#         self.mode = 'r'
#         self.fs = tables.open_file(self.file, self.mode)

#     def search(self, qbuff, align=True):
        
#         @numba.njit
#         def get_cands(cands):
#             (ii, jj) = cands.nonzero()
#             xx = cands.flatten()
#             xx = xx[xx > 0] # - 1 --> this was when using specific entry nr instead if protein offset
#             return np.dstack((ii, jj, xx))[0]
        
#         # k-mer search
#         (cands, counts) = self.lookup(qbuff)
        
#         if align:
#             # load reference buffer
#             rbuff = SequenceBuffer(db=self.db)

#             # ordering n_top with alignments
#             scores_shr = sharedctypes.RawArray(ctypes.c_double, counts.size)

#             with mp.Pool(self.nthreads,
#                      initializer=worker_init,
#                      initargs=(self.matrix, self.gap_open, self.gap_extend, qbuff, rbuff, scores_shr, counts.shape)) as p:
#                 chunk_size = min(10000, 10*self.n_top)
#                 r = p.starmap_async(worker_align, get_cands(cands), chunksize=chunk_size)
#                 pbar = ['|', '/', '-', '\\']
#                 i = 0
#                 sys.stderr.write('\n')
#                 while True:
#                     try:
#                         r.get(timeout=1)
#                         sys.stderr.write('\rAligning {} [Done]\n'.format(pbar[i]))
#                         break
#                     except mp.TimeoutError:
#                         sys.stderr.write('\rAligning {}'.format(pbar[i]))
#                         i = (i + 1) % len(pbar)

#             scores = np.frombuffer(scores_shr, dtype=np.float64).reshape(counts.shape)
            
#             # map to hogs and families
#             hog_results = [self.prot2hog[cands[i][np.argmax(x)]] for i, x in enumerate(scores)]
#             fam_results = [self.hog2fam[x] for x in hog_results]
#             mss_scores = [max(x) for x in scores]
            
#         # k-mer only
#         else:
#             hog_results = [self.prot2hog[x[0]] for x in cands]
#             fam_results = [self.hog2fam[x] for x in hog_results]
#             mss_scores = [x[0] for x in counts]
        
#         return (fam_results, hog_results, mss_scores)

#     def lookup(self, qbuff):
#         return self._lookup(qbuff.buff, qbuff.idx, self.trans, self.table_buff, self.table_idx, self.ki.k, self.db._prot_tab.nrows, self.n_top, self.ki.alphabet.DIGITS_AA_LOOKUP)

#     @lazy_property
#     def _lookup(self):
#         def func(seqs, seqs_idx, trans, entries, idx, k, n, top_n, DIGITS_AA_LOOKUP):
#             results = np.zeros((len(seqs_idx)-1, top_n), dtype=np.uint32)
#             counts = np.zeros((len(seqs_idx)-1, top_n), dtype=np.uint16)

#             for zz in numba.prange(len(seqs_idx)-1):
#                 s = seqs[seqs_idx[zz]:int(seqs_idx[zz+1]-1)]
#                 res = np.zeros(n, dtype=np.uint16) # assume genes len <~2**16
#                 s_norm = DIGITS_AA_LOOKUP[s]
#                 r = np.zeros(len(s)-k+1, dtype=np.uint32) # max kmer 7  # added +1 because missing last k-mer
#                 for i in numba.prange(len(s)-k+1):  # added +1 because missing last k-mer
#                     # numba can't do np.dot with non-float
#                     for j in range(k):
#                         r[i] += trans[j] * s_norm[i+j]
#                 r1 = np.unique(r)

#                 for m in numba.prange(len(r1)):
#                     kmer = r1[m]
#                     x = idx[kmer:kmer+2]
#                     res[np.unique(entries[x[0]:x[1]])] += np.uint16(1)

#                 # For m best matches
#                 t = np.argsort(res)[::-1][:top_n]
#                 results[zz,:len(t)] = t
#                 counts[zz,:len(t)] = res[t] 

#             return results, counts

#         if not self.low_mem:
#             # Set nthreads, note: this only works before numba called first time!
#             numba.config.NUMBA_NUM_THREADS = self.nthreads
#             return numba.jit(func, parallel=True, nopython=True, nogil=True)
#         else:
#             return func


