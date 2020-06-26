import os
import numba
import tables
import numpy as np
from tqdm import tqdm

import ctypes
from ete3 import Tree
from PySAIS import sais
import multiprocessing as mp
from Bio import SeqIO, SearchIO
from multiprocessing import sharedctypes
from property_manager import lazy_property, cached_property

from .hierarchy import get_lca_hog_off
from .alphabets import Alphabet

'''
Questions:
 - can we parallelize the k-mer table computation (parallel=True makes it slower despite using more CPUs)
 - how implement a progress bar/status within numba?
 - why reset cache after building index in orlov?
 - why rewrite __enter__ and __exit__?

TO DO:
 - store reduced_alphabet variab inside hdf5
 - (option for filter based on other attributes than species. e.g. family, representatives ("sampling"), etc.)
'''

@numba.njit
def get_transform(k, DIGITS_AA):
    t = np.zeros(k, dtype=np.uint64)
    for i in numba.prange(k):
        t[i] = len(DIGITS_AA)**(k - (i + 1))
    return t

class Index():
    def __init__(self, db, name=None, path=None, k=6, reduced_alphabet=False, nthreads=1):
        
        assert db.mode == 'r', 'Database must be opened in read mode.'
        
        # load database object
        self.db = db

        # set index related features
        self.k = k
        self.alphabet = Alphabet(n=(21 if not reduced_alphabet else 13))
        
        # performance features
        self.nthreads = nthreads

        # filter based on species attributes
        self.sp_filter = np.full((len(self.db._sp_tab),), False)
        
        # hdf5 file
        self.name = "{}_{}".format(self.db.name, name if name else "k{}_{}".format(self.k, 'A13' if reduced_alphabet else 'A21'))
        self.path = path if path else self.db.path
        self.file = "{}{}.h5".format(self.path, self.name)

        if os.path.isfile(self.file):
            self.mode = 'r'
            self.ki = tables.open_file(self.file, self.mode)
        else:
            self.mode = 'w'
            self.ki = tables.open_file(self.file, self.mode)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.ki.close()
        
    def clean(self):
        '''
        close and remove hdf5 file
        '''
        self.__exit__()
        try:
            os.remove(self.file)
        except FileNotFoundError:
            print("{} already cleaned".format(self.file))
    
    ### same as in database class; easy access to data ###
    def _get_node_if_exist(self, node):
        if node in self.ki:
            return self.ki.get_node(node)

    @property
    def _table_idx(self):
        return self._get_node_if_exist('/TableIndex')

    @property
    def _table_buff(self):
        return self._get_node_if_exist('/TableBuffer')

    @property
    def _fam_count(self):
        return self._get_node_if_exist('/FamCount')

    @property
    def _hog_count(self):
        return self._get_node_if_exist('/HOGcount')

    # @property  
    # def _table_idx(self):
    #     if '/TableIndex' in self.ki:
    #         return self.ki.root.TableIndex
    #     else:
    #         return None

    # @property  
    # def _table_buff(self):
    #     if '/TableBuffer' in self.ki:
    #         return self.ki.root.TableBuffer
    #     else:
    #         return None
        
    # @property  
    # def _fam_count(self):
    #     if '/FamCount' in self.ki:
    #         return self.ki.root.FamCount
    #     else:
    #         return None

    # @property  
    # def _hog_count(self):
    #     if '/HOGcount' in self.ki:
    #         return self.ki.root.HOGcount
    #     else:
    #         return None

    ### main function to build the index ###
    def build_kmer_table(self):
        
        assert (self.mode in {'w', 'a'}), 'Index must be opened in write mode.'

        # build suffix array with option to translate the sequence buffer first
        sa = self._build_suffixarray(self.alphabet.translate(self.db._seq_buff[:]), len(self.db._prot_tab))

        # Set nthreads, note: this only works before numba called first time!
        numba.config.NUMBA_NUM_THREADS = self.nthreads

        self._build_kmer_table(sa)

    @staticmethod
    def _build_suffixarray(seqs, n):
        # Build suffix array
        sa = sais(seqs)
        sa[:n].sort()  # Sort delimiters by position
        return sa

    def _build_kmer_table(self, sa):

        assert (self.mode in {'w', 'a'}), 'Index must be opened in write mode.'

        @numba.njit(parallel=True)
        def _compute_mask_and_filter(sa, sa_mask, sa_filter, k, n, prot2spoff, prot2hogoff, sp_filter):
            '''
            1. compute a mapper between suffixes and HOGs
            2. simultaneously, compute suffix filter for species and suffixe < k
            '''
            for i in numba.prange(n):
                # leverages the sorted protein delimiters at the beggining of the sa to get the suffix offsets by protein
                s = (sa[i - 1] if i > 0 else -1) + 1
                e = (sa[i] + 1)
                sa_mask[s:e] = prot2hogoff[i]
                if sp_filter[prot2spoff[i]]:
                    sa_filter[s:e] = True
                else:
                    sa_filter[(e - k):e] = True

        @numba.njit
        def _same_kmer(seq_buff, sa, kmer, jj, k):
            kmer_jj = seq_buff[sa[jj]:(sa[jj]+k)].view(np.uint8)
            return np.all(np.equal(kmer, kmer_jj))

        @numba.njit
        def _compute_many_lca_hogs(hog_offsets, fam_offsets, hog_parents):
            '''
            compute lca hogs for a list of hogs and their families
            '''
            # as many lca hogs as unique families
            lca_hogs = np.zeros((np.unique(fam_offsets).size,), dtype=np.int64)
            
            # keep track of family, the corresponding hogs and the lca hog offset 
            curr_fam = fam_offsets[0]
            curr_hogs = list(hog_offsets[0:1])  # set the type of items in list to integers
            lca_off = 0

            for i in range(1, len(hog_offsets)):
                fam = fam_offsets[i]
                # wait to have all hogs of the family between computing the lca hog
                if fam == curr_fam:
                    curr_hogs.append(hog_offsets[i])
                else:
                    lca_hogs[lca_off] = get_lca_hog_off(curr_hogs, hog_parents)
                    curr_hogs = list(hog_offsets[i:i+1])
                    curr_fam = fam
                    lca_off += 1
           
            # last family
            lca_hogs[lca_off] = get_lca_hog_off(curr_hogs, hog_parents)
            return lca_hogs

        @numba.njit
        def _compute_kmer_table(sa, seq_buff, sa_mask, hog_fams, hog_parents, table_idx, table_buff, fam_kmer_counts, hog_kmer_counts, k, DIGITS_AA, DIGITS_AA_LOOKUP):

            ii = 0  # pointer to sa offset of kk
            kk = 0  # pointer to k-mer (offset in table_idx)
            ii_table_buff = 0 # pointer to offset in table_buff (~rows of k-mer table)
            trans = get_transform(k, DIGITS_AA)
            while ii < len(sa):
                if ii % 100000 == 0:
                    print(ii)

                ## compute the new k-mer (kk1)
                kmer = seq_buff[sa[ii]:(sa[ii]+k)].view(np.uint8)
                kk1 = 0
                for i in range(k):
                    kk1 += int(DIGITS_AA_LOOKUP[kmer[i]] * trans[i])

                ## find offset of new k-mer in sa (jj)
                # first in windows of 100s
                jj = min(ii + 100, len(sa))
                while ((jj < len(sa)) and _same_kmer(seq_buff, sa, kmer, jj, k)):
                    jj = min(jj + 100, len(sa))

                # then, refine with binary search
                lo = max(ii,(jj - 100) + 1)
                hi = jj
                while lo < hi:
                    m = int(np.floor((lo + hi) / 2))
                    if _same_kmer(seq_buff, sa, kmer, m, k):
                        lo = m + 1
                    else:
                        hi = m
                jj = lo

                ## compute LCA HOGs containing current k-mer (kk)
                # get the hog offsets for each suffix containing kk
                hog_offsets = np.unique(sa_mask[ii:jj])

                # get the corresponding fam offsets
                fam_offsets = hog_fams[hog_offsets]

                # compute the LCA hog offsets
                lca_hog_offsets = _compute_many_lca_hogs(hog_offsets, fam_offsets, hog_parents)

                # store kmer counts of fams and lca hogs
                fam_kmer_counts[np.unique(fam_offsets)] = fam_kmer_counts[np.unique(fam_offsets)] + 1
                hog_kmer_counts[lca_hog_offsets] = hog_kmer_counts[lca_hog_offsets] + 1

                ## store LCA HOGs in table buffer
                nr_lca_hog_offsets = len(lca_hog_offsets)
                table_buff[ii_table_buff : ii_table_buff + nr_lca_hog_offsets] = lca_hog_offsets
                
                ## store buffer offset in table index at offset corresponding to k-mer integer encoding
                table_idx[kk:kk1+1] = ii_table_buff

                ## find buffer offset of new k-mer in table index
                ii_table_buff += nr_lca_hog_offsets
                ii = jj
                kk = kk1 + 1

            # fill until the end
            table_idx[kk:] = ii_table_buff
            return ii_table_buff

        print(' - filter suffix array and compute its HOG mask')
        n = len(self.db._prot_tab)
        sa_mask = np.zeros(sa.shape, dtype=np.uint64)
        sa_filter = np.zeros(sa.shape, dtype=np.bool)

        _compute_mask_and_filter(sa, sa_mask, sa_filter, self.k, n, self.db._prot_tab.col('SpeOff'), self.db._prot_tab.col('HOGoff'), self.sp_filter)

        # before filtering the sa, reorder and reverse the suffix filter
        sa = sa[~sa_filter[sa]]

        # filter and reorder the mask according to this filtered sa
        sa_mask = sa_mask[sa]

        print(' - compute k-mer table')
        table_idx = np.zeros((len(self.alphabet.DIGITS_AA)**self.k + 1), dtype=np.uint64) 

        # initiate buffer of size sa_mask, which is maximum size if all suffixes are from different HOGs
        table_buff = np.zeros((len(sa_mask)), dtype=np.uint64)

        fam_kmer_counts = np.zeros(len(self.db._fam_tab), dtype=np.uint64)
        hog_kmer_counts = np.zeros(len(self.db._hog_tab), dtype=np.uint64)

        ii_table_buff = _compute_kmer_table(
            sa, self.db._seq_buff[:], sa_mask, self.db._hog_tab.col('FamOff'), self.db._hog_tab.col('ParentOff'), table_idx, table_buff,
            fam_kmer_counts, hog_kmer_counts, self.k, self.alphabet.DIGITS_AA, self.alphabet.DIGITS_AA_LOOKUP)

        # remove extra space
        table_buff =  table_buff[:ii_table_buff]

        print(' - write k-mer table')
        self.ki.create_carray('/', 'TableIndex', obj=table_idx, filters=self.db._compr)
        self.ki.create_carray('/', 'TableBuffer', obj=table_buff, filters=self.db._compr)
        self.ki.create_carray('/', 'FamCount', obj=fam_kmer_counts, filters=self.db._compr)  # for these, I can initialize them before...
        self.ki.create_carray('/', 'HOGcount', obj=hog_kmer_counts, filters=self.db._compr)

        # close and re-open in read mode
        self.ki.close()
        self.mode = 'r'
        self.ki = tables.open_file(self.file, self.mode)


class IndexValidation(Index):

    def __init__(self, db, path=None, k=6, reduced_alphabet=False, nthreads=1, stree_path=None, hidden_taxon=None):
        
        assert hidden_taxon, 'A hidden taxon must be defined'
        self.hidden_taxon = ''.join(hidden_taxon.split())
        name = 'wo_{}_k{}_{}'.format(self.hidden_taxon, k, 'A13' if reduced_alphabet else 'A21')

        super().__init__(db, name, path, k, reduced_alphabet, nthreads)

        # hide all species of a given taxon
        hidden_taxa, hidden_species = self.get_clade_specific_taxa(stree_path, hidden_taxon)

        self.hidden_taxa = np.array(list(hidden_taxa))
        self.hidden_species = np.array(list(hidden_species))
        
        # find their offsets in sp_tab
        hidden_species_offsets = np.searchsorted(self.db._sp_tab.col('ID'), self.hidden_species)
        
        # build the species filter
        self.sp_filter = np.zeros((self.db._sp_tab.nrows,), dtype=np.bool)
        self.sp_filter[hidden_species_offsets] = True

    # tax and HOG filters for PlacementValidation
    @cached_property
    def tax_filter(self):
        '''
        boolean filter for the taxonomy (db._tax_tab) given a list of hidden taxa
        '''
        hidden_taxa_offsets = np.searchsorted(self.db._tax_tab.col('ID'), self.hidden_taxa)  # Modified hidden_species to hidden_taxa
        tax_filter = np.zeros((self.db._tax_tab.nrows,), dtype=np.bool)
        tax_filter[hidden_taxa_offsets] = True
        return tax_filter

    @cached_property
    def hog_filter_lca(self):
        '''
        boolean filter for HOGs given the LCA taxon of their members (e.g. to filter out query-specific HOG)
        '''
        return self.tax_filter[self.db._hog_tab.col('LCAtaxOff')]

    # @cached_property
    # def hog_filter_lcasis(self):
    #     '''
    #     idea was to filter out HOGs whose duplication should not be known given the hidden species
    #     however (almost?) no cases and error in implementation (issue when outgroup of other HOGs)
    #     '''
    #     hog_tab = self.db._hog_tab[:]
    #     chog_buff = self.db._chog_arr[:]
    #     hog_filter_lcasis = np.zeros(self.hog_filter_lca.size, dtype=np.bool)

    #     for hog_off in range(self.hog_filter_lca.size):
    #         if hog_tab['ParentOff'][hog_off] != -1:
                
    #             # sister HOGs at same taxon
    #             hog_tax = hog_tab['TaxOff'][hog_off]
    #             sis_hogs = hierarchy.get_sister_hogs(hog_off, hog_tab, chog_buff)
    #             sis_hogs = [h for h in sis_hogs if hog_tab['TaxOff'][h] == hog_tax]
                
    #             # if all sisters are hidden, hide mask
    #             if (self.hog_filter_lca[sis_hogs] == True).all():
    #                 hog_filter_lcasis[hog_off] = True
        
    #     del hog_tab, chog_buff
    #     return hog_filter_lcasis

    @staticmethod
    def get_clade_specific_taxa(stree_path, root_taxon):
        '''
        gather all taxa and species specific to a given clade
        '''
        stree = Tree(stree_path, format=1, quoted_node_names=True)

        pruned_stree = [x for x in stree.traverse() if x.name == root_taxon][0]

        taxa = set()
        species = set()

        for tl in pruned_stree.traverse():
            taxon = tl.name.encode('ascii')
            taxa.add(taxon)
            if tl.is_leaf():
                species.add(taxon)

        return np.array(sorted(taxa)), species


class SequenceBuffer(object):
    '''
    to load sequences from db or files
    adapted from alex
    '''
    def __init__(self, seqs=None, ids=None, db=None, fasta_file=None):
        self.prot_off = 0
        if seqs is not None:
            self.add_seqs(*seqs)
            self.ids = (np.array(ids) if ids else np.array(range(len(seqs))))
        elif db is not None:
            self._prot_tab = db._prot_tab
            self._seq_buff = db._seq_buff
            self.load_from_db()
        elif fasta_file is not None:
            seqs = self.parse_fasta(fasta_file)
            self.add_seqs(*seqs)
        else:
            raise ValueError('need to pass array of seqs or db to load from to SequenceBuffer')

    def __getstate__():
        return (self.prot_nr, self.n, self.buff_shr, self.buff_idx_shr)

    def __setstate__(state):
        (self.prot_nr, self.n, self.buff_shr, self.buff_idx_shr) = state

    def parse_fasta(self, fasta_file):
        ids = []
        seqs = []
        for rec in SeqIO.parse(fasta_file, 'fasta'):
            ids.append(rec.id)
            seqs.append(str(rec.seq))
        self.ids = np.array(ids)
        return seqs

    def add_seqs(self, *seqs):
        self.prot_nr = len(seqs)
        self.n = self.prot_nr + sum(len(s) for s in seqs)

        self.buff_shr = sharedctypes.RawArray(ctypes.c_uint8, self.n)
        self.buff[:] = np.frombuffer((' '.join(seqs) + ' ').encode('ascii'), dtype=np.uint8)

        self.buff_idx_shr = sharedctypes.RawArray(ctypes.c_uint64, self.prot_nr + 1)
        for i in range(len(seqs)):
            self.idx[i+1] = len(seqs[i]) + 1 + self.idx[i]

    def load_from_db(self):
        self.prot_nr = len(self._prot_tab)
        self.n = len(self._seq_buff)

        self.buff_shr = sharedctypes.RawArray(ctypes.c_uint8, self.n)
        self.buff[:] = self._seq_buff[:].view(np.uint8)

        self.buff_idx_shr = sharedctypes.RawArray(ctypes.c_uint64, self.prot_nr + 1)
        #self.idx[:-1] = self._prot_tab.cols.SeqOff[:]
        self.idx[:-1] = self._prot_tab[:]['SeqOff']
        self.idx[-1] = self.n

        # store offsets as ids
        self.ids = np.arange(self.prot_off, self.prot_off + self.prot_nr, dtype=np.uint64)[:,None]

    @lazy_property
    def buff(self):
        return np.frombuffer(self.buff_shr, dtype=np.uint8).reshape(self.n)

    @lazy_property
    def idx(self):
        return np.frombuffer(self.buff_idx_shr, dtype=np.uint64).reshape(self.prot_nr + 1)

    # @lazy_property
    # def prot_offsets(self):
    #     return np.arange(self.prot_off, self.prot_off + self.prot_nr, dtype=np.uint64)

    def __getitem__(self, i):
        s = int(self.idx[i])
        e = int(self.idx[i+1]-1)
        return self.buff[s:e].tobytes().decode('ascii')


class QuerySequenceBuffer(SequenceBuffer):
    '''
    get a sliced version of the sequence buffer object for a given species in db
    TO DO: put an __super__
    '''    
    def __init__(self, db, query_sp):
        self.query_sp = query_sp if isinstance(query_sp, bytes) else query_sp.encode('ascii')
        self.set_query_prot_tab(db)
        self.filter_query_seq_buff(db)
        self.load_from_db()
        
    def set_query_prot_tab(self, db): 
        sp_off = np.searchsorted(db._sp_tab.col('ID'), self.query_sp)
        sp_ent = db._sp_tab[sp_off]
        self.prot_off = sp_ent['ProtOff']
        self._prot_tab = db._prot_tab[self.prot_off: self.prot_off + sp_ent['ProtNum']]
        
    def filter_query_seq_buff(self, db):
        self._seq_buff = db._seq_buff[self._prot_tab[0]['SeqOff']: self._prot_tab[-1]['SeqOff'] + self._prot_tab[-1]['SeqLen']]
        # initialize sequence buffer offset
        self._prot_tab['SeqOff'] -= db._prot_tab[self.prot_off]['SeqOff']

# END of OMAmer
################################################################################################################################################
# baseline classes mimicking IndexValidation structure

class DIAMONDindexValidation():
    '''
    to export reference fasta for BLAST and DIAMOND and mimic IndexValidation

    TO DO:
     - wrap DIAMOND process
    '''
    def __init__(self, db, stree_path=None, hidden_taxon=None):
        
        assert db.mode == 'r', 'Database must be opened in read mode.'
        self.mode = 'r'

        # load database object
        self.db = db

        assert hidden_taxon, 'A hidden taxon must be defined'
        self.hidden_taxon = ''.join(hidden_taxon.split())
        
        # name
        name = 'wo_{}_diamond'.format(self.hidden_taxon)
        self.name = "{}_{}".format(self.db.name, name)
        self.path = "{}{}/".format(self.db.path, self.name)

        # hide all species of a given taxon
        hidden_taxa, hidden_species = IndexValidation.get_clade_specific_taxa(stree_path, hidden_taxon)

        self.hidden_taxa = np.array(list(hidden_taxa))
        self.hidden_species = np.array(list(hidden_species))

        # find their offsets in sp_tab
        hidden_species_offsets = np.searchsorted(self.db._sp_tab.col('ID'), self.hidden_species)
        
        # build the species filter
        self.sp_filter = np.zeros((self.db._sp_tab.nrows,), dtype=np.bool)
        self.sp_filter[hidden_species_offsets] = True

    # tax and HOG filters for PlacementValidation
    @cached_property
    def tax_filter(self):
        '''
        boolean filter for the taxonomy (db._tax_tab) given a list of hidden taxa
        '''
        hidden_taxa_offsets = np.searchsorted(self.db._tax_tab.col('ID'), self.hidden_taxa)  # Modified hidden_species to hidden_taxa
        tax_filter = np.zeros((self.db._tax_tab.nrows,), dtype=np.bool)
        tax_filter[hidden_taxa_offsets] = True
        return tax_filter

    @cached_property
    def hog_filter_lca(self):
        '''
        boolean filter for HOGs given the LCA taxon of their members (e.g. to filter out query-specific HOG)
        '''
        return self.tax_filter[self.db._hog_tab.col('LCAtaxOff')]

    def export_reference_fasta(self):

        ref_sp = self.db._sp_tab.col('ID')[~self.sp_filter]
        chunk_size = self.db._prot_tab.nrows
        fasta_path = '{}reference/'.format(self.path)
        if not os.path.exists(fasta_path):
            os.makedirs(fasta_path)

        self.export_species_fasta(self.db, ref_sp, chunk_size, fasta_path)

    @staticmethod
    def export_species_fasta(db, ref_sp, chunk_size, fasta_path):

        i = 1
        j = 0

        fasta = open("{}{}.fa".format(fasta_path, j), 'w')

        for sp in tqdm(ref_sp):
            sb = QuerySequenceBuffer(db, sp)
            
            for idx, seq in enumerate(sb):
                
                # new file
                if i % chunk_size == 0:
                    fasta.close()
                    fasta = open("{}{}.fa".format(fasta_path, j), 'w')
                    j += 1
                
                fasta.write(">{}\n{}\n".format(sb.ids[idx][0], seq))
                i += 1
                
        fasta.close()


class SWindexValidation(DIAMONDindexValidation):
    '''
    to mimic IndexValidation for SW data
    '''
    def __init__(self, db, stree_path=None, hidden_taxon=None):
        
        super().__init__(db, stree_path, hidden_taxon)
        
        # just new name
        name = 'wo_{}_SW'.format(self.hidden_taxon)
        self.name = "{}_{}".format(self.db.name, name)
        self.path = self.db.path


class OrlovIndexValidation():

    def __init__(self, db, path=None, k=6, reduced_alphabet=False, nthreads=1, stree_path=None, hidden_taxon=None):

        assert db.mode == 'r', 'Database must be opened in read mode.'
        assert hidden_taxon, 'A hidden taxon must be defined'
        self.hidden_taxon = ''.join(hidden_taxon.split())

        # load database object
        self.db = db

        # set index related features
        self.k = k
        self.alphabet = Alphabet(n=(21 if not reduced_alphabet else 13))

        # performance features
        self.nthreads = nthreads

        # hide all species of a given taxon
        hidden_taxa, hidden_species = IndexValidation.get_clade_specific_taxa(stree_path, hidden_taxon)

        self.hidden_taxa = np.array(list(hidden_taxa))
        self.hidden_species = np.array(list(hidden_species))
        
        # find their offsets in sp_tab
        hidden_species_offsets = np.searchsorted(self.db._sp_tab.col('ID'), self.hidden_species)
        
        # build the species filter
        self.sp_filter = np.zeros((self.db._sp_tab.nrows,), dtype=np.bool)
        self.sp_filter[hidden_species_offsets] = True

        # hdf5 file
        self.name = '{}_wo_{}_orlov_k{}_{}'.format(self.db.name, self.hidden_taxon, k, 'A13' if reduced_alphabet else 'A21')
        self.path = path if path else self.db.path
        self.file = "{}{}.h5".format(self.path, self.name)

        if os.path.isfile(self.file):
            self.mode = 'r'
            self.ki = tables.open_file(self.file, self.mode)
        else:
            self.mode = 'w'
            self.ki = tables.open_file(self.file, self.mode)

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.ki.close()
        
    def clean(self):
        '''
        close and remove hdf5 file
        '''
        self.__exit__()
        try:
            os.remove(self.file)
        except FileNotFoundError:
            print("{} already cleaned".format(self.file))

    # tax and HOG filters for PlacementValidation
    @cached_property
    def tax_filter(self):
        '''
        boolean filter for the taxonomy (db._tax_tab) given a list of hidden taxa
        '''
        hidden_taxa_offsets = np.searchsorted(self.db._tax_tab.col('ID'), self.hidden_taxa)  # Modified hidden_species to hidden_taxa
        tax_filter = np.zeros((self.db._tax_tab.nrows,), dtype=np.bool)
        tax_filter[hidden_taxa_offsets] = True
        return tax_filter

    @cached_property
    def hog_filter_lca(self):
        '''
        boolean filter for HOGs given the LCA taxon of their members (e.g. to filter out query-specific HOG)
        '''
        return self.tax_filter[self.db._hog_tab.col('LCAtaxOff')]

    ### same as in database class; easy access to data ###
    @property  
    def _table_idx(self):
        if '/TableIndex' in self.ki:
            return self.ki.root.TableIndex
        else:
            return None

    @property  
    def _table_buff(self):
        if '/TableBuffer' in self.ki:
            return self.ki.root.TableBuffer
        else:
            return None

    ### main function to build the index ###
    def build_kmer_table(self):
        
        assert (self.mode in {'w', 'a'}), 'Index must be opened in write mode.'

        # build suffix array with option to translate the sequence buffer first
        sa = Index._build_suffixarray(self.alphabet.translate(self.db._seq_buff[:]), len(self.db._prot_tab))

        # Set nthreads, note: this only works before numba called first time!
        numba.config.NUMBA_NUM_THREADS = self.nthreads

        self._build_kmer_table(sa)

    def _build_kmer_table(self, sa):

        assert (self.mode in {'w', 'a'}), 'Index must be opened in write mode.'

        @numba.njit(parallel=True)
        def compute_index_mask(sa, idx, mask, n, k):
            for i in numba.prange(n):
                s = (sa[i - 1] if i > 0 else -1) + 1
                e = (sa[i] + 1)
                idx[s:e] = i + 1  # Map here for OG, HOG, etc..
                mask[(e - k):e] = True
        
        @numba.njit
        def _same_kmer(seq_buff, sa, kmer, jj, k):
            '''
            would be nice to have this outside
            '''
            kmer_jj = seq_buff[sa[jj]:(sa[jj]+k)].view(np.uint8)
            return np.all(np.equal(kmer, kmer_jj))

        @numba.njit
        def _compute_kmer_index(sa, seq_buff, idx_off, n, k, DIGITS_AA, DIGITS_AA_LOOKUP):
            ii = 0
            kk = 0
            trans = get_transform(k, DIGITS_AA)
            while ii < len(sa):
                kmer = seq_buff[sa[ii]:(sa[ii]+k)].view(np.uint8)

                #kk1 = int(np.dot(DIGITS_AA_LOOKUP[kmer], trans))
                # Numba only supports np.dot for double / complex
                kk1 = 0
                for i in range(k):
                    kk1 += int(DIGITS_AA_LOOKUP[kmer[i]] * trans[i])
                idx_off[kk:kk1+1] = idx_off[kk]
                kk = kk1

                # Find RHS of kmer in sa
                jj = min(ii + 100, len(sa))
                while ((jj < len(sa)) and _same_kmer(seq_buff, sa, kmer, jj, k)):
                    jj = min(jj + 100, len(sa))

                # Find RHS using binary search
                lo = max(ii,(jj - 100) + 1)
                hi = jj
                while lo < hi:
                    m = int(np.floor((lo + hi) / 2))
                    if _same_kmer(seq_buff, sa, kmer, m, k):
                        lo = m + 1
                    else:
                        hi = m
                jj = lo
                idx_off[kk+1] = jj

                # New start
                ii = jj
                kk += 1

            # Pad to end -> ->
            idx_off[kk:] = idx_off[kk] #-1] this -1 removed the last k-mer
        
        # Build kmer index, using suffix array
        n = len(self.db._prot_tab)

        print(' - creating mask / idx')
        dtype = np.uint32 if n < np.iinfo(np.uint32).max else np.uint64
        idx = np.zeros(sa.shape, dtype=dtype)
        mask = np.zeros(sa.shape, dtype=np.bool)
        compute_index_mask(sa, idx, mask, n, self.k)
        sa = sa[~mask[sa]]
        idx = idx[sa]

        print(' - computing kmer index')
        idx_off = np.zeros((len(self.alphabet.DIGITS_AA)**self.k + 1), dtype=np.uint64)
        _compute_kmer_index(sa, self.db._seq_buff[:], idx_off, n, self.k, self.alphabet.DIGITS_AA, self.alphabet.DIGITS_AA_LOOKUP)

        print(' - write kmer index')
        self.ki.create_carray('/', 'TableIndex', obj=idx_off, filters=self.db._compr)
        self.ki.create_carray('/', 'TableBuffer', obj=idx, filters=self.db._compr)

        # close and re-open in read mode
        self.ki.close()
        self.mode = 'r'
        self.ki = tables.open_file(self.file, self.mode)

# class Index():
#     def __init__(self, db, name=None, path=None, idx_type='lca', k=6, nthreads=None):
        
#         assert db.mode == 'r', 'Database must be opened in read mode.'
        
#         # load database object
#         self.db = db

#         # set index related features
#         self.idx_type = idx_type
#         self.k = k
#         self.DIGITS_AA = DIGITS_AA
#         self.DIGITS_AA_LOOKUP = DIGITS_AA_LOOKUP
        
    #     # performance features
    #     self.nthreads = nthreads if nthreads is not None else os.cpu_count()

    #     # filter based on species attributes
    #     self.sp_filter = np.full((len(self.db._sp_tab),), False)
        
    #     # hdf5 file
    #     self.name = "{}_{}".format(self.db.name, name if name else "{}_k{}".format(self.idx_type, self.k))
    #     self.path = path if path else self.db.path
    #     self.file = "{}{}.h5".format(self.path, self.name)

    #     if os.path.isfile(self.file):
    #         self.mode = 'r'
    #         self.ki = tables.open_file(self.file, self.mode)
    #     else:
    #         self.mode = 'w'
    #         self.ki = tables.open_file(self.file, self.mode)
    
    # def __enter__(self):
    #     return self
    
    # def __exit__(self, *_):
    #     self.ki.close()
        
    # def clean(self):
    #     '''
    #     close and remove hdf5 file
    #     '''
    #     self.__exit__()
    #     try:
    #         os.remove(self.file)
    #     except FileNotFoundError:
    #         print("{} already cleaned".format(self.file))
    
    # ### same as in database class; easy access to data ###
    # @property  
    # def _idx_arr(self):
    #     if '/IndexArray' in self.ki:
    #         return self.ki.root.IndexArray
    #     else:
    #         return None

    # @property  
    # def _idx_off(self):
    #     if '/IndexOffset' in self.ki:
    #         return self.ki.root.IndexOffset
    #     else:
    #         return None
        
    # @property  
    # def _fam_count(self):
    #     if '/FamCount' in self.ki:
    #         return self.ki.root.FamCount
    #     else:
    #         return None

    # @property  
    # def _hog_count(self):
    #     if '/HOGcount' in self.ki:
    #         return self.ki.root.HOGcount
    #     else:
    #         return None
    
    # ### main function to build the index ###
    # def build_index(self):
        
    #     assert (self.mode in {'w', 'a'}), 'Index must be opened in write mode.'

    #     # Set nthreads, note: this only works before numba called first time!
    #     numba.config.NUMBA_NUM_THREADS = self.nthreads

    #     if self.idx_type == 'cs':
    #         self.build_protein_index()
    #     elif self.idx_type == 'lca':
    #         self.build_family_index()
    #     else:
    #         raise ValueError('Index type does not exist')

    #     # reset cache of search

    # ### shared functions among index types ###
    # @staticmethod
    # def filter_sa_and_compute_sa_mask(sa, idx_type, k, n, dtype, prot2spoff, prot2hogoff, sp_filter):

    #     @njit
    #     def _compute_mask_and_filter(sa, sa_mask, sa_filter, idx_type, k, n, prot2spoff, prot2hogoff, sp_filter):
    #         '''
    #         1. compute a mapping between suffix offsets and protein offsets
    #         2. simultaneously, compute a protein filter for species and suffixe < k
    #         '''
    #         for i in prange(n):
    #             # leverages the sorted protein delimiters at the beggining of the sa to get the suffix offsets by protein
    #             s = (sa[i - 1] if i > 0 else -1) + 1
    #             e = (sa[i] + 1)
    #             sa_mask[s:e] = i if idx_type == 'cs' else prot2hogoff[i]
    #             if sp_filter[prot2spoff[i]]:
    #                 sa_filter[s:e] = True
    #             else:
    #                 sa_filter[(e - k):e] = True

    #     sa_mask = np.zeros(sa.shape, dtype=dtype)
    #     sa_filter = np.zeros(sa.shape, dtype=np.bool)

    #     _compute_mask_and_filter(sa, sa_mask, sa_filter, idx_type, k, n, prot2spoff, prot2hogoff, sp_filter)

    #     # before filtering the sa, reorder and reverse the suffix filter
    #     sa = sa[~sa_filter[sa]]

    #     # filter and reorder the mask according to this filtered sa
    #     sa_mask = sa_mask[sa]

    #     return sa, sa_mask
    
    # ### specific functions to build family index ###
    # @staticmethod
    # @njit
    # def _get_root_leaf_hog_offsets(hog_off, hog_parents):
    #     '''
    #     leverages parent pointers to gather hogs (offset from HOG table) until root
    #     '''
    #     leaf_root = [hog_off]
    #     parent_hog = hog_parents[hog_off]
    #     while parent_hog != -1:
    #         leaf_root.append(parent_hog)
    #         parent_hog = hog_parents[parent_hog]
    #     return np.array(leaf_root[::-1], dtype=np.uint64)

    # def build_family_index(self):

    #     assert (self.mode in {'w', 'a'}), 'Index must be opened in write mode.'   

    #     @njit
    #     def _same_kmer(seq_buff, sa, kmer, jj, k):
    #         kmer_jj = seq_buff[sa[jj]:(sa[jj]+k)].view(np.uint8)
    #         return np.all(np.equal(kmer, kmer_jj))

    #     @njit
    #     def _get_lca_hog_off(hog_offsets, hog_parents, fun=self._get_root_leaf_hog_offsets):
    #         '''
    #         compute the last common ancestor (lca) within a list of hog offsets from the same family
    #         '''
    #         hog_nr = len(hog_offsets)
            
    #         # lca of one hog is itself
    #         if hog_nr == 1:
    #             return hog_offsets[0]
            
    #         else:
    #             # gather every root-to-leaf paths and keep track of the shortest one
    #             root_leaf_paths = []
    #             min_path = np.iinfo(np.int64).max
    #             for x in hog_offsets:
    #                 root_leaf = fun(x, hog_parents)
    #                 root_leaf_paths.append(root_leaf)
    #                 if len(root_leaf) < min_path: 
    #                     min_path = len(root_leaf)

    #             # if one hog is root, lca is root
    #             if min_path == 1:
    #                 return root_leaf_paths[0][0]

    #             else:
    #                 # hogs from root to leaves and stop at min_path
    #                 mat = np.zeros((hog_nr, min_path), dtype=np.int64)
    #                 for i in range(hog_nr):
    #                     mat[i] = root_leaf_paths[i][:min_path]
    #                 matT = mat.T

    #                 # the lca hog is the one before hogs start diverging
    #                 i = 0
    #                 while i < min_path and np.unique(matT[i]).size == 1:
    #                     i += 1
    #                 return matT[i - 1][0]

    #     @njit
    #     def _compute_many_lca_hogs(hog_offsets, fam_offsets, hog_parents):
    #         '''
    #         compute lca hogs for a list of hogs and their families
    #         '''
    #         # as many lca hogs as unique families
    #         lca_hogs = np.zeros((np.unique(fam_offsets).size,), dtype=np.int64)
            
    #         # keep track of family, the corresponding hogs and the lca hog offset 
    #         curr_fam = fam_offsets[0]
    #         curr_hogs = list(hog_offsets[0:1])  # set the type of items in list to integers
    #         lca_off = 0

    #         for i in range(1, len(hog_offsets)):
    #             fam = fam_offsets[i]
    #             # wait to have all hogs of the family between computing the lca hog
    #             if fam == curr_fam:
    #                 curr_hogs.append(hog_offsets[i])
    #             else:
    #                 lca_hogs[lca_off] = _get_lca_hog_off(curr_hogs, hog_parents)
    #                 curr_hogs = list(hog_offsets[i:i+1])
    #                 curr_fam = fam
    #                 lca_off += 1
           
    #         # last family
    #         lca_hogs[lca_off] = _get_lca_hog_off(curr_hogs, hog_parents)
    #         return lca_hogs

    #     @njit
    #     def _compute_family_index(sa, seq_buff, sa_mask, hog_fams, hog_parents, lca_idx_off, fam_kmer_counts, hog_kmer_counts, k, DIGITS_AA, DIGITS_AA_LOOKUP):

    #         # initiate idx of size sa_mask, which is maximum size if all suffixes are from different HOGs
    #         lca_idx = np.zeros((len(sa_mask)), dtype=np.uint64)

    #         ii = 0  # pointer to sa offset of kk
    #         kk = 0  # pointer to k-mer = offset in lca_idx_off
    #         ii_lca_idx = 0 # pointer to offset in lca_idx (~rows of k-mer index)
    #         trans = get_transform(k, DIGITS_AA)
    #         while ii < len(sa):

    #             ## compute the new k-mer (kk1)
    #             kmer = seq_buff[sa[ii]:(sa[ii]+k)].view(np.uint8)
    #             kk1 = 0
    #             for i in range(k):
    #                 kk1 += int(DIGITS_AA_LOOKUP[kmer[i]] * trans[i])

    #             ## find offset of new k-mer in sa (jj)
    #             # first in windows of 100s
    #             jj = min(ii + 100, len(sa))
    #             while ((jj < len(sa)) and _same_kmer(seq_buff, sa, kmer, jj, k)):
    #                 jj = min(jj + 100, len(sa))

    #             # then, refine with binary search
    #             lo = max(ii,(jj - 100) + 1)
    #             hi = jj
    #             while lo < hi:
    #                 m = int(np.floor((lo + hi) / 2))
    #                 if _same_kmer(seq_buff, sa, kmer, m, k):
    #                     lo = m + 1
    #                 else:
    #                     hi = m
    #             jj = lo

    #             ## compute LCA HOGs containing current k-mer (kk)
    #             # get the hog offsets for each suffix containing kk
    #             hog_offsets = np.unique(sa_mask[ii:jj])

    #             # get the corresponding fam offsets
    #             fam_offsets = hog_fams[hog_offsets]
                
    #             # compute the LCA hog offsets
    #             lca_hog_offsets = _compute_many_lca_hogs(hog_offsets, fam_offsets, hog_parents)
                
    #             # store kmer counts of fams and lca hogs
    #             fam_kmer_counts[np.unique(fam_offsets)] = fam_kmer_counts[np.unique(fam_offsets)] + 1
    #             hog_kmer_counts[lca_hog_offsets] = hog_kmer_counts[lca_hog_offsets] + 1

    #             ## extend the k-mer index "rows" (lca_idx) and "columns" (lca_idx_off)
    #             nr_lca_hog_offsets = len(lca_hog_offsets)
    #             lca_idx[ii_lca_idx : ii_lca_idx + nr_lca_hog_offsets] = lca_hog_offsets
    #             lca_idx_off[kk:kk1+1] = ii_lca_idx

    #             ## find offset of new k-mer in lca_idx and restart
    #             ii_lca_idx += nr_lca_hog_offsets
    #             ii = jj
    #             kk = kk1 + 1

    #         # fill until the end
    #         lca_idx_off[kk:] = ii_lca_idx
            
    #         # remove extra space
    #         return lca_idx[:ii_lca_idx]

    #     n = len(self.db._prot_tab)
    #     dtype = np.uint32 if n < np.iinfo(np.uint32).max else np.uint64

    #     print(' - filter suffix array and compute its family mask')
    #     sa, sa_mask = self.filter_sa_and_compute_sa_mask(self.db._suff_arr[:], self.idx_type, self.k, n,
    #         dtype, self.db._prot_tab.col('SpeOff'), self.db._prot_tab.col('HOGoff'), self.sp_filter)

    #     lca_idx_off = np.zeros((len(self.DIGITS_AA)**self.k + 1), dtype=np.uint64)    
    #     fam_kmer_counts = np.zeros(len(self.db._fam_tab), dtype=np.uint64)
    #     hog_kmer_counts = np.zeros(len(self.db._hog_tab), dtype=np.uint64)

    #     print(' - compute family index')
    #     lca_idx = _compute_family_index(sa, self.db._seq_buff[:], sa_mask, self.db._hog_tab.col('FamOff'),
    #         self.db._hog_tab.col('ParentOff'), lca_idx_off, fam_kmer_counts, hog_kmer_counts, self.k, self.DIGITS_AA, self.DIGITS_AA_LOOKUP)

    #     print(' - write family index')
    #     self.ki.create_carray('/', 'IndexOffset', obj=lca_idx_off, filters=self.db._compr)
    #     self.ki.create_carray('/', 'IndexArray', obj=lca_idx, filters=self.db._compr)
    #     self.ki.create_carray('/', 'FamCount', obj=fam_kmer_counts, filters=self.db._compr)  # for these, I can initialize them before...
    #     self.ki.create_carray('/', 'HOGcount', obj=hog_kmer_counts, filters=self.db._compr)

    #     # close and re-open in read mode
    #     self.ki.close()
    #     self.mode = 'r'
    #     self.ki = tables.open_file(self.file, self.mode)

    # ### specific functions to build protein index ### 
    # def build_protein_index(self):

    #     assert (self.mode in {'w', 'a'}), 'Index must be opened in write mode.'    
        
    #     @njit
    #     def _same_kmer(seq_buff, sa, kmer, jj, k):
    #         '''
    #         would be nice to have this outside
    #         '''
    #         kmer_jj = seq_buff[sa[jj]:(sa[jj]+k)].view(np.uint8)
    #         return np.all(np.equal(kmer, kmer_jj))

    #     @njit
    #     def _compute_protein_index(sa, seq_buff, idx_off, n, k, DIGITS_AA, DIGITS_AA_LOOKUP):
    #         ii = 0
    #         kk = 0
    #         trans = get_transform(k, DIGITS_AA)
    #         while ii < len(sa):
    #             kmer = seq_buff[sa[ii]:(sa[ii]+k)].view(np.uint8)

    #             #kk1 = int(np.dot(DIGITS_AA_LOOKUP[kmer], trans))
    #             # Numba only supports np.dot for double / complex
    #             kk1 = 0
    #             for i in range(k):
    #                 kk1 += int(DIGITS_AA_LOOKUP[kmer[i]] * trans[i])
    #             idx_off[kk:kk1+1] = idx_off[kk]
    #             kk = kk1

    #             # Find RHS of kmer in sa
    #             jj = min(ii + 100, len(sa))
    #             while ((jj < len(sa)) and _same_kmer(seq_buff, sa, kmer, jj, k)):
    #                 jj = min(jj + 100, len(sa))

    #             # Find RHS using binary search
    #             lo = max(ii,(jj - 100) + 1)
    #             hi = jj
    #             while lo < hi:
    #                 m = int(np.floor((lo + hi) / 2))
    #                 if _same_kmer(seq_buff, sa, kmer, m, k):
    #                     lo = m + 1
    #                 else:
    #                     hi = m
    #             jj = lo
    #             idx_off[kk+1] = jj

    #             # New start
    #             ii = jj
    #             kk += 1

    #         # Pad to end -> ->
    #         idx_off[kk:] = idx_off[kk] #-1] this -1 removed the last k-mer
        
    #     n = len(self.db._prot_tab)
    #     dtype = np.uint32 if n < np.iinfo(np.uint32).max else np.uint64
        
    #     print(' - filter suffix array and compute its protein mask')
    #     sa, sa_mask = self.filter_sa_and_compute_sa_mask(self.db._suff_arr[:], self.idx_type, self.k, n,
    #         dtype, self.db._prot_tab.col('SpeOff'), self.db._prot_tab.col('HOGoff'), self.sp_filter)
        
    #     print(' - compute protein index')
    #     idx_off = np.zeros((len(self.DIGITS_AA)**self.k + 1), dtype=np.uint64)
    #     _compute_protein_index(sa, self.db._seq_buff[:], idx_off, n, self.k, self.DIGITS_AA, self.DIGITS_AA_LOOKUP)
        
    #     print(' - write protein index')
    #     self.ki.create_carray('/', 'IndexOffset', obj=idx_off, filters=self.db._compr)
    #     self.ki.create_carray('/', 'IndexArray', obj=sa_mask, filters=self.db._compr)
        
    #     # close and re-open in read mode
    #     self.ki.close()
    #     self.mode = 'r'
    #     self.ki = tables.open_file(self.file, self.mode)
    

# class IndexValidation(Index):

#     def __init__(self, db, name=None, path=None, idx_type='lca', k=6, nthreads=None, stree_path=None, hidden_taxon=None):
        
#         assert hidden_taxon, 'A hidden taxon must be defined'
#         self.hidden_taxon = ''.join(hidden_taxon.split())
#         name = 'wo_{}_{}'.format(self.hidden_taxon, idx_type)

#         super().__init__(db, name, path, idx_type, k, nthreads)

#         # hide all species of a given taxon
#         hidden_taxa, hidden_species = self.get_clade_specific_taxa(stree_path, hidden_taxon)

#         self.hidden_taxa = np.array(list(hidden_taxa))
#         self.hidden_species = np.array(list(hidden_species))
        
#         # find their offsets in sp_tab
#         hidden_species_offsets = np.searchsorted(self.db._sp_tab.col('ID'), self.hidden_species)
        
#         # build the species filter
#         self.sp_filter = np.zeros((self.db._sp_tab.nrows,), dtype=np.bool)
#         self.sp_filter[hidden_species_offsets] = True

#     # tax and HOG filters for Validation
#     @cached_property
#     def tax_filter(self):
#         hidden_taxa_offsets = np.searchsorted(self.db._tax_tab.col('ID'), self.hidden_species)
#         tax_filter = np.zeros((self.db._tax_tab.nrows,), dtype=np.bool)
#         tax_filter[hidden_taxa_offsets] = True
#         return tax_filter

#     @cached_property
#     def hog_filter_lca(self):
#         return self.tax_filter[self.db._hog_tab.col('LCAtaxOff')]

#     @cached_property
#     def hog_filter_lcasis(self):
#         hog_tab = self.db._hog_tab[:]
#         chog_buff = self.db._chog_arr[:]
#         hog_filter_lcasis = np.zeros(self.hog_filter_lca.size, dtype=np.bool)

#         for hog_off in range(self.hog_filter_lca.size):
#             if hog_tab['ParentOff'][hog_off] != -1:
                
#                 # sister HOGs at same taxon
#                 hog_tax = hog_tab['TaxOff'][hog_off]
#                 sis_hogs = hierarchy.get_sister_hogs(hog_off, hog_tab, chog_buff)
#                 sis_hogs = [h for h in sis_hogs if hog_tab['TaxOff'][h] == hog_tax]
                
#                 # if all sisters are hidden, hide mask
#                 if (self.hog_filter_lca[sis_hogs] == True).all():
#                     hog_filter_lcasis[hog_off] = True
        
#         del hog_tab, chog_buff
#         return hog_filter_lcasis

#     @staticmethod
#     def get_clade_specific_taxa(stree_path, root_taxon):
#         '''
#         gather all taxa and species specific to a given clade
#         '''
#         stree = Tree(stree_path, format=1, quoted_node_names=True)

#         pruned_stree = [x for x in stree.traverse() if x.name == root_taxon][0]

#         taxa = set()
#         species = set()

#         for tl in pruned_stree.traverse():
#             taxon = tl.name.encode('ascii')
#             taxa.add(taxon)
#             if tl.is_leaf():
#                 species.add(taxon)

#         return np.array(sorted(taxa)), species


# class ReferenceExport():
#     '''
#     to export reference fasta for BLAST and DIAMOND and mimic IndexValidation
#     '''
#     def __init__(self, db, stree_path=None, hidden_taxon=None):
        
#         assert db.mode == 'r', 'Database must be opened in read mode.'
#         self.mode = 'r'

#         # load database object
#         self.db = db

#         # set index related features
#         self.idx_type = 'cs'

#         assert hidden_taxon, 'A hidden taxon must be defined'
#         self.hidden_taxon = ''.join(hidden_taxon.split())
        
#         # name
#         name = 'wo_{}_diamond'.format(self.hidden_taxon)
#         self.name = "{}_{}".format(self.db.name, name)
#         self.path = "{}{}/".format(self.db.path, self.name)

#         # hide all species of a given taxon
#         hidden_taxa, hidden_species = IndexValidation.get_clade_specific_taxa(stree_path, hidden_taxon)

#         self.hidden_species = np.array(list(hidden_species))
        
#         # find their offsets in sp_tab
#         hidden_species_offsets = np.searchsorted(self.db._sp_tab.col('ID'), self.hidden_species)
        
#         # build the species filter
#         self.sp_filter = np.zeros((self.db._sp_tab.nrows,), dtype=np.bool)
#         self.sp_filter[hidden_species_offsets] = True

#     # tax and HOG filters for Validation
#     @cached_property
#     def tax_filter(self):
#         hidden_taxa_offsets = np.searchsorted(self.db._tax_tab.col('ID'), self.hidden_species)
#         tax_filter = np.zeros((self.db._tax_tab.nrows,), dtype=np.bool)
#         tax_filter[hidden_taxa_offsets] = True
#         return tax_filter

#     @cached_property
#     def hog_filter_lca(self):
#         return self.tax_filter[self.db._hog_tab.col('LCAtaxOff')]

#     @cached_property
#     def hog_filter_lcasis(self):
#         hog_tab = self.db._hog_tab[:]
#         chog_buff = self.db._chog_arr[:]
#         hog_filter_lcasis = np.zeros(self.hog_filter_lca.size, dtype=np.bool)

#         for hog_off in range(self.hog_filter_lca.size):
#             if hog_tab['ParentOff'][hog_off] != -1:
                
#                 # sister HOGs at same taxon
#                 hog_tax = hog_tab['TaxOff'][hog_off]
#                 sis_hogs = hierarchy.get_sister_hogs(hog_off, hog_tab, chog_buff)
#                 sis_hogs = [h for h in sis_hogs if hog_tab['TaxOff'][h] == hog_tax]
                
#                 # if all sisters are hidden, hide mask
#                 if (self.hog_filter_lca[sis_hogs] == True).all():
#                     hog_filter_lcasis[hog_off] = True
        
#         del hog_tab, chog_buff
#         return hog_filter_lcasis   

#     def export_reference_fasta(self):

#         ref_sp = self.db._sp_tab.col('ID')[~self.sp_filter]
#         chunk_size = self.db._prot_tab.nrows
#         fasta_path = '{}reference/'.format(self.path)
#         if not os.path.exists(fasta_path):
#             os.makedirs(fasta_path)

#         self.export_species_fasta(self.db, ref_sp, chunk_size, fasta_path)

#     @staticmethod
#     def export_species_fasta(db, ref_sp, chunk_size, fasta_path):

#         i = 1
#         j = 0

#         fasta = open("{}{}.fa".format(fasta_path, j), 'w')

#         for sp in tqdm(ref_sp):
#             sb = QuerySequenceBuffer(db, sp)
            
#             for idx, seq in enumerate(sb):
                
#                 # new file
#                 if i % chunk_size == 0:
#                     fasta.close()
#                     fasta = open("{}{}.fa".format(fasta_path, j), 'w')
#                     j += 1
                
#                 fasta.write(">{}\n{}\n".format(sb.ids[idx][0], seq))
#                 i += 1
                
#         fasta.close()


# class SWindexValidation(ReferenceExport):
#     '''
#     to mimic IndexValidation for SW data
#     '''
#     def __init__(self, db, stree_path=None, hidden_taxon=None):
        
#         super().__init__(db, stree_path, hidden_taxon)
        
#         # just new name
#         name = 'wo_{}_SW'.format(self.hidden_taxon)
#         self.name = "{}_{}".format(self.db.name, name)
#         self.path = self.db.path


# class SequenceBuffer(object):
#     '''
#     to load sequences from db or files
#     adapted from alex
#     '''
#     def __init__(self, seqs=None, ids=None, db=None, fasta_file=None):
#         self.prot_off = 0
#         if seqs is not None:
#             self.add_seqs(*seqs)
#             self.ids = (np.array(ids) if ids else np.array(range(len(seqs))))
#         elif db is not None:
#             self._prot_tab = db._prot_tab
#             self._seq_buff = db._seq_buff
#             self.load_from_db()
#         elif fasta_file is not None:
#             seqs = self.parse_fasta(fasta_file)
#             self.add_seqs(*seqs)
#         else:
#             raise ValueError('need to pass array of seqs or db to load from to SequenceBuffer')

#     def __getstate__():
#         return (self.prot_nr, self.n, self.buff_shr, self.buff_idx_shr)

#     def __setstate__(state):
#         (self.prot_nr, self.n, self.buff_shr, self.buff_idx_shr) = state

#     def parse_fasta(self, fasta_file):
#         ids = []
#         seqs = []
#         for rec in SeqIO.parse(fasta_file, 'fasta'):
#             ids.append(rec.id)
#             seqs.append(str(rec.seq))
#         self.ids = np.array(ids)
#         return seqs

#     def add_seqs(self, *seqs):
#         self.prot_nr = len(seqs)
#         self.n = self.prot_nr + sum(len(s) for s in seqs)

#         self.buff_shr = sharedctypes.RawArray(ctypes.c_uint8, self.n)
#         self.buff[:] = np.frombuffer((' '.join(seqs) + ' ').encode('ascii'), dtype=np.uint8)

#         self.buff_idx_shr = sharedctypes.RawArray(ctypes.c_uint64, self.prot_nr + 1)
#         for i in range(len(seqs)):
#             self.idx[i+1] = len(seqs[i]) + 1 + self.idx[i]

#     def load_from_db(self):
#         self.prot_nr = len(self._prot_tab)
#         self.n = len(self._seq_buff)

#         self.buff_shr = sharedctypes.RawArray(ctypes.c_uint8, self.n)
#         self.buff[:] = self._seq_buff[:].view(np.uint8)

#         self.buff_idx_shr = sharedctypes.RawArray(ctypes.c_uint64, self.prot_nr + 1)
#         #self.idx[:-1] = self._prot_tab.cols.SeqOff[:]
#         self.idx[:-1] = self._prot_tab[:]['SeqOff']
#         self.idx[-1] = self.n

#         # store offsets as ids
#         self.ids = np.arange(self.prot_off, self.prot_off + self.prot_nr, dtype=np.uint64)[:,None]

#     @lazy_property
#     def buff(self):
#         return np.frombuffer(self.buff_shr, dtype=np.uint8).reshape(self.n)

#     @lazy_property
#     def idx(self):
#         return np.frombuffer(self.buff_idx_shr, dtype=np.uint64).reshape(self.prot_nr + 1)

#     # @lazy_property
#     # def prot_offsets(self):
#     #     return np.arange(self.prot_off, self.prot_off + self.prot_nr, dtype=np.uint64)

#     def __getitem__(self, i):
#         s = int(self.idx[i])
#         e = int(self.idx[i+1]-1)
#         return self.buff[s:e].tobytes().decode('ascii')


# class QuerySequenceBuffer(SequenceBuffer):
#     '''
#     get a sliced version of the sequence buffer object for a given species in db
#     TO DO: put an __super__
#     '''    
#     def __init__(self, db, query_sp):
#         self.query_sp = query_sp if isinstance(query_sp, bytes) else query_sp.encode('ascii')
#         self.set_query_prot_tab(db)
#         self.filter_query_seq_buff(db)
#         self.load_from_db()
        
#     def set_query_prot_tab(self, db): 
#         sp_off = np.searchsorted(db._sp_tab.col('ID'), self.query_sp)
#         sp_ent = db._sp_tab[sp_off]
#         self.prot_off = sp_ent['ProtOff']
#         self._prot_tab = db._prot_tab[self.prot_off: self.prot_off + sp_ent['ProtNum']]
        
#     def filter_query_seq_buff(self, db):
#         self._seq_buff = db._seq_buff[self._prot_tab[0]['SeqOff']: self._prot_tab[-1]['SeqOff'] + self._prot_tab[-1]['SeqLen']]
#         # initialize sequence buffer offset
#         self._prot_tab['SeqOff'] -= db._prot_tab[self.prot_off]['SeqOff']




