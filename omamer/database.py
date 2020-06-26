import os
import pyham
import tables
import numpy as np
import collections
from Bio import SeqIO
from tqdm import tqdm
from itertools import repeat
from property_manager import lazy_property
from ete3 import Tree
from tables import open_file

from .hierarchy import (
    get_root_leaf_hog_offsets,
    get_lca_hog_off,
    get_descendant_species_taxoffs,
    _children_prot, 
    traverse_HOG, 
    is_ancestor
)

# new orthoxml export
import re
import ete3
from ete3 import orthoxml

'''
UnitTest ideas:
 - correspondance between mappers (e.g. HOG prot children and )
 - parent offset not self (correspondance between parent HOGs and children HOGs)
 - assert size of string <= 255 (should be fine because this is what is used in the OMA hdf5)
'''

class Database():

	# define the format of the tables
	class ProteinTableFormat(tables.IsDescription):
		ID = tables.StringCol(255, pos=1, dflt=b'')
		SeqOff = tables.UInt64Col(pos=2)
		SeqLen = tables.UInt64Col(pos=3)
		SpeOff = tables.UInt64Col(pos=4)
		HOGoff = tables.UInt64Col(pos=5)
		FamOff = tables.UInt64Col(pos=6)

	class HOGtableFormat(tables.IsDescription):
		ID = tables.StringCol(255, pos=1, dflt=b'')
		FamOff = tables.UInt64Col(pos=2)
		TaxOff = tables.UInt64Col(pos=3)
		ParentOff = tables.Int64Col(pos=4)
		ChildrenHOGoff = tables.Int64Col(pos=5)
		ChildrenHOGnum = tables.Int64Col(pos=6)
		ChildrenProtOff = tables.Int64Col(pos=7)
		ChildrenProtNum = tables.Int64Col(pos=8)
		OmaID = tables.StringCol(255, pos=9, dflt=b'')
		LCAtaxOff = tables.UInt64Col(pos=10)

	class FamilyTableFormat(tables.IsDescription):
		ID = tables.UInt64Col(pos=1)
		TaxOff = tables.UInt64Col(pos=2)
		HOGoff = tables.UInt64Col(pos=3)
		HOGnum = tables.UInt64Col(pos=4)
		LevelOff = tables.UInt64Col(pos=5)
		LevelNum = tables.UInt64Col(pos=6)

	class SpeciesTableFormat(tables.IsDescription):
		ID = tables.StringCol(255, pos=1, dflt=b'')
		ProtOff = tables.UInt64Col(pos=2)
		ProtNum = tables.UInt64Col(pos=3)
		TaxOff = tables.UInt64Col(pos=4)

	class TaxonomyTableFormat(tables.IsDescription):
		ID = tables.StringCol(255, pos=1, dflt=b'')
		ParentOff = tables.Int64Col(pos=2)
		ChildrenOff = tables.Int64Col(pos=3)
		ChildrenNum = tables.Int64Col(pos=4)
		SpeOff = tables.Int64Col(pos=5)
		Level = tables.Int64Col(pos=6)

	def __init__(self, path, root_taxon, name=None):
		assert root_taxon, 'A root_taxon must be defined'
		self.path = path
		self.name = name if name else root_taxon
		self.root_taxon = root_taxon
		self.file = "{}{}.h5".format(self.path, self.name)
		
		self._compr = tables.Filters(complevel=6, complib='blosc', fletcher32=True)

		if os.path.isfile(self.file):
			self.mode = 'r'
			self.db = tables.open_file(self.file, self.mode, filters=self._compr)
		else:
			self.mode = 'w'
			self.db = tables.open_file(self.file, self.mode, filters=self._compr)

	def __exit__(self, *_):
		self.db.close()

	def clean(self):
		'''
		close and remove hdf5 file
		'''
		self.__exit__()
		try:
			os.remove(self.file)
		except FileNotFoundError:
			print("{} already cleaned".format(self.file))

    ### method attributes to facilitate access to data stored in hdf5 file ###
	@property
	def _prot_tab(self):
		if '/Protein' in self.db:
			return self.db.root.Protein
		else:
			return self.db.create_table('/', 'Protein', self.ProteinTableFormat, filters=self._compr)

	@property
	def _hog_tab(self):
		if '/HOG' in self.db:
			return self.db.root.HOG
		else:
			return self.db.create_table('/', 'HOG', self.HOGtableFormat, filters=self._compr)

	@property
	def _fam_tab(self):
		if '/Family' in self.db:
			return self.db.root.Family
		else:
			return self.db.create_table('/', 'Family', self.FamilyTableFormat, filters=self._compr)

	@property
	def _sp_tab(self):
		if '/Species' in self.db:
			return self.db.root.Species
		else:
			return self.db.create_table('/', 'Species', self.SpeciesTableFormat, filters=self._compr)

	@property
	def _tax_tab(self):
		if '/Taxonomy' in self.db:
			return self.db.root.Taxonomy
		else:
			return self.db.create_table('/', 'Taxonomy', self.TaxonomyTableFormat, filters=self._compr)

	# arrays
	@property
	def _seq_buff(self):
		if '/SequenceBuffer' in self.db:
			return self.db.root.SequenceBuffer
		else:
			# initiate 
			return self.db.create_earray('/', 'SequenceBuffer', tables.StringAtom(1), (0,), filters=self._compr)
	
	# The below arrays are of fixed length and created later as carray
	@property 
	def _chog_arr(self):
		if '/ChildrenHOG' in self.db:
			return self.db.root.ChildrenHOG
		else:
			return None

	@property
	def _cprot_arr(self):
		if '/ChildrenProt' in self.db:
			return self.db.root.ChildrenProt
		else:
			return None

	@property
	def _ctax_arr(self):
		if '/ChildrenTax' in self.db:
			return self.db.root.ChildrenTax
		else:
			return None

	@property
	def _level_arr(self):
		if '/LevelOffsets' in self.db:
			return self.db.root.LevelOffsets
		else:
			return None

	### functions common to DatabaseFromOMA and DatabaseFromFasta ###
	def initiate_tax_tab(self, stree_path):
		'''
		except SpeOff column
		'''
		assert (self.mode in {'w', 'a'}), 'Database must be opened in write mode.'

		def _parse_stree(stree_path, roottax):

		    stree = Tree(stree_path, format=1, quoted_node_names=True)
		    pruned_stree = [x for x in stree.traverse() if x.name == roottax][0]

		    tax2parent = {}
		    tax2children = {}
		    tax2level = {}
		    species = set()

		    for tl in pruned_stree.traverse():
		        tax = tl.name.encode('ascii')
		        tax2parent[tax] = tl.up.name.encode('ascii') if tl.up else 'root'.encode('ascii')
		        tax2children[tax] = [x.name.encode('ascii') for x in tl.children]
		        tax2level[tax] = tl.get_distance(stree)
		        if tl.is_leaf():
		            species.add(tl.name.encode('ascii'))

		    return tax2parent, tax2children, tax2level, species

		# parse species tree
		tax2parent, tax2children, tax2level, species = _parse_stree(stree_path, self.root_taxon)

		# sort taxa and create mapper to their offsets
		sorted_taxa = sorted(tax2parent.keys())
		tax2taxoff = dict(zip(sorted_taxa, range(len(tax2parent))))

		# collect rows of taxonomy table and children buffer
		tax_rows = []
		children_buffer = []

		children_buffer_off = 0
		for tax in sorted_taxa:

			# must not be root
			par_off = np.searchsorted(sorted_taxa, tax2parent[tax]) if tax != self.root_taxon.encode('ascii') else -1

			child_offsets = [np.searchsorted(sorted_taxa, x) for x in tax2children[tax]]

			# if no children, -1
			tax_rows.append((tax, par_off, children_buffer_off if child_offsets else -1, len(child_offsets), -1, tax2level[tax]))

			children_buffer.extend(child_offsets)
			children_buffer_off += len(child_offsets)

		# fill tax table
		self._tax_tab.append(tax_rows)
		self._tax_tab.flush()

		# store children taxa
		if not self._ctax_arr:
			self.db.create_carray('/', 'ChildrenTax', obj=np.array(children_buffer, dtype=np.int64), filters=self._compr)

		return tax2taxoff, species

	def update_hog_and_fam_tabs(self, fam2hogs, hog2taxoff, hog2protoffs, hog2oma_hog=None):
		'''
		'''
		assert (self.mode in {'w', 'a'}), 'Database must be opened in write mode.'

		def _get_parents(hog_off, hog_num, hogs):

			parents = []
			parent2child_hogs = collections.defaultdict(list)

			# sort alphabetically and track their true offsets
			alpha_hogs, alpha_hogs_offs = map(np.array, zip(*sorted(zip(hogs, range(hog_num)))))

			for i, hog_id in enumerate(hogs):
				child_hog = hog_off + i

				# use alphabetically sorted hogs here because np.searchsorted
				parent_hog = self.find_parent_hog(hog_id, alpha_hogs) 
				if parent_hog != -1:
					# map back to original hog offset and add the global hog_off
					parent_hog = hog_off + alpha_hogs_offs[parent_hog]
				parents.append(parent_hog)

				if parent_hog != -1: 
					parent2child_hogs[parent_hog].append(child_hog)

			return parents, parent2child_hogs

		def _get_child_hogs(hog_off, hog_num, parent2child_hogs, child_hogs_off):

			children_offsets = []
			children_numbers = []
			children = []
			children_off = child_hogs_off

			for tmp_hog_off in range(hog_off, hog_off + hog_num):

				# get child hogs from the current HOG's offset
				tmp_children = parent2child_hogs.get(tmp_hog_off, [])
				children_num = len(tmp_children)

				# if >0 child hogs, keep track of offset, number and child hogs
				if children_num > 0:
					children_offsets.append(children_off)
					children.extend(tmp_children)

				# else, store -1 as offset and 0 as count
				else:
					children_offsets.append(-1)

				# keep track of number of children
				children_numbers.append(children_num)

				# update children offset
				children_off += children_num

			return children_offsets, children_numbers, children, children_off

		def _get_child_prots(hogs, hog2protoffs, child_prots_off):

			children_offsets = []
			children_numbers = []
			children = []
			children_off = child_prots_off

			for hog in hogs:

				# get child prots from the current HOG id
				tmp_children = hog2protoffs.get(hog, [])
				children_num = len(tmp_children)

				# if >0 child prots, keep track of offset, number and child hogs
				if children_num > 0:
					children_offsets.append(children_off)
					children.extend(tmp_children)

				# else, store -1 as offset and 0 as count
				else:
					children_offsets.append(-1)

				# keep track of number of children
				children_numbers.append(children_num)

				# update children offset
				children_off += children_num

			return children_offsets, children_numbers, children, children_off

		# main
		hog_rows = []
		fam_rows = []

		hog_off = 0
		fam_off = 0 

		# initiate HOG and protein children arrays
		child_hogs = []
		child_prots = []
		child_hogs_off = 0
		child_prots_off = 0
 
		# initiate level array
		level_offsets = [0]
		level_offsets_off = 0

		for fam_id, hogs in sorted(fam2hogs.items(), key=lambda x: int(x[0])):

			### hogs
			# sort by level
			hogs = sorted(hogs, key=lambda x: len(x.split(b'.')))
			hog_num = len(hogs)
			hog_taxoffs = list(map(lambda x: hog2taxoff[x], hogs))

			# parents
			parents, parent2child_hogs = _get_parents(hog_off, hog_num, hogs)

			# levels
			hog_levels = list(map(lambda x: len(x.split(b'.')), hogs))
			hog_level_offsets = np.cumsum(np.unique(hog_levels, return_counts=True)[1]) + hog_off
			hog_level_offsets_num = len(hog_level_offsets)
			level_offsets.extend(hog_level_offsets)

			# children hogs
			child_hogs_offsets, child_hogs_numbers, tmp_child_hogs, child_hogs_off = _get_child_hogs(
				hog_off, hog_num, parent2child_hogs, child_hogs_off)

			child_hogs.extend(tmp_child_hogs)

			# children prots
			child_prots_offsets, child_prots_numbers, tmp_child_prots, child_prots_off = _get_child_prots(
				hogs, hog2protoffs, child_prots_off)

			child_prots.extend(tmp_child_prots) 

			# OMA hogs
			oma_hogs = list(map(lambda x: hog2oma_hog[x], hogs)) if hog2oma_hog else repeat(b'', hog_num)

			hog_rows.extend(list(zip(hogs, repeat(fam_off), hog_taxoffs, parents, child_hogs_offsets,
				child_hogs_numbers, child_prots_offsets, child_prots_numbers, oma_hogs, repeat(0))))

			### fams
			fam_rows.append((fam_id, hog_taxoffs[0], hog_off, hog_num, level_offsets_off, hog_level_offsets_num))

			hog_off += hog_num
			fam_off += 1
			level_offsets_off += hog_level_offsets_num

		# fill family and HOG tables
		self._fam_tab.append(fam_rows)
		self._fam_tab.flush()
		self._hog_tab.append(hog_rows)
		self._hog_tab.flush()
		
		# store children simulaneously without previous initiation because I did not know the size of these
		if not self._chog_arr:
			self.db.create_carray('/', 'ChildrenHOG', obj=np.array(child_hogs, dtype=np.uint64), filters=self._compr)
		if not self._cprot_arr:
			self.db.create_carray('/', 'ChildrenProt', obj=np.array(child_prots, dtype=np.uint64), filters=self._compr)
		
		# store offsets of levels of hogs inside families. copy the last one to go reverse
		level_offsets.append(level_offsets[-1])
		if not self._level_arr:
			self.db.create_carray('/', 'LevelOffsets', obj=np.array(level_offsets, dtype=np.int64), filters=self._compr)

	def add_speoff_col(self):
		'''
		to the taxonomy table
		'''
		# load species and taxa
		species = self._sp_tab.col('ID')
		taxa = self._tax_tab.col('ID')

		# potential idx of each taxon in species
		species_idx = np.searchsorted(species, taxa)

		# add extra elem because np.searchsorted gave idx 1 too high
		species = np.append(species, b'')

		# mask for taxa being species 
		species_mask = species[species_idx] == taxa

		# species offsets
		spe_offsets = np.arange(0, species.size - 1)

		# create SpeOff column
		speoff_col = np.full(taxa.size, -1)
		speoff_col[species_mask] = spe_offsets

		# update tax table
		self._tax_tab.modify_column(colname="SpeOff", column=speoff_col)

	def add_taxoff_col(self):
		'''
		to the species table
		'''
		# load species and taxa
		species = self._sp_tab.col('ID')
		taxa = self._tax_tab.col('ID')

		tax_offsets = np.searchsorted(taxa, species)

		self._sp_tab.modify_column(colname="TaxOff", column=tax_offsets)

	def update_prot_tab(self, hog2protoffs):
		'''
		add HOGoff and FamOff rows in the protein table
		'''
		assert (self.mode in {'w', 'a'}), 'Database must be opened in write mode.'

		def _get_protoffs2hogoff(hog2protoffs, hog2hogoff):
			protoff2hogoff = dict()
			for hog, protoffs in hog2protoffs.items():
				protoff2hogoff.update(dict(zip(protoffs,repeat(hog2hogoff[hog]))))
			return protoff2hogoff

		hog2hogoff = dict(zip(self._hog_tab.col("ID"), range(len(self._hog_tab))))
		protoffs2hogoff = _get_protoffs2hogoff(hog2protoffs, hog2hogoff)

		# get the hogoff column from it and the famoff col
		prot_offsets, hogoff_col = list(zip(*sorted(protoffs2hogoff.items())))

		assert len(prot_offsets) == (prot_offsets[-1] + 1), 'missing protein offsets, could be singletons'

		famoff_col = self._hog_tab[np.array(hogoff_col)]["FamOff"]

		# replace the empty columns
		self._prot_tab.modify_column(colname="HOGoff", column=hogoff_col)
		self._prot_tab.modify_column(colname="FamOff", column=famoff_col)

	def add_lcataxoff_col(self):
		'''
		compute the LCA taxon between the HOG species members
		'''
		hog_tab = self._hog_tab[:]
		chog_buff = self._chog_arr[:]
		cprot_buff = self._cprot_arr[:]
		prot2speoff = self._prot_tab.col('SpeOff')
		speoff2taxoff = self._sp_tab.col('TaxOff')
		parent_arr = self._tax_tab.col('ParentOff')		
		
		lcatax_offsets = np.zeros(hog_tab.size, dtype=np.uint64)
		
		for hog_off in range(hog_tab.size):
			species_tax_offs = get_descendant_species_taxoffs(hog_off, hog_tab, chog_buff, cprot_buff, prot2speoff, speoff2taxoff)
			lcatax_offsets[hog_off] = get_lca_hog_off(species_tax_offs, parent_arr)

		self._hog_tab.modify_column(colname="LCAtaxOff", column=lcatax_offsets)

	### generic functions
	@staticmethod
	def parse_hogs(hog_id):
		if isinstance(hog_id, str):
			split_hog_id = hog_id.split('.')
			return ['.'.join(split_hog_id[:i + 1]) for i in range(len(split_hog_id))]
		elif isinstance(hog_id, bytes):
			split_hog_id = hog_id.split(b'.')
			return [b'.'.join(split_hog_id[:i + 1]) for i in range(len(split_hog_id))]

	@staticmethod
	def find_parent_hog(hog_id, hogs):
		if isinstance(hog_id, str):
			split_hog = hog_id.split('.')
			parent_hog = -1
			if len(split_hog) > 1:
				parent_hog = np.searchsorted(hogs, '.'.join(split_hog[:-1]))
			return parent_hog
		elif isinstance(hog_id, bytes):
			split_hog = hog_id.split(b'.')
			parent_hog = -1
			if len(split_hog) > 1:
				parent_hog = np.searchsorted(hogs, b'.'.join(split_hog[:-1]))
			return parent_hog

	### export functions
	@staticmethod
	def format_stree_template(stree_file):
		'''
		add speciation events and taxa to the species tree
		use as templates to build the HOG gene trees inbetween duplication events
		'''
		stree = ete3.Tree(stree_file, format=1, quoted_node_names=True)
		for node in stree.traverse():

			# keep track of taxon and its level (distance from root)
			node.add_features(taxon = node.name, root_dist = node.get_distance(stree))

			if node.is_leaf():
			    continue
			node.add_features(S=node.name, Ev='0>1')
		    
		return stree

	@staticmethod
	def _HOG2GT(hog_off, hog2taxon2subtrees, leaves, hog_tab2, tax_tab, template_stree, tax_id2tax_off, cprot_buff, sp_tab, prot_tab):
	    hog_ent = hog_tab2[hog_off]
	    
	    # create the HOG subtree as a copy the taxonomy (species tree) from the taxon where the HOG is defined
	    taxon = tax_tab[hog_ent['TaxOff']]['ID'].decode('ascii')
	    hog_st = (template_stree&taxon).copy()

	    # add the HOG id at each node
	    hog_id = hog_ent['ID'].decode('ascii')
	    for node in hog_st.traverse():
	        node.add_features(hog_name = hog_id)
	        
	    ### add the previously computed HOG subtrees to the current one
	    if hog_off in hog2taxon2subtrees:
	        
	        ## start by grouping taxon on the same path (can happen when >1 subsequent dupl AND some losses)
	        taxon2subtrees = hog2taxon2subtrees[hog_off]
	        taxon_ids = list(taxon2subtrees.keys())
	        tax_levels = tax_tab['Level'][[tax_id2tax_off[x] for x in taxon_ids]]

	        postdupl_tax2taxa = collections.defaultdict(list)
	        connected_taxa = set()

	        # top-down traversal
	        for postdupl_tax, l1 in sorted(zip(taxon_ids, tax_levels), key=lambda x: x[1]):

	            # skip if already connected to a postdupl_tax
	            if postdupl_tax in connected_taxa:
	                continue

	            # bottom-up
	            for tax, l2 in sorted(zip(taxon_ids, tax_levels), key=lambda x: x[1], reverse=True):

	                if is_ancestor(tax_id2tax_off[postdupl_tax], tax_id2tax_off[tax], tax_tab['ParentOff']):
	                    postdupl_tax2taxa[postdupl_tax].append(tax)
	                    connected_taxa.add(tax)
	                    
	        ## add duplications and graph corresponding HOG subtrees
	        for pdtax, taxa in postdupl_tax2taxa.items():

	            # add the duplication node
	            parent = (hog_st&pdtax).up
	            dupl = parent.add_child(name='dupl')
	            dupl.dist = 0.5
	            dupl.add_features(Ev='1>0', hog_name=hog_id, taxon=pdtax, root_dist = parent.root_dist + 0.5)

	            # add subtrees to the duplication node
	            for tax in taxa:
	                for t in taxon2subtrees[tax]:
	                    t.dist = 0.5
	                    dupl.add_child(t)

	            # remove the original taxon subtree
	            (hog_st&pdtax).detach()

	    ### traverse the HOG subtree, relabel extant genes and prune species without genes
	    hog_prots = _children_prot(hog_off, hog_tab2, cprot_buff)
	    hog_species = list(map(lambda x:x.decode('ascii'), sp_tab[prot_tab[hog_prots]['SpeOff']]['ID']))
	    hog_sp2prot = dict(zip(hog_species, hog_prots))
	    leaves.update(hog_prots)

	    for leaf in hog_st.get_leaves():
	        lname = leaf.name 

	        # rename
	        if lname in hog_species:
	            leaf.name = hog_sp2prot[lname]

	        # prune
	        elif lname not in leaves:
	            leaf.delete(preserve_branch_length=True)

	    ### keep track of extant genes and return the hog gene tree
	    parent_hog_off = hog_ent['ParentOff']
	    if parent_hog_off not in hog2taxon2subtrees:
	        hog2taxon2subtrees[parent_hog_off] = collections.defaultdict(set)
	    hog2taxon2subtrees[parent_hog_off][taxon].add(hog_st)
	    
	    return hog2taxon2subtrees, leaves

	def HOG2GT(self, hog_off, hog_tab, chog_buff, tax_tab, template_stree, tax_id2tax_off, cprot_buff, sp_tab, prot_tab):
	    
	    hog2taxon2subtrees = {}
	    leaves = set()
	    
	    hog2taxon2subtrees, leaves = traverse_HOG(
	        hog_off, hog_tab, chog_buff, postorder_fun=self._HOG2GT, leaf_fun=self._HOG2GT, acc1=hog2taxon2subtrees, 
	        acc2=leaves, hog_tab2=hog_tab, tax_tab=tax_tab, template_stree=template_stree, tax_id2tax_off=tax_id2tax_off, 
	        cprot_buff=cprot_buff, sp_tab=sp_tab, prot_tab=prot_tab)
	    

	    hog_root_taxon = tax_tab[hog_tab[hog_off]['TaxOff']]['ID'].decode('ascii')
	    hog_tree = list(hog2taxon2subtrees[-1][hog_root_taxon])[0]

	    # remove single child speciations
	    hog_tree.prune(hog_tree.get_leaves())
	    
	    return hog_tree

	# def export_orthoXML(
	# 	self, fam_offsets, stree_file, output_path, orthoxml_name, origin='OMAmer 0.1', version='0.3', originVersion='0.2'):

	# 	def HOG2tree(hog_off, hog2taxon2subtrees, leaves, tax_tab, hog_tab2, stree, cprot_buff, sp_tab, prot_tab):
	# 	    hog_ent = hog_tab2[hog_off]

	# 	    # copy the species tree from the HOG taxonomic level
	# 	    taxon = tax_tab[hog_ent['TaxOff']]['ID'].decode('ascii')
	# 	    st = (stree&taxon).copy()

	# 	    # add HOG id at each node
	# 	    hog_id = hog_ent['ID'].decode('ascii')
	# 	    for node in st.traverse():
	# 	        node.add_features(hog = hog_id)

	# 	    # add the previously computed HOG subtrees
	# 	    if hog_off in hog2taxon2subtrees:
	# 	        for ctax, subtrees in hog2taxon2subtrees[hog_off].items():
	# 	            assert len(subtrees) > 1, 'a duplication should lead to mininum 2 sub-HOGs'

	# 	            # add the duplication node and the HOG gene trees
	# 	            parent = (st&ctax).up
	# 	            dupl = parent.add_child(name='dupl')
	# 	            dupl.add_features(Ev='1>0')
	# 	            for t in subtrees:
	# 	                dupl.add_child(t)

	# 	            # remove the taxon subtree
	# 	            (st&ctax).detach()

	# 	    # get the species and corresponding proteins for that HOG (excluding sub-HOGs)
	# 	    hog_prots = _children_prot(hog_off, hog_tab, cprot_buff)
	# 	    hog_species = list(map(lambda x:x.decode('ascii'), sp_tab[prot_tab[hog_prots]['SpeOff']]['ID']))
	# 	    hog_sp2prot = dict(zip(hog_species, hog_prots))

	# 	    # update leaves
	# 	    leaves.update(hog_prots)

	# 	    # traverse species tree, relabel extand genes and prune species without genes
	# 	    for leaf in st.get_leaves():
	# 	        lname = leaf.name 

	# 	        # rename
	# 	        if lname in hog_species:
	# 	            leaf.name = hog_sp2prot[lname]

	# 	        # prune
	# 	        elif lname not in leaves:
	# 	            leaf.delete()

	# 	    # keep track of extant genes and return the hog gene tree
	# 	    parent_hog_off = hog_ent['ParentOff']
	# 	    if parent_hog_off not in hog2taxon2subtrees:
	# 	        hog2taxon2subtrees[parent_hog_off] = collections.defaultdict(set)
	# 	    hog2taxon2subtrees[parent_hog_off][taxon].add(st)

	# 	    return hog2taxon2subtrees, leaves

	# 	fam_tab = self._fam_tab[:]
	# 	hog_tab = self._hog_tab[:]
	# 	tax_tab = self._tax_tab[:]
	# 	sp_tab = self._sp_tab[:]
	# 	prot_tab = self._prot_tab[:]
	# 	chog_buff = self._chog_arr[:]
	# 	cprot_buff = self._cprot_arr[:]

	# 	### initiate orthoXML  
	# 	# add an ortho group container to the orthoXML document
	# 	ortho_groups = orthoxml.groups()
	# 	xml = orthoxml.orthoXML(origin=origin, version=version, originVersion=originVersion)
	# 	xml.set_groups(ortho_groups)  

	# 	# store orthoxml species and gene objects
	# 	sp2genes = {}

	# 	# specific to the orthoXML file
	# 	unique_id = 1

	# 	### edit reference species with internal node labels (taxon and speciation event)
	# 	stree = ete3.Tree(stree_file, format=1, quoted_node_names=True)
	# 	for node in stree.traverse():
	# 	    if node.is_leaf():
	# 	        continue
	# 	    node.add_features(S=node.name, Ev='0>1')

	# 	for fam_off in fam_offsets:
	# 	    fam_ent = fam_tab[fam_off]
	# 	    root_hog_off = fam_ent['HOGoff']

	# 	    hog2taxon2subtrees = {}
	# 	    leaves = set()
		    
	# 	    hog2taxon2subtrees, leaves = traverse_HOG(
	# 	        root_hog_off, hog_tab, chog_buff, postorder_fun=HOG2tree, leaf_fun=HOG2tree, acc1=hog2taxon2subtrees, 
	# 	        acc2=leaves, tax_tab=tax_tab, hog_tab2=hog_tab, stree=stree, cprot_buff=cprot_buff, sp_tab=sp_tab, 
	# 	        prot_tab=prot_tab)

	# 	    fam_root_taxon = tax_tab[fam_ent['TaxOff']]['ID'].decode('ascii')
	# 	    fam = list(hog2taxon2subtrees[-1][fam_root_taxon])[0]
		    
	# 	    # remove single child speciations
	# 	    fam.prune(fam.get_leaves())
		    
	# 	    ### add genes and species to the orthoXML
	# 	    leaf_id2unique_id = {}
	# 	    for leaf in fam.get_leaves():
	# 	        leaf_id = leaf.name
	# 	        prot_ent = prot_tab[leaf_id]
	# 	        spname = sp_tab[prot_ent['SpeOff']]['ID'].decode('ascii')
	# 	        uniprot_id = prot_ent['ID'].decode('ascii')

	# 	        # create species and genes containers
	# 	        if spname not in sp2genes:
	# 	            sp = orthoxml.species(spname)
	# 	            datb = orthoxml.database(name='OMAmer')
	# 	            sp.add_database(datb)
	# 	            genes = orthoxml.genes()
	# 	            datb.set_genes(genes)
	# 	            sp2genes[spname] = genes
	# 	            # add info to the orthoXML document
	# 	            xml.add_species(sp)
	# 	        else:
	# 	            genes = sp2genes[spname]

	# 	        # store the leaf gene in 'genes' of 'sp'
	# 	        gn = orthoxml.gene(protId=uniprot_id, id=unique_id)
	# 	        genes.add_gene(gn)

	# 	        # track gene id
	# 	        leaf_id2unique_id[leaf_id]=unique_id
	# 	        unique_id += 1 
		    
	# 	    ### add groups to the orthoXML
	# 	    if fam.is_leaf():
	# 	        continue

	# 	    assert fam.Ev=='0>1', 'A family should start with a speciation node'

	# 	    node2group = {}

	# 	    # create the root-HOG
	# 	    taxon = orthoxml.property('TaxRange', fam.S)
	# 	    node2group[fam] = orthoxml.group(id=fam.hog, property=[taxon])
	# 	    ortho_groups.add_orthologGroup(node2group[fam])

	# 	    # top-down traversal
	# 	    for node in fam.traverse("preorder"):
	# 	        if node.is_leaf():
	# 	            continue

	# 	        group = node2group[node]
	# 	        hog_id = group.id
	# 	        event = node.Ev

	# 	        for child in node.children:
	# 	            if child.is_leaf():
	# 	                # Add the gene to the group 
	# 	                unique_id = leaf_id2unique_id[child.name]
	# 	                group.add_geneRef(orthoxml.geneRef(id=unique_id))
	# 	                continue

	# 	            child_event = child.Ev

	# 	            # A duplication here means no new HOG
	# 	            if child_event == "1>0":
	# 	                node2group[child] = orthoxml.group(id=hog_id)
	# 	                group.add_paralogGroup(node2group[child])

	# 	            else:
	# 	                taxon = orthoxml.property('TaxRange', child.S)

	# 	                # A speciation following a duplication means a new (explicit) HOG
	# 	                if event == "1>0": 
	# 	                    node2group[child] = orthoxml.group(id=child.hog, property=[taxon])

	# 	                else:
	# 	                    node2group[child] = orthoxml.group(id=hog_id, property=[taxon])

	# 	                group.add_orthologGroup(node2group[child])

	# 	with open('{}{}_tmp.orthoxml'.format(output_path, orthoxml_name), 'w') as outf:
	# 	    xml.export(outf, 0, namespace_="")

	# 	with open('{}{}_tmp.orthoxml'.format(output_path, orthoxml_name), 'r') as inf:
	# 	    # skip first line
	# 	    inf.readline()
	# 	    xml_read = inf.read()

	# 	pat = re.compile(r'(.*=)b\'(?P<byte>\".*\")\'(.*)', re.MULTILINE)
	# 	xml_tmp = pat.sub(r'\1\2\3', xml_read) # because two pattern per line
	# 	xml_write = pat.sub(r'\1\2\3', xml_tmp)  

	# 	with open('{}{}.orthoxml'.format(output_path, orthoxml_name), 'w') as outf:
	# 	    outf.write('<?xml version="1.0" encoding="UTF-8"?>\n')  # not sure if necessary
	# 	    outf.write('<orthoXML xmlns="http://orthoXML.org/2011/" version="{}" origin="{}" originVersion="{}">\n'.format(version, origin, originVersion))
	# 	    outf.write(xml_write)

	# 	os.remove('{}{}_tmp.orthoxml'.format(output_path, orthoxml_name))


class DatabaseFromOMA(Database):
	'''
	parse the OMA hdf5
	'''
	def __init__(self, path, root_taxon, name=None, min_prot_nr=6, max_prot_nr=1000000, include_younger_fams=True):
		super().__init__(path, root_taxon, name)

		self.min_prot_nr = min_prot_nr
		self.max_prot_nr = max_prot_nr

		self.include_younger_fams = include_younger_fams
	
	### main function ###
	def build_database(self, oma_h5_path, stree_path):

		assert (self.mode in {'w', 'a'}), 'Database must be opened in write mode.'

		# build taxonomy table except the SpeOff column
		print("initiate taxonomy table")
		tax2taxoff, species = self.initiate_tax_tab(stree_path)

		# load OMA h5 file
		h5file = open_file(oma_h5_path, mode="r")

		print("select and strip OMA HOGs")
		fam2hogs, hog2oma_hog, hog2tax = self.select_and_strip_OMA_HOGs(h5file)

		print("fill sequence buffer, species table and initiate protein table")
		fam2hogs, hog2protoffs, hog2tax, hog2oma_hog = self.select_and_filter_OMA_proteins(
			h5file, fam2hogs, hog2oma_hog, hog2tax, species, self.min_prot_nr, self.max_prot_nr)

		print("add SpeOff and TaxOff columns in taxonomy and species tables, respectively")
		self.add_speoff_col()
		self.add_taxoff_col()

		# mapper HOG to taxon offset
		hog2taxoff = {h: tax2taxoff.get(t, -1) for h, t in hog2tax.items()}

		print("fill family and HOG tables")
		self.update_hog_and_fam_tabs(fam2hogs, hog2taxoff, hog2protoffs, hog2oma_hog)

		# add family and hog offsets
		print("complete protein table")
		self.update_prot_tab(hog2protoffs)

		print("compute LCA taxa")
		self.add_lcataxoff_col()

		# close and open in read mode
		h5file.close()
		self.db.close()
		self.mode = 'r'
		self.db = tables.open_file(self.file, self.mode, filters=self._compr)	

	### functions to parse OMA database ###
	def select_and_strip_OMA_HOGs(self, h5file):

		# def _process_oma_hog(tax2level, curr_oma_taxa, curr_oma_hog, curr_oma_roothog, fam, fam2hogs, hog2oma_hog, hog2tax, roottax):
		# 	'''
		# 	- decide whether an OMA HOG should be stored based on current root-HOG and HOG taxa
		# 	- update root-HOG if necessary and keep track of HOG taxa
		# 	'''
		# 	def _is_descendant(hog1, hog2):
		# 	    '''
		# 	    True if hog1 descendant of hog2
		# 	    '''
		# 	    return hog2 in self.parse_hogs(hog1)

		# 	def _store(fam, curr_oma_hog, curr_oma_roothog, fam2hogs, hog2oma_hog, hog2tax, tax):
		# 	    '''
		# 	    - compute SPhog HOG id
		# 	    - fill containers
		# 	    '''
		# 	    hog = b'.'.join([str(fam).encode('ascii'), *curr_oma_hog.split(b'.')[len(curr_oma_roothog.split(b'.')):]])

		# 	    fam2hogs[fam].add(hog)
		# 	    hog2oma_hog[hog] = curr_oma_hog
		# 	    hog2tax[hog] = tax
			    
		# 	# compute most ancestral taxon; when absent, consider more ancestral (-1)
		# 	tax_levels = list(map(lambda x: tax2level.get(x, -1), curr_oma_taxa))
		# 	min_level = min(tax_levels)

		# 	if min_level == -1:
		# 	    tax = None     
		# 	else:
		# 	    tax = curr_oma_taxa[tax_levels.index(min_level)]

		# 	# store if descendant of current OMA root-HOG
		# 	if _is_descendant(curr_oma_hog, curr_oma_roothog):
			    
		# 	    _store(fam, curr_oma_hog, curr_oma_roothog, fam2hogs, hog2oma_hog, hog2tax, tax)

		# 	# store, update fam and current OMA root-HOG if not descendant and start at root taxon
		# 	# ! ignorint OMA root-HOG younger than roottax !
		# 	elif tax == roottax:
			    
		# 	    # care about HOG quality or number of members?

		# 	    fam += 1
		# 	    curr_oma_roothog = curr_oma_hog           
			    
		# 	    # store after updating fam
		# 	    _store(fam, curr_oma_hog, curr_oma_roothog, fam2hogs, hog2oma_hog, hog2tax, tax)
			    
		# 	return fam, curr_oma_roothog

		# def _process_oma_fam(fam_tab_sort, tax2level, fam, fam2hogs, hog2oma_hog, hog2tax, roottax):
		# 	'''
		# 	apply _process_oma_hog to one OMA family
		# 	 - fam_tab_sort: slice of the HogLevel table for one family sorted by HOG ids
		# 	 - tax2level: mapper between taxa and their distance from roottax
		# 	'''
		# 	# bookeepers for HOGs
		# 	curr_oma_hog = fam_tab_sort[0]['ID']
		# 	curr_oma_roothog = None
		# 	curr_oma_taxa = []

		# 	for r in fam_tab_sort:
		# 	    oma_hog, oma_tax = r[1], b''.join(r[2].split())
			    
		# 	    # evaluation at new oma HOG 
		# 	    if oma_hog != curr_oma_hog:

		# 	        fam, curr_oma_roothog = _process_oma_hog(tax2level, curr_oma_taxa, curr_oma_hog, curr_oma_roothog, fam, fam2hogs, hog2oma_hog, hog2tax, roottax)
			        
		# 	        # reset for new HOG
		# 	        curr_oma_taxa = []
		# 	        curr_oma_hog = oma_hog
			        
		# 	    # track taxa of current oma HOG 
		# 	    curr_oma_taxa.append(oma_tax)

		# 	# end
		# 	fam, curr_oma_roothog = _process_oma_hog(tax2level, curr_oma_taxa, curr_oma_hog, curr_oma_roothog, fam, fam2hogs, hog2oma_hog, hog2tax, roottax)

		# 	return fam
		def _process_oma_hog(
			tax2level, curr_oma_taxa, curr_oma_hog, curr_oma_roothog, fam, fam2hogs, hog2oma_hog, hog2tax, roottax, 
			include_younger_fams, tax_id2tax_off, tax_off2parent):
			'''
			- decide whether an OMA HOG should be stored based on current root-HOG and HOG taxa
			- update root-HOG if necessary and keep track of HOG taxa
			'''
			def _is_descendant(hog1, hog2):
			    '''
			    True if hog1 descendant of hog2
			    '''
			    return hog2 in self.parse_hogs(hog1)

			def _store(fam, curr_oma_hog, curr_oma_roothog, fam2hogs, hog2oma_hog, hog2tax, tax):
			    '''
			    - compute SPhog HOG id
			    - fill containers
			    '''
			    hog = b'.'.join([str(fam).encode('ascii'), *curr_oma_hog.split(b'.')[len(curr_oma_roothog.split(b'.')):]])

			    fam2hogs[fam].add(hog)
			    hog2oma_hog[hog] = curr_oma_hog
			    hog2tax[hog] = tax

			# compute most ancestral taxon; when absent, consider more ancestral (-1)
			tax_levels = list(map(lambda x: tax2level.get(x, -1), curr_oma_taxa))
			min_level = min(tax_levels)

			if min_level == -1:
			    tax = None     
			else:
			    tax = curr_oma_taxa[tax_levels.index(min_level)]

			# store if descendant of current OMA root-HOG
			if _is_descendant(curr_oma_hog, curr_oma_roothog):

			    _store(fam, curr_oma_hog, curr_oma_roothog, fam2hogs, hog2oma_hog, hog2tax, tax)

			# store, update fam and current OMA root-HOG if passes tax_filter
			else:
				# either taxon is more ancestral
				if not tax:
					tax_filter = False

				# or it is the root-taxon
				elif tax == roottax:
					tax_filter = True

				# or it is younger, and we care
				elif include_younger_fams:
					tax_filter = True

				# younger and we don't care
				else:
					tax_filter = False

				# no need to check ancestry because if it is not none and not root-taxon, it must be younger
				# if not tax_filter and include_younger_fams:
				#     tax_filter = is_ancestor(
				#         tax_id2tax_off[roottax], tax_id2tax_off[tax], tax_off2parent)

				if tax_filter:
				    # place to filter with completeness or implied loss scores

				    fam += 1
				    curr_oma_roothog = curr_oma_hog           

				    # store after updating fam
				    _store(fam, curr_oma_hog, curr_oma_roothog, fam2hogs, hog2oma_hog, hog2tax, tax)

			return fam, curr_oma_roothog

		def _process_oma_fam(
		    fam_tab_sort, tax2level, fam, fam2hogs, hog2oma_hog, hog2tax, roottax, include_younger_fams, tax_id2tax_off, 
		    tax_off2parent):
		    '''
		    apply _process_oma_hog to one OMA family
		     - fam_tab_sort: slice of the HogLevel table for one family sorted by HOG ids
		     - tax2level: mapper between taxa and their distance from roottax
		    '''
		    # bookeepers for HOGs
		    curr_oma_hog = fam_tab_sort[0]['ID']
		    curr_oma_roothog = None
		    curr_oma_taxa = []

		    for r in fam_tab_sort:
		        oma_hog, oma_tax = oma_hog, oma_tax = r[1], r[2]

		        # evaluation at new oma HOG 
		        if oma_hog != curr_oma_hog:

		            fam, curr_oma_roothog = _process_oma_hog(
		                tax2level, curr_oma_taxa, curr_oma_hog, curr_oma_roothog, fam, fam2hogs, hog2oma_hog, hog2tax, roottax,
		                include_younger_fams, tax_id2tax_off, tax_off2parent)

		            # reset for new HOG
		            curr_oma_taxa = []
		            curr_oma_hog = oma_hog

		        # track taxa of current oma HOG 
		        curr_oma_taxa.append(oma_tax)

		    # end
		    fam, curr_oma_roothog = _process_oma_hog(
		        tax2level, curr_oma_taxa, curr_oma_hog, curr_oma_roothog, fam, fam2hogs, hog2oma_hog, hog2tax, roottax, 
		        include_younger_fams, tax_id2tax_off, tax_off2parent)

		    return fam

		#
		tax2level = dict(zip(self._tax_tab[:]['ID'], self._tax_tab[:]['Level']))
		tax_off2parent = self._tax_tab.col('ParentOff')
		tax_id2tax_off = dict(zip(self._tax_tab.col('ID'), range(self._tax_tab.nrows))) 

		# load families in memory
		hog_tab = h5file.root.HogLevel
		families = hog_tab.col('Fam')

		# containers
		fam2hogs = collections.defaultdict(set)
		hog2oma_hog = dict()
		hog2tax = dict()

		# bookeepers for families
		fam = 0
		curr_fam = families[0]
		i = 0
		j = 0

		for oma_fam in tqdm(families):
			if oma_fam != curr_fam:

				# load fam table and sort by HOG ids
				fam_tab = hog_tab[i:j]
				fam_tab = np.sort(fam_tab, order='ID')

				# select and strip HOGs of one family
				fam = _process_oma_fam(fam_tab, tax2level, fam, fam2hogs, hog2oma_hog, hog2tax, self.root_taxon.encode('ascii'),
					self.include_younger_fams, tax_id2tax_off, tax_off2parent)

				# move pointer and update current family
				i = j
				curr_fam = oma_fam       
			j += 1

		# end
		fam_tab = hog_tab[i:j]
		fam_tab = np.sort(fam_tab, order='ID')
		fam = _process_oma_fam(fam_tab, tax2level, fam, fam2hogs, hog2oma_hog, hog2tax, self.root_taxon.encode('ascii'), 
			self.include_younger_fams, tax_id2tax_off, tax_off2parent)

		del hog_tab, families

		return fam2hogs, hog2oma_hog, hog2tax

	def select_and_filter_OMA_proteins(self, h5file, fam2hogs, hog2oma_hog, hog2tax, species, min_prot_nr, max_prot_nr):
		'''
		One small diff compared to DatabaseFromFasta is that proteins in protein table are not following the species in species table. This should be fine.
		'''
		genome_tab = h5file.root.Genome[:]
		ent_tab = h5file.root.Protein.Entries
		oma_seq_buffer = h5file.root.Protein.SequenceBuffer

		# temporary mappers
		oma_hog2hog = dict(zip(hog2oma_hog.values(), hog2oma_hog.keys()))
		spe2speoff = dict(zip(sorted(species), range(len(species))))  # this is to sort the species table

		print(" - select proteins from selected HOGs")
		# remember the size of families and bookkeep family of each protein for filtering 
		fam2prot_nr = collections.Counter()
		prot_fams = []

		# bookkeep HOG of each protein to later build the hog2protoffs mapper
		prot_hogs = []

		# bookkeep sequence offset of each protein to later build the local seq_buff
		oma_seq_offsets = []

		# store temporary rows for species and protein tables
		oma_species = [b''] * len(species)  # to keep species sorted
		prot_rows = []

		for r in tqdm(genome_tab):

			# sp = b''.join(r['SciName'].split())  # because stop merging species and genus
			sp = r['SciName']

			# for ~27 cases the uniprot id replaces the scientific name in OMA species tree
			if sp not in species:   
				sp = r['UniProtSpeciesCode']

			if sp in species:
				entry_off = r['EntryOff']
				entry_num = r['TotEntries']

				spe_off = spe2speoff[sp]
				oma_species[spe_off] = sp

				# sub entry table for species
				sp_ent_tab = ent_tab[entry_off: entry_off + entry_num]

				for rr in sp_ent_tab:
				    oma_hog = rr['OmaHOG']

				    # select protein if member of selected OMA HOG
				    if oma_hog in oma_hog2hog:

				        # update counter and track hogs and families
				        hog = oma_hog2hog[oma_hog]
				        fam = int(hog.split(b'.')[0].decode('ascii'))
				        fam2prot_nr[fam] += 1
				        prot_hogs.append(hog)
				        prot_fams.append(fam) 

				        # track sequence length and offset
				        oma_seq_off = rr['SeqBufferOffset']
				        seq_len = rr['SeqBufferLength']
				        oma_seq_offsets.append(oma_seq_off)

				        # store protein row
				        prot_rows.append((rr['EntryNr'], 0, seq_len, spe_off, 0, 0))
		
		print(" - filter by family protein number")
		# now filter the mappers
		f_fam2hogs = collections.defaultdict(set)
		f_hog2oma_hog = dict()
		f_hog2tax = dict()

		for fam, hogs in fam2hogs.items():

			prot_nr = fam2prot_nr[fam]

			# filter by size
			if prot_nr >= min_prot_nr and prot_nr <= max_prot_nr:
			    
			    f_fam2hogs[fam] = hogs
			    f_hog2oma_hog.update(dict(zip(hogs, list(map(lambda x: hog2oma_hog[x], hogs)))))
			    f_hog2tax.update(dict(zip(hogs, list(map(lambda x: hog2tax[x], hogs)))))

		print(" - filter proteins based on filtered families")
		# bookkeeping for later
		hog2protoffs = collections.defaultdict(set)

		# pointer to species in species table
		curr_spe_off = prot_rows[0][3]

		# pointer to protein in protein table
		prot_off = self._prot_tab.nrows
		curr_prot_off = prot_off

		# pointer to sequence in buffer
		seq_off = self._seq_buff.nrows

		# store rows for species and protein tables	and sequence buffer
		spe_rows = [()] * len(species)  # keep sorted
		f_prot_rows = []
		seq_buff = []

		# iter trough temporary protein rows
		i = 0
		for r in tqdm(prot_rows):

			if prot_fams[i] in f_fam2hogs:

				spe_off = r[3]

				# change of species --> store current species
				if spe_off != curr_spe_off:

					spe_rows[curr_spe_off] = (oma_species[curr_spe_off], curr_prot_off, prot_off - curr_prot_off, 0)

					curr_prot_off = prot_off
					curr_spe_off = spe_off

					# take advantage to dump sequence buffer
					self._seq_buff.append(seq_buff)
					self._seq_buff.flush()
					seq_buff = []

				# sequence
				oma_seq_off = oma_seq_offsets[i]
				seq_len = r[2]
				seq = oma_seq_buffer[oma_seq_off: oma_seq_off + seq_len]  
				seq_buff.extend(list(seq)) 

				# store protein row
				f_prot_rows.append((r[0], seq_off, seq_len, spe_off, 0, 0))

				# book keeping
				hog2protoffs[prot_hogs[i]].add(prot_off)

				# update offset of protein sequence in buffer and of protein raw in table
				seq_off += seq_len
				prot_off += 1       
			i += 1

		# end
		spe_rows[curr_spe_off] = (oma_species[curr_spe_off], curr_prot_off, prot_off - curr_prot_off, 0)

		self._seq_buff.append(seq_buff)
		self._seq_buff.flush()

		# fill species and protein tables
		self._sp_tab.append(spe_rows)
		self._sp_tab.flush()
		self._prot_tab.append(f_prot_rows)
		self._prot_tab.flush()

		del genome_tab, ent_tab, oma_seq_buffer

		return f_fam2hogs, hog2protoffs, f_hog2tax, f_hog2oma_hog

	@staticmethod
	def format_stree_template(stree_file):
		'''
		TO DO: to merge it with the parent class one, I need to stop formating species name
		'''
		stree = ete3.Tree(stree_file, format=1, quoted_node_names=True)
		for node in stree.traverse():

			# mimic formating of tax table (remove the space)
			node.name = ''.join(node.name.split())

			# keep track of taxon and its level (distance from root)
			node.add_features(taxon = node.name, root_dist = node.get_distance(stree))

			if node.is_leaf():
			    continue
			node.add_features(S=node.name, Ev='0>1')
		    
		return stree


class DatabaseFromOrthoXML(Database):
	'''
	PANTHER could be a subclass of this one, even including the panther2orthoxml step.
	One difference is how sequences are imported and the addition of GO information
	'''
	def __init__(self, path, root_taxon, name=None):
		super().__init__(path, root_taxon, name)

	def build_database(self, orthoxml_file, stree_file, aln_files, pthfam_an2prot_file, overwrite=True, gene_id2hog_id_file=None, fam_hog_pthfam_ans_file=None):
		'''
		if HOG ids have been computed during the orthoxml file generation, overwrite can be set to False if implicit (leaf) HOGs are described in a mapper (hog_id2prot_ids) 
		'''
		assert (self.mode in {'w', 'a'}), 'Database must be opened in write mode.'

		# build taxonomy table except the SpeOff column
		print("initiate taxonomy table")
		tax_id2tax_off, species = self.initiate_tax_tab(stree_file)

		print("parse orthoXML file")
		tree_str = pyham.utils.get_newick_string(stree_file, type="nwk")
		ham_analysis = pyham.Ham(tree_str, orthoxml_file, use_internal_name=True)

		print("parse species and proteins")
		prot_id2prot_off = self.parse_species_and_proteins(ham_analysis)

		print("add SpeOff and TaxOff columns in taxonomy and species tables, respectively")
		self.add_speoff_col()
		self.add_taxoff_col()

		print("parse families and HOGs")
		if overwrite:
			fam_id2hog_ids, hog_id2prot_ids, hog_id2tax_id = self.parse_families_and_hogs(ham_analysis, overwrite)
		else:
			with open(gene_id2hog_id_file, 'r') as inf:
			    gene_id2hog_id = dict(map(lambda x: x.rstrip().split(), inf.readlines()))

			fam_id2hog_ids, hog_id2prot_ids, hog_id2tax_id = self.parse_families_and_hogs(ham_analysis, overwrite, gene_id2hog_id)

		# mapper HOG to taxon offset
		hog_id2tax_off = {h: tax_id2tax_off.get(t, -1) for h, t in hog_id2tax_id.items()}

		# mapper to prot_offs instead of prot_ids
		hog_id2prot_offs = {hog_id: set(map(lambda x: prot_id2prot_off[x], prot_ids)) for hog_id, prot_ids in hog_id2prot_ids.items()}

		print("fill family and HOG tables")
		self.update_hog_and_fam_tabs(fam_id2hog_ids, hog_id2tax_off, hog_id2prot_offs)

		# add family and hog offsets
		print("complete protein table")
		self.update_prot_tab(hog_id2prot_offs)

		print("compute LCA taxa")
		self.add_lcataxoff_col()

		if aln_files:
			print("load sequence buffer")
			with open(pthfam_an2prot_file, 'r') as inf:
				pthfam_an_id2prot_id = dict(map(lambda x: x.rstrip().split(), inf.readlines()))
			self.load_sequence_buffer(aln_files, pthfam_an_id2prot_id)

		# fill panther ancestral node ids to HOG table
		if fam_hog_pthfam_ans_file:
			def split_line(line):
				x = line.rstrip().split()
				return (x[1], ':'.join([x[2], x[3].split(';')[0]]))

			with open(fam_hog_pthfam_ans_file, 'r') as inf:
				hog_id2pthfam_an_id = dict(map(lambda x: split_line(x), inf.readlines()))

			pthfam_an_col = list(map(lambda x: hog_id2pthfam_an_id[x.decode('ascii')].encode('ascii'), self._hog_tab.col('ID')))
			self._hog_tab.modify_column(colname="OmaID", column=pthfam_an_col)

		# close and open in read mode
		self.db.close()
		self.mode = 'r'
		self.db = tables.open_file(self.file, self.mode, filters=self._compr)

	def parse_species_and_proteins(self, ham_analysis):

		# bookkeeping for later
		prot_id2prot_off = {}

		# store rows for species and protein tables
		spe_rows = []
		prot_rows = []

		# pointer to protein in protein table
		prot_off = 0
		curr_prot_off = prot_off

		# need to be sorted for later (add_speoff_col)
		for spe_off, genome in enumerate(sorted(ham_analysis.get_list_extant_genomes(), key=lambda x: x.name)):

			for gene in genome.genes:
				# skip singletons (they are problematic with the update_prot_tab function)
				if not gene.is_singleton():
					prot_id = gene.prot_id.encode('ascii')
					prot_rows.append((prot_id, 0, 0, spe_off, 0, 0))
					prot_id2prot_off[prot_id] = prot_off
					prot_off += 1

			spe_rows.append((genome.name.encode('ascii'), curr_prot_off, prot_off - curr_prot_off, 0))
			curr_prot_off = prot_off

		# fill species and protein tables
		self._sp_tab.append(spe_rows)
		self._sp_tab.flush()
		self._prot_tab.append(prot_rows)
		self._prot_tab.flush()

		return prot_id2prot_off
	    
	@staticmethod
	def parse_families_and_hogs(ham_analysis, overwrite=True, gene_id2hog_id=None):
	    
	    def _parse_hogs(hog, hog_id, fam_id2hog_ids, fam_id, hog_id2tax_id, hog_id2prot_ids, overwrite, gene_id2hog_id):
	        '''
	        traverse a family top-down and parse key informations
	        option to overwrite internal HOG ids
	        create leaf HOG ids from member of paralogous groups
	        '''
	        subhog_id = 1

	        for child in hog.children:

	            # Genes and leaf HOGs (only implicit in orthoXML)
	            if isinstance(child, pyham.Gene):

	                # create new HOG id if the gene arose by duplication
	                if child.arose_by_duplication:

	                    # option to retain original HOG ids (through mapper because implicit in orthoxml format)
	                    if overwrite:
	                        tmp_hog_id = '{}.{}'.format(hog_id, subhog_id)
	                    else:
	                        tmp_hog_id = gene_id2hog_id[child.unique_id]

	                    # store new HOG and its taxon (=species)
	                    fam_id2hog_ids[fam_id].add(tmp_hog_id.encode('ascii'))
	                    hog_id2tax_id[tmp_hog_id.encode('ascii')] = child.genome.name.encode('ascii')

	                    subhog_id += 1
	                else:
	                    tmp_hog_id = hog_id

	                # store protein id 
	                hog_id2prot_ids[tmp_hog_id.encode('ascii')].add(child.prot_id.encode('ascii'))

	            # Internal HOGs
	            elif child.arose_by_duplication:

	                # option to retain original HOG ids
	                if overwrite:
	                    tmp_hog_id = '{}.{}'.format(hog_id, subhog_id)
	                else:
	                    child_hog_id = child.hog_id
	                    if child_hog_id:
	                        tmp_hog_id = child_hog_id
	                    
	                    # to skip intermediate HOGs that have no name
	                    else:
	                        assert len(child.children)==1, 'intermediate HOGs should have a single child'
	                        tmp_hog = child.children[0]
	                        
	                        # either we reach a gene (leaf HOG) or a named HOG
	                        while not isinstance(tmp_hog, pyham.Gene) and not tmp_hog.hog_id:
	                            assert len(tmp_hog.children)==1, 'intermediate HOGs should have a single child\nchildren: {}'.format(gene.children)
	                            tmp_hog = tmp_hog.children[0]
	                        
	                        # leaf HOG case
	                        if isinstance(tmp_hog, pyham.Gene):
	                            tmp_hog_id = gene_id2hog_id[tmp_hog.unique_id]
	                        
	                        # internal HOG case
	                        else:
	                            tmp_hog_id = tmp_hog.hog_id

	                # store new HOG and its taxon
	                fam_id2hog_ids[fam_id].add(tmp_hog_id.encode('ascii'))  # fam ids are stored in integer. could change
	                hog_id2tax_id[tmp_hog_id.encode('ascii')] = child.genome.name.encode('ascii')

	                _parse_hogs(child, tmp_hog_id, fam_id2hog_ids, fam_id, hog_id2tax_id, hog_id2prot_ids, overwrite, gene_id2hog_id)    
	                subhog_id += 1
	            else:
	                _parse_hogs(child, hog_id, fam_id2hog_ids, fam_id, hog_id2tax_id, hog_id2prot_ids, overwrite, gene_id2hog_id)

	    roothog_id = 1  # used if overwrite

	    fam_id2hog_ids = collections.defaultdict(set)
	    hog_id2prot_ids = collections.defaultdict(set)
	    hog_id2tax_id = {}

	    for fam in ham_analysis.get_list_top_level_hogs():
	        fam_id = roothog_id if overwrite else fam.hog_id

	        # root HOG first (= family)
	        fam_id2hog_ids[fam_id].add(fam_id.encode('ascii'))
	        hog_id2tax_id[fam_id.encode('ascii')] = fam.genome.name.encode('ascii')

	        _parse_hogs(fam, fam_id, fam_id2hog_ids, fam_id, hog_id2tax_id, hog_id2prot_ids, overwrite, gene_id2hog_id)
	        
	        roothog_id += 1 # used if overwrite
	    
	    return fam_id2hog_ids, hog_id2prot_ids, hog_id2tax_id

	def load_sequence_buffer(self, aln_files, pthfam_an_id2prot_id):

		prot_id2prot_off = dict(zip(map(lambda x: x.decode('ascii'), self._prot_tab.col('ID')), range(self._prot_tab.nrows)))

		seq_buff = ""
		seq_off = 0

		seqoff_col = [0] * self._prot_tab.nrows
		seqlen_col = [0] * self._prot_tab.nrows

		for af in aln_files:
			pthfam_id = af.split('/')[-1].rstrip('.AN.fasta')

			for rec in SeqIO.parse(af, 'fasta'):
				pthfam_an_id = '{}:{}'.format(pthfam_id, rec.id)

				# if the panther .tree file as not been parsed, pthfam_an_id2prot_id will lack some pthfam_an_ids
				if pthfam_an_id in pthfam_an_id2prot_id:
					prot_id = pthfam_an_id2prot_id[pthfam_an_id]
					
					# singletons are not stored in prot_id2prot_off
					if prot_id in prot_id2prot_off:
						prot_off = prot_id2prot_off[prot_id]

						seq = "{} ".format(rec.seq.ungap('-'))
						seq_len = len(seq)
						seq_buff += seq

						seqoff_col[prot_off] = seq_off
						seqlen_col[prot_off] = seq_len

						seq_off += seq_len

		# replace the empty columns
		self._prot_tab.modify_column(colname="SeqOff", column=seqoff_col)
		self._prot_tab.modify_column(colname="SeqLen", column=seqlen_col)

		# add sequence buffer
		self._seq_buff.append(np.frombuffer(seq_buff.encode('ascii'), dtype=tables.StringAtom(1)))
		self._seq_buff.flush()


class DatabaseFromFasta(Database):
	'''
	describe the required Fasta format which is quite tricky
	'''
	def __init__(self, path, root_taxon, name=None):
		super().__init__(path, root_taxon, name)

	### main function ###
	def build_database(self, proteome_path, stree_path):

		assert (self.mode in {'w', 'a'}), 'Database must be opened in write mode.'

		# build taxonomy table except the SpeOff column
		print(" - initiate taxonomy table")
		tax2taxoff, species = self.initiate_tax_tab(stree_path)

		# build sequence buffer ans species table. initiate protein table and 
		# book-keeps mapping between families and hogs, hogs and proteins and tax
		print(" - fill sequence buffer, species table and initiate protein table")
		fam2hogs, hog2protoffs, hog2tax = self.parse_proteome_path(proteome_path, species)

		print(len(fam2hogs))
		print(len(hog2tax))
		print(len(hog2protoffs))

		print("add SpeOff and TaxOff columns in taxonomy and species tables, respectively")
		self.add_speoff_col()
		self.add_taxoff_col()

		# mapper HOG to taxon offset
		# exception of opistokonta...
		hog2taxoff = {h: tax2taxoff.get(t, -1) for h, t in hog2tax.items()}

		# fill hog and family tables
		print(" - fill family and HOG tables")
		self.update_hog_and_fam_tabs(fam2hogs, hog2taxoff, hog2protoffs)

		# complete protein table with family and hog offsets
		print(" - complete protein table")
		self.update_prot_tab(hog2protoffs)

		print("compute LCA taxa")
		self.add_lcataxoff_col()

		# close and open in read mode
		self.db.close()
		self.mode = 'r'
		self.db = tables.open_file(self.file, self.mode, filters=self._compr)	

	### functions to build database from proteomes in fastas either from scratch or updating the database ###
	def parse_proteome_path(self, proteome_path, species):
		'''
		parse a path full of proteomes in fasta format
		'''
		assert (self.mode in {'w', 'a'}), 'Database must be opened in write mode.'

		def _parse_proteome(fasta_record, prot_rows, fam2hogs, hog2protoffs, hog2tax, spe_off, prot_off):
			'''
			deals with a single proteome fasta file
			'''
			# store sequence buffer
			seq_buff = ""

			seq_off = len(self._seq_buff)

			for rec in fasta_record:

				# sequence
				seq = "{} ".format(rec.seq)
				seq_len = len(seq)
				seq_buff += seq

				# store protein row
				prot_rows.append((rec.id, seq_off, seq_len, spe_off, 0, 0))

				# book keeping for later
				ms_hog, taxonomy = rec.description.split()[1].encode('ascii'), rec.description.split()[2]
				hogs = self.parse_hogs(ms_hog)
				fam = int(hogs[0])

				fam2hogs[fam].update(hogs)
				hog2protoffs[ms_hog].add(prot_off)
				hog2tax.update(dict(zip(hogs, [x.encode('ascii') for x in taxonomy.split('_')])))
				
				# update offset of protein sequence in buffer and of protein raw in table
				seq_off += seq_len
				prot_off += 1

			# append to sequence buffer
			self._seq_buff.append(np.frombuffer(seq_buff.encode('ascii'), dtype=tables.StringAtom(1)))
			self._seq_buff.flush()

			return prot_off

		# get file names
		fa_exts = {'fa', 'fasta'}
		prot_files = sorted(list(filter(lambda fn: fn.split('.')[-1].lower() in fa_exts, os.listdir(proteome_path))))

		assert len(species) == len(prot_files), 'The number of fasta file does not match the number of species \n {} species for {} files'.format(len(species), len(prot_files))

		# store rows of protein and species table
		prot_rows = []
		spe_rows = []

		# book keeping for the functions below
		fam2hogs = collections.defaultdict(set)
		hog2protoffs = collections.defaultdict(set)
		hog2tax = dict()

		# keep track of current number of species and proteins
		spe_off = len(self._sp_tab)
		prot_off = len(self._prot_tab)

		for pf in tqdm(prot_files, desc='Adding species'):
			species = '.'.join(pf.split('.')[:-1])
			pf = os.path.join(proteome_path, pf)

			# parse one proteome
			new_prot_off = _parse_proteome(SeqIO.parse(pf, 'fasta'), prot_rows, fam2hogs, 
				hog2protoffs, hog2tax, spe_off, prot_off)

			# store species row
			spe_rows.append((species, prot_off, new_prot_off - prot_off, 0))

			spe_off += 1
			prot_off = new_prot_off

		# fill protein and species tables
		self._prot_tab.append(prot_rows)
		self._prot_tab.flush()
		self._sp_tab.append(spe_rows)
		self._sp_tab.flush()

		return fam2hogs, hog2protoffs, hog2tax