from ete3 import Tree, TreeStyle, NodeStyle, TextFace, ImgFace, add_face_to_node
from property_manager import cached_property
import itertools

'''
TO DO:
 - take as input directly some validation or search objects (clean code from notebook: 2020.05.25_visu_and_case_studies)
 - automatize image coloring (while still relying on custom images)
'''

'''
POLISH:
 - margins (legend and ref when placed queries)
 - deal with root and add nodes
 - maybe HOG info should be displayed on the branch-right location
 - more intuitive truth feature (e.g. simili diverging branch?)
'''

class VisHOGgraph():
    def __init__(self, tree_str, height):
        
        self.tree = Tree(tree_str, format=1)
                
        # query protein img faces
        self.query_features = ['truth', 'sw', 'omamer']
        self.qf_nr = len(self.query_features)
        self.img_width = 1000
        self.img_mid_margin = self.img_width / 10
        self.img_right_margin = self.img_mid_margin * 10
        self.img_left_margin = (self.qf_nr * self.img_width + (self.qf_nr - 1) * 2 * self.img_mid_margin + self.img_right_margin) / 2
        
        # reference HOG text faces
        self.ref_features = ['ID', 'taxon']
        self.font_size = self.img_width / 2.5
        self.text_margin_top = self.img_width / 5
        
        # tree style
        self.line_width = self.img_width / 12.5
        self.height = height
        self.branch_vertical_margin = 0
        
        
    @cached_property
    def basic_ns(self):
        ns = NodeStyle()
        ns['shape'] = 'square'#'circle'
        ns['size'] = 0 #self.line_width * 5
        ns['fgcolor'] = 'Grey'
        ns['vt_line_width'] = self.line_width
        ns['vt_line_color'] = "Grey"
        ns['hz_line_color'] = "Grey"
        ns['hz_line_width'] = self.line_width
        return ns
    
    def plot(self, hog2id, hog2taxon, hog2qfeat2queries, img_path, qf2shape, outfile="%%inline"):          
            
        def _add_query_faces(self, hog2qfeat2queries, hog_off, all_queries, img_path, qf2shape, node):
            '''
            to visualize queries in HOGs
            either true vs. predicted locations or only predicted with an option for another dimension (e.g. genome)
            '''
            qfeat2queries = hog2qfeat2queries.get(hog_off, None)
            
            if qfeat2queries:
                # first dimension is query features
                for i, qf in enumerate(self.query_features):
                    qf_queries = qfeat2queries[qf]

                    # second is queries
                    for j, q in enumerate(all_queries):

                        # get the image
                        if q in qf_queries:
                            img_file = '{}{}_{}.png'.format(img_path, qf2shape[qf], j + 1)

                        # get the empty image (TO DO: option to stack results)
                        else:
                            img_file = '{}empty.png'.format(img_path)

                        # create image face
                        imgf = ImgFace(img_file, width = self.img_width)

                        # get left margin
                        if i == 0:
                            imgf.margin_left = self.img_left_margin
                        else:
                            imgf.margin_left = self.img_mid_margin

                        # get right margin
                        if i == (len(self.query_features) - 1):
                            imgf.margin_right = self.img_right_margin
                        else:
                            imgf.margin_right = self.img_mid_margin

                        # add face
                        add_face_to_node(imgf, node, column=i, position="branch-top")
        
        def _add_text_face(self, text, node, fstyle):
            tf = TextFace(text, tight_text=True, fsize=self.font_size, fstyle=fstyle)
            tf.margin_top = self.text_margin_top
            tf.margin_left = self.img_left_margin + 50
            add_face_to_node(tf, node, column=0, position="branch-bottom")            
        
        def get_all_queries(self, hog2qfeat2queries):
            all_queries = set()
            for node in self.tree.traverse():
                if node.name != '-1':
                    qf2queries = hog2qfeat2queries.get(int(node.name), None)
                    if qf2queries:
                        all_queries.update(itertools.chain(*qf2queries.values()))
            assert len(all_queries) <= 3, 'not enough colors'
            return all_queries
        
        all_queries = get_all_queries(self, hog2qfeat2queries)

        def _layout(node):
            
            if node.name != '-1':

                # node style
                node.set_style(self.basic_ns)

                # plot reference features
                hog_off = int(node.name)
                for rf in self.ref_features:

                    # HOG id
                    if rf == 'ID':
                        hog_id = 'HOG {}'.format(hog2id[hog_off])
                        _add_text_face(self, hog_id, node, "normal")

                    # taxon
                    elif rf == 'taxon':
                        taxon = hog2taxon[hog_off]
                        _add_text_face(self, taxon, node, "italic")

                # plot query features
                _add_query_faces(self, hog2qfeat2queries, hog_off, all_queries, img_path, qf2shape, node)

        # tree style
        ts = TreeStyle()
        ts.layout_fn = _layout
        ts.show_leaf_name = False
        ts.show_scale = False
        ts.optimal_scale_level = 'full'
        ts.branch_vertical_margin = self.branch_vertical_margin
        return self.tree.render(outfile, tree_style=ts, h = self.height)


# this is my old tree visualization for single query predictions with features such as k-mer counts, score, etcs. # more for sanity check
# from property_manager import cached_property, lazy_property
# from ete3 import Tree, TreeStyle, NodeStyle, TextFace, BarChartFace, add_face_to_node
# import numpy as np
# import hierarchy

# class TreeVis():

# 	def __init__(self, va_lca, va_cs = None):

# 		assert va_lca or va_cs, 'At least one Validation object must be provided'
# 		if va_lca:
# 			assert va_lca.db.mode == 'r', 'Database must be opened in read mode.'
# 			assert va_lca.ki.mode == 'r', 'Index must be opened in read mode.'
# 			assert va_lca.fs.mode == 'r', 'FlatSearch must be opened in read mode.'
# 			assert va_lca.se.mode == 'r', 'Search must be opened in read mode.'
# 			assert va_lca.mode == 'r', 'LCA Validation must be opened in read mode.'
# 		# if va_cs:
# 		# 	assert va_cs.db.mode == 'r', 'Database must be opened in read mode.'
# 		# 	assert va_cs.ki.mode == 'r', 'Index must be opened in read mode.'
# 		# 	assert va_cs.fs.mode == 'r', 'FlatSearch must be opened in read mode.'
# 		# 	assert va_cs.se.mode == 'r', 'Search must be opened in read mode.'
# 		# 	assert va_cs.mode == 'r', 'LCA Validation must be opened in read mode.'

# 		self.db = va_lca.db
# 		self.ki = va_lca.ki
# 		self.fs = va_lca.fs
# 		self.se = va_lca.se
# 		self.va_lca = va_lca

# 		### main param for tree
# 		# tree style
# 		self.show_leaf_name = False
# 		self.show_scale = False
# 		self.scale = 150
# 		self.branch_vertical_margin = -40
# 		self.tree_margin_top = 10
# 		self.tree_margin_bottom = 10
# 		self.tree_margin_right= 10
# 		self.tree_margin_left = 10

# 		# node style
# 		self.node_size = 5
# 		self.line_width = 2

# 		# bar face
# 		self.bar_width = 20
# 		self.bar_height = 80
# 		self.bar_labels = False
# 		self.bar_label_fsize = 310
# 		self.bar_scale_fsize = 5
# 		self.bar_margin_right = 10
# 		self.bar_margin_left = 10
# 		self.bar_margin_top = 10
# 		self.bar_margin_bottom= 10

# 		# text face
# 		self.text_margin_bottom = 5
# 		self.text_margin_left = 5
# 		self.text_fsize = 10

# 		# image
# 		self.h = 1000
# 		self.w = 1000
# 		self.name = '%%inline'

# 	# node styles
# 	@cached_property
# 	def tp_ns(self):
# 		tp_ns = NodeStyle()
# 		tp_ns['size'] = self.node_size
# 		tp_ns['fgcolor'] = '#1b9e77'
# 		tp_ns['vt_line_type'] = 2
# 		tp_ns['vt_line_width'] = self.line_width
# 		tp_ns['hz_line_type'] = 0
# 		tp_ns['hz_line_color'] = '#1b9e77'
# 		tp_ns['hz_line_width'] = self.line_width
# 		return tp_ns

# 	@cached_property
# 	def fp_ns(self):
# 		fp_ns = NodeStyle()
# 		fp_ns['size'] = self.node_size
# 		fp_ns['fgcolor'] = '#d95f02'
# 		fp_ns['vt_line_type'] = 2
# 		fp_ns['vt_line_width'] = self.line_width
# 		fp_ns['hz_line_color'] = "#d95f02"
# 		fp_ns['hz_line_width'] = self.line_width
# 		return fp_ns

# 	@cached_property
# 	def fn_ns(self):
# 		fn_ns = NodeStyle()
# 		fn_ns['size'] = self.node_size
# 		fn_ns['fgcolor'] = '#7570b3'
# 		fn_ns['vt_line_type'] = 2
# 		fn_ns['vt_line_width'] = self.line_width
# 		fn_ns['hz_line_color'] = "#7570b3"
# 		fn_ns['hz_line_width'] = self.line_width
# 		return fn_ns

# 	@cached_property
# 	def tn_ns(self):
# 		tn_ns = NodeStyle()
# 		tn_ns['size'] = self.node_size
# 		tn_ns['fgcolor'] = 'Black'
# 		tn_ns['vt_line_type'] = 2
# 		tn_ns['vt_line_width'] = self.line_width
# 		tn_ns['hz_line_color'] = "Black"
# 		tn_ns['hz_line_width'] = self.line_width
# 		return tn_ns

# 	# cached tables and arrays

# 	@cached_property
# 	def fam_tab(self):
# 		return self.db._fam_tab[:]

# 	@cached_property
# 	def hog_tab(self):
# 		return self.db._hog_tab[:]

# 	@cached_property
# 	def level_arr(self):
# 		return self.db._level_arr[:]

# 	@cached_property
# 	def chog_buff(self):
# 		return self.db._chog_arr[:]

# 	@cached_property
# 	def cprot_buff(self):
# 		return self.db._cprot_arr[:]

# 	# lazy mappers

# 	@lazy_property
# 	def hog2count(self):
# 		return self.ki._hog_count[:]

# 	@lazy_property
# 	def hog2ccount(self):
# 		return self.se.hog_cum_count[:]

# 	@lazy_property	
# 	def hog2id(self):
# 		return self.hog_tab['ID']

# 	@lazy_property
# 	def hog2taxid(self):
# 		return self.db._tax_tab[:][self.hog_tab['TaxOff']]['ID']

# 	@lazy_property
# 	def hog2lcataxid(self):
# 		return self.db._tax_tab[:][self.hog_tab['LCAtaxOff']]['ID']	

# 	@staticmethod
# 	def traverse_hog_nwk(hog_off, hog_tab, chog_buff):
	    
# 	    def _traverse_hog_nwk(hog_off, hog_tab, chog_buff):
	    
# 	        if (hog_tab[hog_off]['ChildrenHOGoff'] == -1):
# 	            return str(hog_off)

# 	        tmp = []
# 	        for chog in hierarchy._children_hog(hog_off, hog_tab, chog_buff):
# 	            tmp.append(str(_traverse_hog_nwk(chog, hog_tab, chog_buff)))

# 	        return '({}){}'.format(','.join(tmp), hog_off)
	    
# 	    return '{};'.format(_traverse_hog_nwk(hog_off, hog_tab, chog_buff))


# 	def plot(self, qoff, bar_mapper_offs, text_mapper_offs, tresh_off, log10=True, w=None, h=None):

# 		# gather HOG offsets of predicted family
# 		fam_off = self.se._res_tab[qoff]['FamOff']
# 		fam_ent = self.db._fam_tab[fam_off]
# 		hog_off = fam_ent['HOGoff']
# 		hog_num = fam_ent['HOGnum']
# 		hog_offsets = np.arange(hog_off, hog_off + hog_num, dtype=np.uint64)

# 		# get mappers for bars
# 		bar_mappers = []
# 		bar_names = []
# 		for bo in bar_mapper_offs:
# 			if bo == 0:
# 				bar_mappers.append(self.hog2count)
# 				bar_names.append('HOG k-mer count')
# 			elif bo == 1:
# 				bar_mappers.append(self.hog2ccount)
# 				bar_names.append('HOG cum. k-mer count')
# 			elif bo == 2:
# 				bar_mappers.append(self.fs._queryHog_count[qoff])
# 				bar_names.append('query-HOG k-mer count')
# 			elif bo == 3:
# 				bar_mappers.append(self.se.cumulate_counts_nfams(
# 					self.fs._queryHog_count[qoff], self.fam_tab, self.level_arr, self.hog_tab['ParentOff'], self.se.cumulate_counts_1fam, self.se._sum, self.se._max))
# 				bar_names.append('query-HOG cum. k-mer count')
# 			elif bo == 4:
# 				bar_names.append('normalized score')
# 				pass

# 		# find out max k-mer count to normalize bar plots
# 		max_count = np.max(np.array(list(map(lambda x: x[hog_offsets], bar_mappers))))
# 		max_count = np.log10(max_count) if log10 else max_count

# 		# get mappers for text
# 		text_mappers = []
# 		for to in text_mapper_offs:
# 			if to == 0:
# 				text_mappers.append(self.hog2id)
# 			elif to == 1:
# 				text_mappers.append(self.hog2taxid)
# 			elif to == 2:
# 				text_mappers.append(self.hog2lcataxid)
# 			elif to == 3:
# 				text_mappers.append(self.se._hog_score[qoff])

# 		# compute tree
# 		t = Tree(self.traverse_hog_nwk(hog_off, self.hog_tab, self.chog_buff), format=1)

# 		# mappers for nodes
# 		hog2tp = self.va_lca._tp_pre[qoff, :, tresh_off]
# 		hog2fp = self.va_lca._fp[qoff, :, tresh_off]
# 		hog2fn = self.va_lca._fn[qoff, :, tresh_off]

# 		# layout function for nodes
# 		def layout(node):

# 			# bar plots
# 			kmer_counts = list(map(lambda x: x[int(node.name)], bar_mappers))
# 			kmer_counts = np.log10(kmer_counts) if log10 else kmer_counts
# 			bar = BarChartFace(kmer_counts, width = self.bar_width * len(bar_mappers), height=self.bar_height, max_value=max_count)
# 			if self.bar_labels:
# 				bar.labels = bar_names
# 				bar.label_fsize = self.bar_label_fsize
# 			bar.scale_fsize = self.bar_scale_fsize
# 			bar.margin_right = self.bar_margin_right
# 			bar.margin_bottom = self.bar_margin_bottom
# 			bar.margin_top = self.bar_margin_top
# 			bar.margin_left = self.bar_margin_left
# 			add_face_to_node(bar, node, column=0, position="branch-bottom")

# 			# textfaces
# 			for tm in text_mappers:
# 				x = tm[int(node.name)]
# 				text = TextFace(x.decode('ascii') if isinstance(x, bytes) else round(x, 2), tight_text=True)
# 				text.margin_bottom = self.text_margin_bottom
# 				text.margin_right = 0
# 				text.margin_top = 0
# 				text.margin_left = self.text_margin_left
# 				text.fsize = self.text_fsize
# 				add_face_to_node(text, node, column=0, position="branch-top")

# 			# node styles
# 			if hog2tp[int(node.name)]:
# 				node.set_style(self.tp_ns)
# 			elif hog2fp[int(node.name)]:
# 				node.set_style(self.fp_ns)
# 			elif hog2fn[int(node.name)]:
# 				node.set_style(self.fn_ns)
# 			else:
# 				node.set_style(self.tn_ns)

# 		# create tree style with layout
# 		ts = TreeStyle()
# 		ts.show_leaf_name = self.show_leaf_name
# 		ts.show_scale = self.show_scale
# 		ts.scale = self.scale
# 		ts.branch_vertical_margin = self.branch_vertical_margin
# 		ts.layout_fn = layout
# 		ts.margin_top = self.tree_margin_top
# 		ts.margin_bottom = self.tree_margin_bottom
# 		ts.margin_right = self.tree_margin_right
# 		ts.margin_left = self.tree_margin_left

# 		if self.h:
# 			return t.render(self.name, h = h if h else self.h, tree_style=ts)
# 		elif self.w:
# 			print(self.w)
# 			return t.render(self.name, w = w if w else self.w, tree_style=ts)
# 		else:
# 			return t.render(self.name, tree_style=ts)

# 	def plot_true_family(self, qoff, w=None, h=None):
# 		'''
# 		when predicted family is wrong and we want to look at the true family
# 		'''
# 		# gather HOG offsets of true family
# 		fam_off = self.db._prot_tab[self.se._res_tab[qoff]['QueryOff']]['FamOff']
# 		fam_ent = self.db._fam_tab[fam_off]
# 		hog_off = fam_ent['HOGoff']
# 		hog_num = fam_ent['HOGnum']
# 		hog_offsets = np.arange(hog_off, hog_off + hog_num, dtype=np.uint64)

# 		# gather true HOGs
# 		true_hog_off = self.db._prot_tab[self.se._res_tab[qoff]['QueryOff']]['HOGoff']
		
# 		# get mappers for text
# 		text_mappers = [self.hog2id, self.hog2taxid, self.hog2lcataxid]

# 		# compute tree
# 		t = Tree(self.traverse_hog_nwk(hog_off, self.hog_tab, self.chog_buff), format=1)

# 		# layout function for nodes
# 		def layout(node):

# 			# textfaces
# 			for tm in text_mappers:
# 				x = tm[int(node.name)]
# 				text = TextFace(x.decode('ascii') if isinstance(x, bytes) else round(x, 2), tight_text=True)
# 				text.margin_bottom = self.text_margin_bottom
# 				text.margin_right = 0
# 				text.margin_top = 5
# 				text.margin_left = self.text_margin_left
# 				text.fsize = self.text_fsize
# 				add_face_to_node(text, node, column=0, position="branch-top")

# 			if true_hog_off == int(node.name):
# 				node.set_style(self.tp_ns)
# 			else:
# 				node.set_style(self.tn_ns)

# 		# create tree style with layout
# 		ts = TreeStyle()
# 		ts.show_leaf_name = self.show_leaf_name
# 		ts.show_scale = self.show_scale
# 		ts.scale = self.scale
# 		ts.branch_vertical_margin = self.branch_vertical_margin
# 		ts.layout_fn = layout
# 		ts.margin_top = self.tree_margin_top
# 		ts.margin_bottom = self.tree_margin_bottom
# 		ts.margin_right = self.tree_margin_right
# 		ts.margin_left = self.tree_margin_left

# 		if self.h:
# 			return t.render(self.name, h = h if h else self.h, tree_style=ts)
# 		elif self.w:
# 			print(self.w)
# 			return t.render(self.name, w = w if w else self.w, tree_style=ts)
# 		else:
# 			return t.render(self.name, tree_style=ts)


# 	def test_plot(self):
# 		t = Tree( "((4,5)3,2)1;", format=1)
# 		(t & "1").set_style(self.tp_ns)
# 		(t & "2").set_style(self.tn_ns)
# 		(t & "3").set_style(self.tp_ns)
# 		(t & "4").set_style(self.fp_ns)
# 		(t & "5").set_style(self.fn_ns)

# 		text = TextFace('Metazoa', tight_text=True)
# 		text.margin_bottom = 5
# 		text.fsize = 10
# 		(t & "1").add_face(text, column=0, position="branch-top")

# 		bar = BarChartFace([20, 30, 10, 24], width = 50, height=50, max_value=50)
# 		bar.scale_fsize = 5
# 		bar.labels = ['raw k-mer counts      ', 'cumulated counts', 'shared', '3232']
# 		bar.label_fsize = 3
# 		bar.margin_right = 5
# 		(t & "2").add_face(bar, column=0, position="branch-bottom")

# 		return t.render("%%inline", h = self.h, tree_style=self.ts)


# def reconstruct_seq(prot_off, prot_tab, seq_buff):
#     return ''.join(np.array(seq_buff[prot_tab[prot_off]['SeqOff']: prot_tab[prot_off]['SeqOff'] + prot_tab[prot_off]['SeqLen']], dtype=str)[:-1])

# def splitseq(seq, k):
#     return set([ seq[i:(i+k)] for i in range( len(seq)-k+1) ])
