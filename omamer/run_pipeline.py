import omamer

#########################################################################################################################################################################
### OMAmer 

def run_OMAmerSearch_pipeline(
    out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, score, cum_mode,
    overwrite_flatsearch=False, overwrite_search=False):
    
    db = omamer.database.DatabaseFromOMA(out_path, root_taxon=root_taxon, min_prot_nr=min_prot_nr, max_prot_nr=max_prot_nr)
    if db.mode != 'r':
        db.build_database(oma_hdf5_path, stree_path)

    # k-mer table
    ki = omamer.IndexValidation(db, k=6, reduced_alphabet=False, stree_path=stree_path, hidden_taxon=hidden_taxon)
    if ki.mode != 'r':
        ki.build_kmer_table()

    fs = omamer.FlatSearchValidation(ki, query_species=query_sp)
    if overwrite_flatsearch:
        fs.clean()
        fs = omamer.FlatSearchValidation(ki, query_species=query_sp)
    if fs.mode != 'r':
        fs.flat_search()

    se = omamer.Search(fs, score=score, cum_mode=cum_mode)
    if overwrite_flatsearch or overwrite_search:
        se.clean()
        se = omamer.Search(fs, score=score, cum_mode=cum_mode)
    if se.mode != 'r':
        se.search(top_n=10)

    del db, ki, fs
    return se

def run_OMAmerFamilyValidation_pipeline(
    out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, score, cum_mode, 
    neg_root_taxon, family_method_species2thresholds, method2prob, 
    overwrite_flatsearch=False, overwrite_search=False, overwrite_neg_search=False, overwrite_validation=False):
    
    se = omamer.run_OMAmerSearch_pipeline(
        out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, score, cum_mode, 
        overwrite_flatsearch, overwrite_search)
    
    va = omamer.OMAmerValidationFamily(se, neg_root_taxon=neg_root_taxon)
    if overwrite_neg_search:
        va.clean()
        va = omamer.OMAmerValidationFamily(se, neg_root_taxon=neg_root_taxon)
    elif overwrite_validation:
        va.clean_vf()
        va = omamer.OMAmerValidationFamily(se, neg_root_taxon=neg_root_taxon)
    if va.mode != 'r':
        va.validate(family_method_species2thresholds[('omamer_{}_{}'.format(score, cum_mode), query_sp)], method2prob['omamer_{}_{}'.format(score, cum_mode)], stree_path, oma_hdf5_path)

    del se
    return va

def run_OMAmerSubfamilyValidation_pipeline(
    out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, score, cum_mode,
    val_mode, subfamily_method_species2thresholds, method2prob, 
    overwrite_flatsearch=False, overwrite_search=False, overwrite_validation=False):
    
    se = run_OMAmerSearch_pipeline(
        out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, score, cum_mode,
        overwrite_flatsearch, overwrite_search)
    
    va = omamer.Validation(se, val_mode=val_mode)
    if overwrite_validation:
        va.clean()
        va = omamer.Validation(se, val_mode=val_mode)
    if va.mode != 'r':
        hog2taxbin, tax2taxbin = va.bin_taxa(stree_path, root_taxon, va.db._tax_tab.col('ID'), va.db._tax_tab[:], 
                                             query_sp, va.db._hog_tab.col('TaxOff'), bin_num=2, root=True, merge_post_lca_taxa=True)
        va.validate(subfamily_method_species2thresholds[('omamer_{}_{}'.format(score, cum_mode), query_sp)], hog2taxbin, x_num=2, prob=method2prob['omamer_{}_{}'.format(score, cum_mode)])
        
    del se
    return va

#########################################################################################################################################################################
### DIAMOND

def initiate_DIAMONDsearchValidation(
    out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp):

    db = omamer.DatabaseFromOMA(out_path, root_taxon=root_taxon, min_prot_nr=min_prot_nr, max_prot_nr=max_prot_nr)
    if db.mode != 'r':
        db.build_database(oma_hdf5_path, stree_path)

    ki = omamer.DIAMONDindexValidation(db, stree_path=stree_path, hidden_taxon=hidden_taxon)
    ki.export_reference_fasta()
    
    fs = omamer.DIAMONDsearchValidation(ki, query_species=query_sp)
    if fs.mode != 'r':
        fs.export_query_fasta()
    fs.clean()

    del db, ki, fs

def run_DIAMONDsearchValidation_pipeline(
    out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp):

    db = omamer.DatabaseFromOMA(out_path, root_taxon=root_taxon, min_prot_nr=min_prot_nr, max_prot_nr=max_prot_nr)
    if db.mode != 'r':
        db.build_database(oma_hdf5_path, stree_path)
    
    ki = omamer.DIAMONDindexValidation(db, stree_path=stree_path, hidden_taxon=hidden_taxon)
    
    # could open in append mode like BLAST search but inheritance issue I think
    fs = omamer.DIAMONDsearchValidation(ki, query_species=query_sp)

    if fs.mode != 'r':
        fs.export_query_fasta()
        fs.import_blast_result(score='evalue', prob=True)
    
    del db, ki
    return fs

def initiate_DIAMONDfamilyValidation(
    out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, neg_root_taxon):
    
    se = omamer.run_DIAMONDsearchValidation_pipeline(
        out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp)
    
    va = omamer.DIAMONDvalidationFamily(se, neg_root_taxon=neg_root_taxon)
    va.initiate_negative()
    va.clean()
    
    del se, va

def run_DIAMONDfamilyValidation_pipeline(
    out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, 
    neg_root_taxon, family_method_species2thresholds, overwrite_validation=False):
    
    se = omamer.run_DIAMONDsearchValidation_pipeline(
        out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp)
    
    va = omamer.DIAMONDvalidationFamily(se, neg_root_taxon=neg_root_taxon)
    if overwrite_validation:
        va.clean()
        va = omamer.DIAMONDvalidationFamily(se, neg_root_taxon=neg_root_taxon)
    if va.mode != 'r':
        va.initiate_negative()
        va.validate(family_method_species2thresholds[('diamond', query_sp)])
    
    del se
    return va

def run_DIAMONDsubfamilyValidation_pipeline(
    out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, 
    val_mode, subfamily_method_species2thresholds, overwrite_validation=False):

    fs = omamer.run_DIAMONDsearchValidation_pipeline(
        out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp)

    va = omamer.ValidationCS(fs, val_mode=val_mode)
    if overwrite_validation:
        va.clean()
        va = omamer.ValidationCS(fs, val_mode=val_mode)        
    if va.mode != 'r':
        hog2taxbin, tax2taxbin = va.bin_taxa(stree_path, root_taxon, va.db._tax_tab.col('ID'), va.db._tax_tab[:], 
                                             query_sp, va.db._hog_tab.col('TaxOff'), bin_num=2, root=True, merge_post_lca_taxa=True)
        va.validate(subfamily_method_species2thresholds[('diamond', query_sp)], hog2taxbin, x_num=2, prob=True)

    del fs
    return va
    
#########################################################################################################################################################################
### Smith-Waterman

def run_SWsearchValidation_pipeline(
    out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, overwrite_search=False):
    
    db = omamer.DatabaseFromOMA(out_path, root_taxon=root_taxon, min_prot_nr=min_prot_nr, max_prot_nr=max_prot_nr)
    if db.mode != 'r':
        db.build_database(oma_hdf5_path, stree_path)

    ki = omamer.SWindexValidation(db, stree_path, hidden_taxon)

    fs = omamer.SWsearchValidation(ki, query_sp)
    if overwrite_search:
        fs.clean()
    fs = omamer.SWsearchValidation(ki, query_sp)
    if fs.mode != 'r':
        fs.get_closest(oma_hdf5_path)

    del db, ki
    return fs

def run_SWfamilyValidation_pipeline(
    out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, 
    family_method_species2thresholds, overwrite_search=False, overwrite_validation=False):

    fs = omamer.run_SWsearchValidation_pipeline(
        out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, overwrite_search)
    
    va = omamer.SWvalidationFamily(fs)
    if overwrite_validation:
        va.clean()
    va = omamer.SWvalidationFamily(fs)
    if va.mode != 'r':
        va.validate(family_method_species2thresholds[('sw', query_sp)], False)   
        
    del fs
    return va

def run_SWsubfamilyValidation_pipeline(
    out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp, 
    val_mode, subfamily_method_species2thresholds, overwrite_validation=False):

    fs = omamer.run_SWsearchValidation_pipeline(
        out_path, root_taxon, min_prot_nr, max_prot_nr, oma_hdf5_path, stree_path, hidden_taxon, query_sp)

    va = omamer.ValidationCS(fs, val_mode=val_mode)
    if overwrite_validation:
        va.clean()
        va = omamer.ValidationCS(fs, val_mode=val_mode)
    if va.mode != 'r':
        hog2taxbin, tax2taxbin = va.bin_taxa(stree_path, root_taxon, va.db._tax_tab.col('ID'), va.db._tax_tab[:], 
                                             query_sp, va.db._hog_tab.col('TaxOff'), bin_num=2, root=True, merge_post_lca_taxa=True)
        va.validate(subfamily_method_species2thresholds[('sw', query_sp)], hog2taxbin, x_num=2, prob=False)  # bitscore

    del fs
    return va
    





