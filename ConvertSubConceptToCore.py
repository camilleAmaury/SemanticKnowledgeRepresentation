import pandas as pd
import numpy as np

"""
    Function which convert the label of subconcepts into core concepts labels
"""
def convert_sub_concepts_to_core(ontology, abstract_concepts_file):
    # open files
    concepts = pd.read_csv(abstract_concepts_file, sep=",", header=0).values
    
    # create dictionnary of core concepts
    core_concepts = set(concepts[:,1])
    core_concepts = {k:i for i,k in enumerate(core_concepts)}
    
    # create dictionnary of subconcept to coreconcept
    sub_to_core = concepts[:,1]
    for i in range(ontology.shape[0]):
        ontology[i,1] = core_concepts[sub_to_core[ontology[i,1]]]
    
    return ontology, core_concepts
    
    
    