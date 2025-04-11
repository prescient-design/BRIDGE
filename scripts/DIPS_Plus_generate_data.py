import os
import multiprocessing
import multiprocessing.dummy as mp

import numpy
import torch

import dill as pickle

import yaml
from tqdm import tqdm
import logging

from packman.constants import THREE_LETTER_TO_ONE_LETTER
from scipy.spatial import Delaunay


# Initializations
if not os.path.exists('meta'):
    os.makedirs('meta')

# Load config.yaml
with open('config.yaml', 'r') as file:
    yaml_input = yaml.safe_load(file)
# Storage
storage_location = yaml_input['storage']

# Distance cutoff after Delaunay calculation instead of Alpha (Experimental)
DIST_CUTOFF = float(yaml_input['DIST_CUTOFF'])
logging.basicConfig(filename='meta/logs.log', level=logging.INFO)
logging.info(str(DIST_CUTOFF)+' Cutoff selected')

# Residue to int
class ResidueToInt(torch.nn.Module):
    def __init__(self, all_token_elements):
        '''Automatic sorting is included.
        '''
        super().__init__()
        self.all_token_elements = sorted(all_token_elements)
        self.aa_to_int_token = {i:numi for numi,i in enumerate(self.all_token_elements)}

    
    def forward(self, seq):
        '''Given a sequence in form of three letter codes, return the token vector.
        '''
        seq_to_int = []
        for i in seq:
            try:
                seq_to_int.append( self.aa_to_int_token[i] )
            except:
                seq_to_int.append( self.aa_to_int_token['UNK'] )
        return torch.tensor(seq_to_int, dtype=torch.int8)


def single_entry_process(entry):

    # Load single pickle
    try:
        with open(storage_location+'dips+/raw/'+entry, 'rb') as f:
            fh = pickle.load(f)
    except IsADirectoryError:
        print(entry+' is a dir.')
        return

    seq_1_calpha = fh.df0[fh.df0['atom_name']=='CA']
    seq_2_calpha = fh.df1[fh.df1['atom_name']=='CA']

    # Sequences
    seq_1 = seq_1_calpha.resname.to_numpy()
    seq_2 = seq_2_calpha.resname.to_numpy()
    # Function to convert str to torch.int8 to save some space and make one hot encoding easier later.
    residue_to_int = ResidueToInt(THREE_LETTER_TO_ONE_LETTER.keys())
    seq_1 = residue_to_int(seq_1)
    seq_2 = residue_to_int(seq_2)

    chain1_calpha_location = list( seq_1_calpha[['x','y','z']].to_numpy() )
    chain2_calpha_location = list( seq_2_calpha[['x','y','z']].to_numpy() )

    all_calpha_locations = chain1_calpha_location + chain2_calpha_location
    all_calpha_locations = numpy.array([list(i) for i in all_calpha_locations])

    DT = Delaunay(all_calpha_locations)

    # Adjacency
    chain_1_edge_list = []
    chain_2_edge_list = []
    chain_1_2_edge_list = []

    for i in DT.simplices:
        sorted_i = sorted(i)
        # Decomposing a tessellation into edges.
        for j in range(0,len(sorted_i)):
            for k in range(j+1,len(sorted_i)):
                # In chain 1 and distance less than cutoff
                distance_between_calpha = numpy.linalg.norm( all_calpha_locations[sorted_i[j]] - all_calpha_locations[sorted_i[k]] )
                if( (sorted_i[j] < len(chain1_calpha_location) and sorted_i[k] < len(chain1_calpha_location) ) and distance_between_calpha <= DIST_CUTOFF  ):
                    #chain1_adj[sorted_i[j]][sorted_i[k]] = 1
                    chain_1_edge_list.append( (sorted_i[j], sorted_i[k]) )
                # In chain 2 and distance less than cutoff
                elif( (sorted_i[j] >= len(chain1_calpha_location) and sorted_i[k] >= len(chain1_calpha_location) ) and distance_between_calpha <= DIST_CUTOFF  ):
                    #chain2_adj[sorted_i[j]-len(chain1_calpha_location)][sorted_i[k]-len(chain1_calpha_location)] = 1
                    chain_2_edge_list.append( (sorted_i[j]-len(chain1_calpha_location), sorted_i[k]-len(chain1_calpha_location)) )
                # Inter chain connections
                elif(distance_between_calpha <= DIST_CUTOFF):
                    #chain_1_2_adj[sorted_i[j]][sorted_i[k]-len(chain1_calpha_location)] = 1
                    chain_1_2_edge_list.append( (sorted_i[j], sorted_i[k]-len(chain1_calpha_location)) )
    
    # Remove duplicates
    chain_1_edge_list = list(set(chain_1_edge_list))
    chain_2_edge_list = list(set(chain_2_edge_list))
    chain_1_2_edge_list = list(set(chain_1_2_edge_list))

    single_entry_data = { 'seq_1': numpy.array(seq_1),
    'seq_2' : numpy.array(seq_2),
    'adj_1' : numpy.array(chain_1_edge_list).T,
    'adj_2' : numpy.array(chain_2_edge_list).T,
    'adj_1_2' : numpy.array(chain_1_2_edge_list).T}

    return (entry, single_entry_data)


def main():
    
    pool = mp.Pool( multiprocessing.cpu_count() )

    # Training
    train_entries = open(storage_location+'dips+/raw/pairs-postprocessed-train.txt')
    input = [i.strip() for i in train_entries]

    data = {}
    for result in tqdm(pool.imap_unordered( single_entry_process, input ), total=len(input)):
        if(result!=None):
            data[result[0]] = result[1]
    
    with open(yaml_input['storage']+'pickles/dips+_gnn_train.pckl', 'wb') as handle:
        pickle.dump(data, handle)
    logging.basicConfig(filename='meta/logs.log', level=logging.INFO)
    logging.info('train dataset generated.')

    # Validation
    train_entries = open(storage_location+'dips+/raw/pairs-postprocessed-val.txt')
    input = [i.strip() for i in train_entries]

    data = {}
    for result in tqdm(pool.imap_unordered( single_entry_process, input ), total=len(input)):
        if(result!=None):
            data[result[0]] = result[1]
    with open(yaml_input['storage']+'pickles/dips+_gnn_val.pckl', 'wb') as handle:
        pickle.dump(data, handle)
    logging.basicConfig(filename='meta/logs.log', level=logging.INFO)
    logging.info('val dataset generated.')

    return True


if(__name__=='__main__'):
    main()