import yaml
import os

import torch
import numpy

from torch_geometric.data import InMemoryDataset, Data

from torch.utils.data import Dataset

from model import BridgeEncoder

def device():
    # To put everything on GPU
    if(torch.cuda.is_available()):
        use_device = 'cuda:'+str(torch.cuda.current_device())
        #torch.multiprocessing.set_start_method('spawn')
    else:
        use_device = 'cpu'
    return use_device
use_device = device()

def load_config():
    # Load config.yaml
    with open('config.yaml', 'r') as file:
        yaml_input = yaml.safe_load(file)
    # Storage
    return yaml_input
yaml_input = load_config()

def load_BRIDGE():
    model = BridgeEncoder.load_from_checkpoint(yaml_input['BRIDGE_location']).to(use_device)
    # disable randomness, dropout, etc...
    model.eval()
    return model

def make_meta():
    if not os.path.exists('meta'):
        os.makedirs('meta')

def negative_log_10(norm_arr):
    norm_arr = -numpy.log10(norm_arr)
    return norm_arr

class dispense_entry(InMemoryDataset):
    '''
    In case of classification, we are treating 'y' as an integer.
    '''
    def __init__(self, pckl):
        super().__init__()
        self.data = pckl
        self.keys = list(pckl.keys())
        self.device = device()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        _id = self.keys[index]

        entry = self.data[_id]

        x1 = torch.tensor(entry['seq_1'], device=use_device, dtype=torch.int64)
        x2 = torch.tensor(entry['seq_2'], device=use_device, dtype=torch.int64)

        graph_1 = Data( x = x1, edge_index = torch.tensor(entry['adj_1'], device=use_device, dtype=torch.int64) )
        graph_2 = Data( x = x2, edge_index = torch.tensor(entry['adj_2'], device=use_device, dtype=torch.int64) )
        
        y = Data(x1=x1, x2=x2, edge_index=torch.tensor(entry['adj_1_2'], device=use_device, dtype=torch.int64))

        #information = entry['information']
        information = entry['affinity']

        return [graph_1, graph_2, y, self.keys[index], information]
    
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

# Single entry to extract embeddings
class SingleDataEntry():
    def __init__(self, x, edge_index):
        self.x = torch.tensor(x, device=use_device, dtype=torch.int64)
        self.edge_index = torch.tensor(edge_index, device=use_device, dtype=torch.int64)

class SimpleDispenseEntry(Dataset):
    def __init__(self, pckl):
        super().__init__()
        
        self.data = pckl
        self.keys = list(pckl.keys())
        self.device = device()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        _id = self.keys[index]

        X, y = self.data[_id]

        X = torch.tensor(X, device=use_device)
        y = torch.tensor(y, device=use_device, dtype=torch.float)

        return _id, X, y

class SimpleDispenseEntryWithExampleWeights(Dataset):
    def __init__(self, pckl):
        super().__init__()
        
        self.data = pckl
        self.keys = list(pckl.keys())
        self.device = device()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        _id = self.keys[index]

        X, y, w = self.data[_id]

        X = torch.tensor(X, device=use_device)
        y = torch.tensor(y, device=use_device, dtype=torch.float)
        w = torch.tensor(w, device=use_device, dtype=torch.float)

        return _id, X, y, w