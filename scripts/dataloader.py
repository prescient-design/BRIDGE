import torch
import pickle

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.data import Data

import yaml

# Load config.yaml
with open('config.yaml', 'r') as file:
    yaml_input = yaml.safe_load(file)
# Storage
storage_location = yaml_input['storage']

# To put everything on GPU
if(torch.cuda.is_available()):
    use_device = 'cuda:'+str(torch.cuda.current_device())
    torch.multiprocessing.set_start_method('spawn')
else:
    use_device = 'cpu'

class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)

class dispense_entry(InMemoryDataset):
    '''
    In case of classification, we are treating 'y' as an integer.
    '''
    def __init__(self, pckl):
        super().__init__()
        self.data = pckl
        self.keys = list(pckl.keys())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        _id = self.keys[index]

        entry = self.data[_id]

        x1 = torch.tensor(entry['seq_1'], device=use_device, dtype=torch.int64)
        x2 = torch.tensor(entry['seq_2'], device=use_device, dtype=torch.int64)

        graph_1 = Data( x = x1, edge_index = torch.tensor(entry['adj_1'], device=use_device, dtype=torch.int64) )
        graph_2 = Data( x = x2, edge_index = torch.tensor(entry['adj_2'], device=use_device, dtype=torch.int64) )
        
        y = BipartiteData(x_s=x1, x_t=x2, edge_index=torch.tensor(entry['adj_1_2'], device=use_device, dtype=torch.int64))

        return graph_1, graph_2, y, self.keys[index]

def get_data(train_batchsize, validation_batchsize):
    '''
    '''
    train_pckl = pickle.load( open(storage_location+'pickles/dips+_gnn_train.pckl','rb') )
    valid_pckl = pickle.load( open(storage_location+'pickles/dips+_gnn_val.pckl','rb') )

    train_loader = DataLoader(dispense_entry(train_pckl), batch_size=train_batchsize, shuffle=True, drop_last=True )
    valid_loader = DataLoader(dispense_entry(valid_pckl), batch_size=validation_batchsize, shuffle=False, drop_last=True )

    return train_loader, valid_loader