
import multiprocessing.dummy as mp
import multiprocessing

import csv
import pickle
from tqdm import tqdm
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR


import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from torch.nn.utils.rnn import pad_sequence

import numpy

from scipy.spatial import Delaunay
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error


import wandb
from matplotlib import pyplot as plt

from packman import molecule
from packman.constants import THREE_LETTER_TO_ONE_LETTER

from utils import load_config, ResidueToInt, load_BRIDGE, SingleDataEntry, SimpleDispenseEntry, SimpleDispenseEntryWithExampleWeights, device, make_meta

# Set up
use_device = device()

# Config
yaml_input = load_config()
DIST_CUTOFF = float(yaml_input['DIST_CUTOFF'])

# Load the main model
BRIDGE = load_BRIDGE()
BRIDGE.eval()

# Model
class regression(pl.LightningModule):

    def __init__(self, num_features=1400000, lr=3e-4, gamma=0.99):
        super(regression, self).__init__()

        self.num_features = num_features
        self.lr = lr
        self.lr_gamma = gamma
        self.Best_Train_Loss = float('Inf')
        self.Best_Val_Loss = float('Inf')

        # Loss
        #self.loss = torch.nn.MSELoss(reduction='none')
        self.loss = torch.nn.HuberLoss(reduction='none')

        # Network
        self.network = torch.nn.Sequential(
        torch.nn.Linear(self.num_features, 50, bias=False, device=use_device),
        torch.nn.LayerNorm(50, device=use_device),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.05),
        torch.nn.Linear(50, 50, bias=False, device=use_device),
        torch.nn.LayerNorm(50, device=use_device),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.05),
        torch.nn.Linear(50, 1, device=use_device)
        )

    def forward(self, features):
        features = torch.unsqueeze(features, 1)
        x_out = self.network(features)
        return torch.squeeze(x_out)
    
    def training_step(self, batch):
        id, X, y, w = batch
        x_out = self.forward(X)

        loss = self.loss(x_out, y)
        new_loss = torch.mean(loss * w)
        
        self.log('train_loss', new_loss )
        return new_loss
    
    def validation_step(self, batch):
        id, X, y, w = batch
        x_out = self.forward(X)

        loss = self.loss(x_out, y)

        new_loss = torch.mean(loss * w)
        
        self.log('val_loss', new_loss)
        
        if(new_loss < self.Best_Val_Loss):
            cpu_y = y.cpu().numpy()
            cpu_x_out = x_out.cpu().numpy()

            max_of_all = numpy.max( (cpu_y, cpu_x_out) )

            plt.scatter(cpu_y, cpu_x_out, edgecolors='k', s=7, alpha=0.7)
            plt.xlim(0, max_of_all+(max_of_all/20))
            plt.ylim(0, max_of_all+(max_of_all/20))
            plt.xlabel('Experimental Affinity -log10(Kd)')
            plt.ylabel('Predicted Affinity -log10(Kd)')

            plt.text(0.05, max_of_all-(max_of_all/20), 'Val Points = %3i' % len(cpu_y))
            plt.text(0.05, max_of_all-(max_of_all/10), 'Val Pearson R = %0.3f' % pearsonr(cpu_y, cpu_x_out).statistic)
            plt.text(0.05, max_of_all-1.5*(max_of_all/10), 'Val R² Score = %0.3f' % r2_score(cpu_y, cpu_x_out))
            plt.text(0.05, max_of_all-2*(max_of_all/10), 'Val RMSD = %0.3f' % numpy.sqrt(  ( numpy.sum((cpu_y-cpu_x_out)**2) / len(cpu_y) )  ) )
            plt.text(0.05, max_of_all-2.5*(max_of_all/10), 'Val MAE = %0.3f' % mean_absolute_error(cpu_y, cpu_x_out))
            
            plt.savefig('images/affinity_regression.png', dpi=300)
            plt.close()

        self.log('val_pearsonr', pearsonr(cpu_y, cpu_x_out).statistic )
        self.log('val_r2_score', r2_score(cpu_y, cpu_x_out) )
        self.log('val_delta_std', numpy.std(cpu_x_out-cpu_y))
        
        return new_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": ExponentialLR(optimizer, gamma=self.lr_gamma)} }

def single_entry_process(entry, test=False):
    
    if(test == True):
        try:
            model = molecule.load_pdb(yaml_input['storage']+'Piercelab/'+entry[0])[int(entry[1])]
        except ValueError:
            model = molecule.load_cif(yaml_input['storage']+'Piercelab/'+entry[0])[int(entry[1])]
    else:
        try:
            model = molecule.load_pdb(yaml_input['storage']+'all_structures/imgt/'+entry[0])[int(entry[1])]
        except FileNotFoundError:
            print(entry,'failed. | file not found')
            return

    Ab_chains = []
    Ag_chains = [i.strip() for i in entry[4].split('|')]

    if(entry[2] != 'NA'): Ab_chains.append(entry[2])
    if(entry[3] != 'NA'): Ab_chains.append(entry[3])

    ab_residues = []
    ag_residues = []
    
    for i in Ab_chains:
        try:
            ab_residues.extend([j for j in model[i].get_calpha() if j is not None])
        except:
            continue
    
    if(ab_residues == []):
        print(entry,'failed. | ab chains not found')
        return
    
    try:
        for i in Ag_chains:
            ag_residues.extend([j for j in model[i].get_calpha() if j is not None])
    except:
        print(entry,'failed. | ag chain not found')
        return

    # Check if ab or ag is empty
    if(ab_residues==[]):
        print(entry,'failed. | ab residues empty')
        return
    if(ag_residues==[]):
        print(entry,'failed. | ag residues empty')
        return 
    
    
    ab_calpha_location = [i.get_location() for i in ab_residues]
    ag_calpha_location = [i.get_location() for i in ag_residues]

    all_calpha_locations = ab_calpha_location + ag_calpha_location
    all_calpha_locations = numpy.array([list(i) for i in all_calpha_locations])

    DT = Delaunay(all_calpha_locations)

    # Adjacency
    ab_edge_list = []
    ag_edge_list = []
    ab_ag_edge_list = []

    for i in DT.simplices:
        sorted_i = sorted(i)
        # Decomposing a tessellation into edges.
        for j in range(0,len(sorted_i)):
            for k in range(j+1,len(sorted_i)):
                # In ab and distance less than cutoff
                distance_between_calpha = numpy.linalg.norm( all_calpha_locations[sorted_i[j]] - all_calpha_locations[sorted_i[k]] )
                if( (sorted_i[j] < len(ab_calpha_location) and sorted_i[k] < len(ab_calpha_location) ) and distance_between_calpha <= DIST_CUTOFF  ):
                    ab_edge_list.append( (sorted_i[j], sorted_i[k]) )
                # In ag and distance less than cutoff
                elif( (sorted_i[j] >= len(ab_calpha_location) and sorted_i[k] >= len(ab_calpha_location) ) and distance_between_calpha <= DIST_CUTOFF  ):
                    ag_edge_list.append( (sorted_i[j]-len(ab_calpha_location), sorted_i[k]-len(ab_calpha_location)) )
                # Inter group
                elif(distance_between_calpha <= DIST_CUTOFF):
                    ab_ag_edge_list.append( (sorted_i[j], sorted_i[k]-len(ab_calpha_location)) )
    
    # Remove duplicates
    ab_edge_list = numpy.array(list(set(ab_edge_list)))
    ag_edge_list = numpy.array(list(set(ag_edge_list)))
    ab_ag_edge_list = numpy.array(list(set(ab_ag_edge_list)))

    # Parse to model
    residue_to_int = ResidueToInt(THREE_LETTER_TO_ONE_LETTER.keys())
    seq_1 = residue_to_int([i.get_parent().get_name() for i in ab_residues])
    seq_2 = residue_to_int([i.get_parent().get_name() for i in ag_residues])

    ab_data_entry = SingleDataEntry(x=seq_1, edge_index=ab_edge_list)
    ag_data_entry = SingleDataEntry(x=seq_2, edge_index=ag_edge_list)

    try:
        out1, out2 = BRIDGE(ab_data_entry, ag_data_entry)
    except IndexError:
        print(entry,'failed. | Most likely synthetic construct. (Verify)')
        return

    dot_prod = torch.mm(out1, out2.T)
    padded_dot_prod = torch.zeros((822,1634), device=use_device)
    padded_dot_prod[:dot_prod.shape[0], :dot_prod.shape[1]] = dot_prod

    dot_prod = padded_dot_prod.detach().flatten().cpu().numpy()

    entry_key = entry[0].split('.')[0].upper()+'_'+''.join(Ab_chains)+':'+''.join(Ag_chains)

    affinity = -numpy.log10(float(entry[-1]))

    return ( entry_key, dot_prod, affinity )
    
def create_vectorized_data_from_pdb():
    input = [i for i in csv.reader(open('data/SAbDab-Affinity.csv'))]

    # Parallel
    pool = mp.Pool( multiprocessing.cpu_count() )
    data = {}
    max_len = 0
    for result in tqdm(pool.imap_unordered( single_entry_process, input[1:] ), total=len(input)):
        if(result!=None):
            data[result[0]] = (result[1], result[2])

            if( int(result[1].shape[0]) > max_len ):
                max_len = int(result[1].shape[0])
    
    # Serial
    '''
    data = {}
    max_len = 0
    for i in tqdm(input):
        result = single_entry_process(i)
        if(result is not None):
            data[result[0]] = (result[1], result[2])

            if( int(result[1].shape[0]) > max_len ):
                max_len = int(result[1].shape[0])
    '''

    with open('data/SAbDab_Affinity.pckl', 'wb') as handle:
        pickle.dump(data, handle)
    
    print('Total entires:', len(data), 'Max length (Flattened embeddings):', max_len)

def get_dataloaders(train_batch_size=64, validation_batch_size=16, training_proportion=0.9, deduplicate=True, instance_weighting=True):

    def collate_fn(data, max_len=1400000):
        id, tensors, targets, weights = zip(*data)
        #id, tensors, targets = zip(*data)
        tensors = list(tensors)

        tensors[0] = torch.nn.ConstantPad1d((0, max_len - tensors[0].shape[0]), 0)(tensors[0])

        features = pad_sequence(tensors, batch_first=True)

        targets = torch.stack(targets)

        weights = torch.stack(weights)

        return id, features, targets, weights

    with open('data/SAbDab_Affinity.pckl', 'rb') as f:
        data = pickle.load(f)

    all_keys = list(data.keys())
    n = len(all_keys)

    # Remove Piercelab entries
    piercelab_data = []
    for i in open('data/Piercelab-Affinity.tsv', 'r'):
        piercelab_data.append(i.strip().split('\t')[0][:4])

    
    not_piercelab = [i for i in all_keys if i[:4] not in piercelab_data]

    pdb_affinity = []
    for i in not_piercelab:
        pdb_affinity.append( (i.split('_')[0], data[i][1]) )

    counter =  Counter(pdb_affinity)

    counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=False)}


    # Deduplicating -- Problem with doing this is that antigens have no representation but are forced to be validated.
    if(deduplicate):
        train_pdbs = []
        train_pdbs_counter = 0
        valid_pdbs = []
        for i in counter.keys():
            if(train_pdbs_counter < len(not_piercelab)*training_proportion):
                train_pdbs.append( i[0] )
                train_pdbs_counter += counter[i]
            else:
                valid_pdbs.append( i[0] )

        train_pdbs = list(set(train_pdbs))
        valid_pdbs = list(set(valid_pdbs))
        
        train_list, valid_list = [], []
        for i in not_piercelab:
            if(i[:4] in train_pdbs):
                train_list.append(i)
            elif(i[:4] in valid_pdbs):
                valid_list.append(i)
            else:
                print('You should not be getting this message. Check for errors.')
        print('Deduplication is ON.')
    else:
        train_list = not_piercelab[:int(training_proportion*n)]
        valid_list = not_piercelab[int(training_proportion*n):]
        print('Deduplication is OFF.')

    make_meta()
    fh = open('meta/train-valid-splits.txt', 'w')
    fh.write( 'Train List:\t'+'\t'.join(train_list)+'\n' )
    fh.write( 'Valid List:\t'+'\t'.join(valid_list) )

    print( 'Train size:', len(train_list), 'Valid size:', len(valid_list) )

    # Weights for examples
    if(instance_weighting):
        PDB_FREQ = {i[0]:counter[i] for i in counter}
        train_pckl = {i: (data[i][0], data[i][1], PDB_FREQ[i[:4]]) for i in train_list}
        valid_pckl = {i: (data[i][0], data[i][1], PDB_FREQ[i[:4]]) for i in valid_list}
        print('Instance Weighting is ON.')
    else:
        train_pckl = {i: (data[i][0], data[i][1], 1) for i in train_list}
        valid_pckl = {i: (data[i][0], data[i][1], 1) for i in valid_list}
        print('Instance Weighting is OFF.')

    # Test
    train_loader = DataLoader(SimpleDispenseEntryWithExampleWeights(train_pckl), batch_size=train_batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn )
    valid_loader = DataLoader(SimpleDispenseEntryWithExampleWeights(valid_pckl), batch_size=validation_batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn )

    return train_loader, valid_loader

def train():
    train_data, valid_data = get_dataloaders(train_batch_size=32, validation_batch_size=16, training_proportion=0.8)

    #logger = pl.loggers.TensorBoardLogger('./tensorboard')
    wandb.init(entity=yaml_input['wandb_entity'], project='gnn_interface_affinity_pred')
    logger = WandbLogger(project='gnn_interface_affinity_pred')

    # 54000: number of features (flattned protein-protein interactions)
    model = regression(num_features=1400000, lr=0.00008, gamma=0.995)
    model.to(use_device)

    # Checkpointing
    checkpoints = 'meta/checkpoints'
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=checkpoints,
            filename="affinity_regression_{epoch:04d}-{val_loss:0.3f}-{val_pearsonr:0.3f}-{val_delta_std:0.3f}-{val_r2_score:0.3f}",
            monitor="val_loss",
            save_top_k=5,
            save_last=True,
            verbose=True)
    
    trainer = pl.Trainer(accelerator='gpu',
        devices=torch.cuda.device_count(),
        num_nodes=1,
        logger=logger,
        max_epochs=2000,
        gradient_clip_val=3,
        gradient_clip_algorithm="norm",
        callbacks=[ LearningRateMonitor("epoch"), checkpoint ],
        enable_progress_bar=False
        #overfit_batches = 1,
        #profiler="simple",
        #precision=16#,
        )
    
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=valid_data)

def test():
   #single_entry_process()
    def collate_fn(data, max_len=1400000):
        id, tensors, targets, weights = zip(*data)
        tensors = list(tensors)

        tensors[0] = torch.nn.ConstantPad1d((0, max_len - tensors[0].shape[0]), 0)(tensors[0])

        features = pad_sequence(tensors, batch_first=True)

        targets = torch.stack(targets)

        weights = torch.stack(weights)

        return id, features, targets, weights

    pdb_affinities = {}
    for i in open('data/Piercelab-Affinity.tsv'):
        templine = i.strip().split('\t')
        try:
            pdb_affinities[templine[0]] =  float(templine[2]) * float(1e-9)
            # Delta double g
            #pdb_affinities[templine[0]] =  -float(templine[3])
        except ValueError:
            continue

    input = []
    for i in open('data/Piercelab-Complexes.tsv').readlines()[1:]:
        templine = i.strip().split('\t')

        filename = templine[0][:4]

        # yaml_input['storage']+'Piercelab/'

        try:
            H, L = templine[0].split(':')[0].split('_')[1]
        except ValueError:
            H = templine[0].split(':')[0].split('_')[1]
            L = 'NA'
        
        AG = '|'.join( templine[0].split(':')[1].strip() )
        
        try:
            input.append( (filename+'.cif', 0, H, L, AG, pdb_affinities[filename]) )
        except KeyError:
            print(filename, 'affinity not present.')
    
    data = {}
    max_len = 0
    for i in tqdm(input):
        result = single_entry_process(i, test=True)
        if(result is not None):
            data[result[0]] = (result[1], result[2], 1)

            if( int(result[1].shape[0]) > max_len ):
                max_len = int(result[1].shape[0])

    
    DL = DataLoader(SimpleDispenseEntryWithExampleWeights(data), batch_size=len(data), shuffle=True, drop_last=True, collate_fn=collate_fn )

    model = regression.load_from_checkpoint(yaml_input['BRIDGE_Affinity_location']).to(use_device)
    # Stop training, disable dropout etc
    model.eval()

    for i in DL:
        id, x, y, w = i

        x_out = model.forward(x)

        cpu_y = y.cpu().numpy()
        cpu_x_out = x_out.detach().cpu().numpy()

        print(cpu_y, cpu_x_out)

        plt.scatter(cpu_y, cpu_x_out, edgecolors='k', s=7, alpha=0.7)
        plt.xlim(0, 15)
        plt.ylim(0, 15)
        plt.xlabel('Experimental Affinity -log10(Kd)')
        plt.ylabel('Predicted Affinity -log10(Kd)')

        plt.text(0.05, 14.5, 'Test Points = %3i' % len(cpu_y))
        plt.text(0.05, 14, 'Test Pearson R = %0.3f' % pearsonr(cpu_y, cpu_x_out).statistic)
        plt.text(0.05, 13.5, 'Test R² Score = %0.3f' % r2_score(cpu_y, cpu_x_out))
        plt.text(0.05, 13, 'Test RMSD = %0.3f' % numpy.sqrt(  ( numpy.sum((cpu_y-cpu_x_out)**2) / len(cpu_y) )  ) )
        plt.text(0.05, 12.5, 'Test MAE = %0.3f' % mean_absolute_error(cpu_y, cpu_x_out))
        
        plt.savefig('images/Piercelab_affinity_regression.png', dpi=300)
        plt.close()
    
    print('Total entires:', len(data), 'Max length (Flattened embeddings):', max_len)

def regenerate_validation_graph():
    def collate_fn(data, max_len=1400000):
        id, tensors, targets, weights = zip(*data)
        tensors = list(tensors)

        tensors[0] = torch.nn.ConstantPad1d((0, max_len - tensors[0].shape[0]), 0)(tensors[0])

        features = pad_sequence(tensors, batch_first=True)

        targets = torch.stack(targets)

        weights = torch.stack(weights)

        return id, features, targets, weights

    valid = open('models/train-valid-splits.txt').readlines()[1]
    split_file = valid.strip().split('\t')

    # Affinities
    pdb_affinities = {}
    
    for i in open('data/SAbDab-Affinity.csv'):
        templine = i.strip().split(',')
        try:
            pdb_affinities[templine[0]] =  float(templine[-1])
            # Delta double g
            #pdb_affinities[templine[0]] =  -float(templine[3])
        except ValueError:
            continue

    input = []
    for i in split_file[1:]:
        filename = i[:4].lower()+'.pdb'

        # yaml_input['storage']+'Piercelab/'

        try:
            H, L = i.split(':')[0].split('_')[1]
        except ValueError:
            H = i.split(':')[0].split('_')[1]
            L = 'NA'
        
        AG = '|'.join( i.split(':')[1].strip() )
        

        try:
            input.append( (filename, 0, H, L, AG, pdb_affinities[filename]) )
        except KeyError:
            print(filename, 'affinity not present.')
    
    data = {}
    max_len = 0
    for i in tqdm(input):
        result = single_entry_process(i)
        if(result is not None):
            data[result[0]] = (result[1], result[2], 1)

            if( int(result[1].shape[0]) > max_len ):
                max_len = int(result[1].shape[0])

    
    DL = DataLoader(SimpleDispenseEntryWithExampleWeights(data), batch_size=len(data), shuffle=True, drop_last=True, collate_fn=collate_fn )

    model = regression.load_from_checkpoint(yaml_input['BRIDGE_Affinity_location']).to(use_device)
    # Stop training, disable dropout etc
    model.eval()

    for i in DL:
        id, x, y, w = i

        x_out = model.forward(x)

        cpu_y = y.cpu().numpy()
        cpu_x_out = x_out.detach().cpu().numpy()

        print(cpu_y, cpu_x_out)

        plt.scatter(cpu_y, cpu_x_out, edgecolors='k', s=7, alpha=0.7)
        plt.xlim(0, 15)
        plt.ylim(0, 15)
        plt.xlabel('Experimental Affinity -log10(Kd)')
        plt.ylabel('Predicted Affinity -log10(Kd)')

        plt.text(0.05, 14.5, 'Test Points = %3i' % len(cpu_y))
        plt.text(0.05, 14, 'Test Pearson R = %0.3f' % pearsonr(cpu_y, cpu_x_out).statistic)
        plt.text(0.05, 13.5, 'Test R² Score = %0.3f' % r2_score(cpu_y, cpu_x_out))
        plt.text(0.05, 13, 'Test RMSD = %0.3f' % numpy.sqrt(  ( numpy.sum((cpu_y-cpu_x_out)**2) / len(cpu_y) )  ) )
        plt.text(0.05, 12.5, 'Test MAE = %0.3f' % mean_absolute_error(cpu_y, cpu_x_out))
        
        plt.savefig('images/Piercelab_affinity_regression.png', dpi=300)
        plt.close()
    
    print('Total entires:', len(data), 'Max length (Flattened embeddings):', max_len)

def BRIDGE_UMAP():
    from umap import UMAP

    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    from scipy.stats import gaussian_kde

    train_data, valid_data = get_dataloaders()

    xs, ys = [], []
    for i in train_data:
        _id, local_x, local_y, w = i

        xs.append(local_x.detach().cpu().numpy())
        ys.extend(local_y.detach().cpu().numpy())
    
    for i in valid_data:
        _id, local_x, local_y, w = i

        xs.append(local_x.detach().cpu().numpy())
        ys.extend(local_y.detach().cpu().numpy())
    
    xs = numpy.vstack(xs)
    viscosity = ys

    plane_mapper = UMAP(n_jobs=1, n_components=2, n_neighbors=10, min_dist=0, metric='canberra', init='random', verbose=True, n_epochs=500, negative_sample_rate=20, random_state=42).fit_transform( xs, y=ys )

    x = plane_mapper[:, 0]
    y = plane_mapper[:, 1]

    # Perform kernel density estimation (KDE)
    xy = numpy.vstack([x, y])
    kde = gaussian_kde(xy)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Create a regular grid
    xi = numpy.linspace(xmin, xmax, 200)
    yi = numpy.linspace(ymin, ymax, 200)
    xi, yi = numpy.meshgrid(xi, yi)

    # Evaluate KDE on the grid
    density = kde(numpy.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

    # Smooth density grid for better contours
    density_smooth = gaussian_filter(density, sigma=1)

    # Connect density with tourf
    values_grid = griddata((x, y), viscosity, (xi, yi), method='linear')

    fig = plt.figure(figsize=(6,12))
    sp1 = fig.add_subplot(211)

    # Create the contour plot with color based on dominant values
    contour = sp1.contour(xi, yi, density_smooth, levels=6, cmap='gray', alpha=0.35, linestyles='--')  # Density contours
    #cbar = sp1.colorbar(contour, label="Point Density")

    plot = sp1.scatter( x, y, s=1, c=viscosity, cmap='seismic', alpha=0.75)#, edgecolors='black')
    sp1.scatter( x, y, s=1000, c=viscosity, cmap='seismic', alpha=0.01, edgecolor='none')#, edgecolors='black')
    
    #plt.colorbar(plot)
    sp1.title.set_text('UMAP Analysis')
    
    sp2 = fig.add_subplot(212)
    sp2.title.set_text('Affinity Distribution (-log10 transform)')

    plt.colorbar(plot)
    plt.hist(viscosity, edgecolor = 'black', color='lightblue')
    plt.savefig('images/affinity_umap_combined.png', dpi=300)
    plt.close()

def main():
    create_vectorized_data_from_pdb()

    train()

    #test()

    #regenerate_validation_graph()

    BRIDGE_UMAP()
    
    return True


if(__name__=='__main__'):
    main()