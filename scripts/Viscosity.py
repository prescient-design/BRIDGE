
import numpy

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.nn.utils.rnn import pad_sequence

import pickle
import random

from scipy.spatial import Delaunay
from scipy.stats import spearmanr, sem
from sklearn.metrics import r2_score, mean_absolute_error

from packman import molecule
from packman.constants import THREE_LETTER_TO_ONE_LETTER

import os
import argparse
import wandb
from matplotlib import pyplot as plt

from utils import load_config, ResidueToInt, load_BRIDGE, SingleDataEntry, SimpleDispenseEntry, device, make_meta

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

    def __init__(self, num_features=54000, lr=3e-4, gamma=0.99):
        super(regression, self).__init__()

        self.num_features = num_features
        self.lr = lr
        self.lr_gamma = gamma
        self.Best_Train_Loss = float('Inf')
        self.Best_Val_Loss = float('Inf')

        # Loss
        self.loss = torch.nn.MSELoss()

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
        id, X, y = batch
        x_out = self.forward(X)

        loss = self.loss(x_out, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        id, X, y = batch
        x_out = self.forward(X)

        loss = self.loss(x_out, y)
        self.log('val_loss', loss)
        if(loss < self.Best_Val_Loss):
            cpu_y = y.cpu().numpy()
            cpu_x_out = x_out.cpu().numpy()

            # Plotting
            '''
            plt.scatter(cpu_y, cpu_x_out, edgecolors='k', s=7, alpha=0.7)
            plt.xlim(0, 3)
            plt.ylim(0, 3)
            plt.xlabel('Experimental Viscosity log10(cP)')
            plt.ylabel('Predicted Viscosity log10(cP)')

            plt.text(0.05, 2.9, 'Val Points = %3i' % len(cpu_y))
            plt.text(0.05, 2.8, 'Val Spearman R = %0.3f' % spearmanr(cpu_y, cpu_x_out).statistic)
            plt.text(0.05, 2.7, 'Val R² Score = %0.3f' % r2_score(cpu_y, cpu_x_out))
            plt.text(0.05, 2.6, 'Val RMSD = %0.3f' % numpy.sqrt(  ( numpy.sum((cpu_y-cpu_x_out)**2) / len(cpu_y) )  ) )
            plt.text(0.05, 2.5, 'Val MAE = %0.3f' % mean_absolute_error(cpu_y, cpu_x_out))
            
            plt.savefig('images/viscosity_regression.png', dpi=300)
            plt.close()
            '''
        
        self.log('y', y)
        self.log('pred_y', x_out)
        #self.log('val_spearmanr', spearmanr(cpu_y, cpu_x_out).statistic )
        #self.log('val_r2_score', r2_score(cpu_y, cpu_x_out) )
        #self.log('val_delta_std', numpy.std(cpu_x_out-cpu_y))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": ExponentialLR(optimizer, gamma=self.lr_gamma)} }

def create_vectorized_data_from_pdb():
    
    def single_entry(filename, viscosity):
        
        mol = molecule.load_pdb(yaml_input['storage']+'/viscosity_structures/'+filename+'.pdb')

        h_calpha = [residue.get_calpha() for residue in mol[0]['H'].get_residues() if residue is not None]
        l_calpha = [residue.get_calpha() for residue in mol[0]['L'].get_residues() if residue is not None]

        all_calpha = h_calpha + l_calpha
        all_calpha_locations = [calpha.get_location() for calpha in all_calpha]

        DT = Delaunay(all_calpha_locations)

        # Adjacency
        edge_list = []
        for simplex in DT.simplices:
            sorted_i = sorted(simplex)

            for j in range(0,len(sorted_i)):
                for k in range(j+1,len(sorted_i)):
                    distance_between_calpha = numpy.linalg.norm( all_calpha_locations[sorted_i[j]] - all_calpha_locations[sorted_i[k]] )
                    if(distance_between_calpha<=DIST_CUTOFF):
                        edge_list.append( (sorted_i[j], sorted_i[k]) )
        
        # Seq
        residue_to_int = ResidueToInt(THREE_LETTER_TO_ONE_LETTER.keys())
        seq = residue_to_int([calpha.get_parent().get_name() for calpha in all_calpha])

        edge_list = list(set(edge_list))
        edge_list =  numpy.array(edge_list)

        single_data_entry = SingleDataEntry(x=seq, edge_index=edge_list)

        out1, out2 = BRIDGE(single_data_entry, single_data_entry)

        dot_prod = torch.mm(out1, out2.T)

        # IMPORTANT IF SOMETHING IS GOING WRONG, CHECK BELOW FIRST
        return (filename, ( dot_prod.flatten().detach().cpu().numpy(), viscosity ) )

    # Ab21
    Ab21_data = {}
    for i in open('data/Ab21.tsv'):
        line = i.split('\t')
        try:
            out = single_entry( line[0], float(line[-1]) )
            # IgG1 filter
            if(out is not None and line[1][:4]=='IgG1'):
                Ab21_data[ out[0] ] = out[1]
            else:
                continue
        except ValueError:
            None
    
    with open('data/Ab21(Viscosity).pckl', 'wb') as handle:
        pickle.dump(Ab21_data, handle)

    # PDGF38
    PDGF38_data = {}
    for i in open('data/PDGF38_raw.csv'):
        line = i.split(',')
        try:
            out = single_entry( line[0], float(line[1]) )
            if(out is not None):
                PDGF38_data[ out[0] ] = out[1]
            else:
                continue
        except ValueError:
            None
    
    with open('data/PDGF38(Viscosity).pckl', 'wb') as handle:
        pickle.dump(PDGF38_data, handle)

    # Ab8
    Ab8_data = {}
    for i in open('data/Ab8.csv'):
        line = i.split(',')
        try:
            out = single_entry( line[0], float(line[1]) )
            if(out is not None):
                Ab8_data[ out[0] ] = out[1]
            else:
                continue
        except ValueError:
            None
        except FileNotFoundError:
            print('Error in', line[0], 'FileNotFoundError')
            continue
    
    with open('data/Ab8(Viscosity).pckl', 'wb') as handle:
        pickle.dump(Ab8_data, handle)
    
def get_dataloaders(leave_out_idx, train_batch_size=16, valid_batch_size=1):

    def collate_fn(data, max_len=54000):
        id, tensors, targets = zip(*data)
        tensors = list(tensors)

        tensors[0] = torch.nn.ConstantPad1d((0, max_len - tensors[0].shape[0]), 0)(tensors[0])

        features = pad_sequence(tensors, batch_first=True)

        targets = torch.stack(targets)
        return id, features, targets
    
    with open('data/Ab21(Viscosity).pckl', 'rb') as handle:
        Ab21 = pickle.load(handle)

    with open('data/PDGF38(Viscosity).pckl', 'rb') as handle:
        PDGF38 = pickle.load(handle)

    #with open('data/Ab8(Viscosity).pckl', 'rb') as handle:
    #    Ab8 = pickle.load(handle)

    # Ab21
    Ab21_list = list(sorted(Ab21.keys()))

    # Leave one out
    left_out = Ab21_list[leave_out_idx]
    Ab21_list = Ab21_list[:leave_out_idx] + Ab21_list[leave_out_idx+1:]
    
    n = len(Ab21_list)
    random.shuffle(Ab21_list)
    #50 / 50 train test division
    Ab21_train = Ab21_list#[:int(0.7*n)]
    #Ab21_valid = Ab21_list[int(0.7*n):]

    # PDGF38
    PDGF38_list = list(PDGF38.keys())
    m = len(PDGF38_list)
    random.shuffle(PDGF38_list)
    #80 / 20 train validation division
    PDGF38_train = PDGF38_list#[:int(train_proportion*m)]
    #PDGF38_valid = PDGF38_list[int(train_proportion*m):]

    # Ab8
    #Ab8_list = list(Ab8.keys())
    #o = len(Ab8_list)
    #random.shuffle(Ab8_list)
    #80 / 20 train validation division
    #Ab8_train = Ab8_list#[:int(train_proportion*o)]
    #Ab8_valid = Ab8_list[int(train_proportion*o):]

    train = {}
    train.update( {i:Ab21[i] for i in Ab21_train} )
    train.update( {i:PDGF38[i] for i in PDGF38_train} )
    #train.update( {i:Ab8[i] for i in Ab8_train} )

    valid = {}
    #valid.update( {i:Ab21[i] for i in Ab21_valid} )
    #valid.update( {i:PDGF38[i] for i in PDGF38_valid} )
    #valid.update( {i:Ab8[i] for i in Ab8_valid} )

    # Leave one out
    valid = {left_out: Ab21[left_out]}

    print('Train examples:', len(train), 'Valid examples:', len(valid))

    train_loader = DataLoader(SimpleDispenseEntry(train), batch_size=train_batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    valid_loader = DataLoader(SimpleDispenseEntry(valid), batch_size=valid_batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

    return train_loader, valid_loader

def train_LOOCV(leave_out_idx, seed):
    train_data, valid_data = get_dataloaders(leave_out_idx)

    #logger = pl.loggers.TensorBoardLogger('./tensorboard')
    wandb.init(entity=yaml_input['wandb_entity'], project='gnn_interface_viscosity_pred')
    logger = WandbLogger(project='gnn_interface_viscosity_pred')

    # 54000: number of features (flattned protein-protein interactions)
    model = regression(num_features=54000, lr=1e-2, gamma=0.999)
    model.to(use_device)
    model.log('leave_out_idx', leave_out_idx)
    model.log('seed', seed)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=200, verbose=False, mode="min")

    pre_name = "BRIDGE_viscosity_leave_out_idx={:02d}-seed={:04d}".format(leave_out_idx, seed)
    # Checkpointing
    checkpoints = 'meta/checkpoints'
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=checkpoints,
            #filename=pre_name+"-{epoch:04d}-{val_loss:0.3f}-{val_spearmanr:0.3f}-{val_delta_std:0.3f}-{val_r2_score:0.3f}",
            filename=pre_name+"-{epoch:04d}-{val_loss:0.3f}-{y:0.3f}-{pred_y:0.3f}",
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
        callbacks=[ LearningRateMonitor("epoch"), checkpoint, early_stop_callback ],
        enable_progress_bar=False
        )
    
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=valid_data)

def train():
    train_data, valid_data = get_dataloaders(train_batch_size=16, valid_batch_size=8)

    #logger = pl.loggers.TensorBoardLogger('./tensorboard')
    wandb.init(entity=yaml_input['wandb_entity'], project='gnn_interface_viscosity_pred')
    logger = WandbLogger(project='gnn_interface_viscosity_pred')

    # 54000: number of features (flattned protein-protein interactions)
    model = regression(num_features=54000, lr=0.01, gamma=0.99)
    model.to(use_device)

    # Checkpointing
    checkpoints = 'meta/checkpoints'
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=checkpoints,
            filename="viscosity_regression_{epoch:04d}-{val_loss:0.3f}-{val_spearmanr:0.3f}-{val_delta_std:0.3f}-{val_r2_score:0.3f}",
            monitor="val_loss",
            save_top_k=5,
            save_last=True,
            verbose=True)
    
    trainer = pl.Trainer(accelerator='gpu',
        devices=torch.cuda.device_count(),
        num_nodes=1,
        logger=logger,
        max_epochs=500,
        gradient_clip_val=3,
        gradient_clip_algorithm="norm",
        callbacks=[ LearningRateMonitor("epoch"), checkpoint ],
        enable_progress_bar=False,
        #overfit_batches = 1
        profiler="simple"
        #precision=16#,
        )
    
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=valid_data)

def BRIDGE_UMAP():
    from umap import UMAP

    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    from scipy.stats import gaussian_kde

    train_data, valid_data = get_dataloaders()

    xs, ys = [], []
    for i in train_data:
        local_x, local_y = i

        xs.append(local_x.detach().cpu().numpy())
        ys.extend(local_y.detach().cpu().numpy())
    
    for i in valid_data:
        local_x, local_y = i

        xs.append(local_x.detach().cpu().numpy())
        ys.extend(local_y.detach().cpu().numpy())
    
    xs = numpy.vstack(xs)
    viscosity = ys


    plane_mapper = UMAP(n_jobs=1, n_components=2, n_neighbors=15, min_dist=0, metric='canberra', init='random', verbose=True, n_epochs=500, negative_sample_rate=20, random_state=42).fit_transform( xs )

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
    contour = sp1.contour(xi, yi, density_smooth, levels=6, cmap='gray', alpha=0.5, linestyles='--')  # Density contours
    #cbar = sp1.colorbar(contour, label="Point Density")

    plot = sp1.scatter( x, y, s=1, c=viscosity, cmap='seismic', alpha=0.75)#, edgecolors='black')
    sp1.scatter( x, y, s=1000, c=viscosity, cmap='seismic', alpha=0.1, edgecolor='none')#, edgecolors='black')
    
    #plt.colorbar(plot)
    sp1.title.set_text('UMAP Analysis')
    
    sp2 = fig.add_subplot(212)
    sp2.title.set_text('Viscosity Distribution (-log10 transform)')

    plt.colorbar(plot)
    plt.hist(viscosity, edgecolor = 'black', color='lightblue')
    plt.savefig('images/viscosity_umap_combined.png', dpi=300)
    plt.close()

def after_sweep_analysis():
    leave_one_out = {}
    for i in os.listdir('meta/checkpoints'):
        local_dir = {}
        for j in i.split('.ckpt')[0].split('-'):
            try:
                key, val = j.split('=')
                local_dir[key] = float(val)
            except:
                break
        
        if(local_dir == {}):
            continue
        
        try:
            leave_one_out[ int(local_dir['BRIDGE_viscosity_leave_out_idx']) ].append( local_dir )
        except KeyError:
            leave_one_out[ int(local_dir['BRIDGE_viscosity_leave_out_idx']) ] = [local_dir]
    
    
    # Analysis starts (5 seeds * 5 epochs saved for each run = 25)
    global_pred_y = []
    y = []
    sem_arr = []
    for i in leave_one_out:
        pred_y = []
        for j in leave_one_out[i]:
            pred_y.append(j['pred_y'])
        
        global_pred_y.append( numpy.median(pred_y) )
        y.append( leave_one_out[i][0]['y'] )
        sem_arr.append( sem(pred_y) )
    
    stat = spearmanr(global_pred_y, y)
    print(sem_arr)
    print( 'Spearman Corr', stat )
    print( 'R2 score:', r2_score(y, global_pred_y) )
    
    plt.scatter(global_pred_y, y)
    plt.xlim(0, 125)
    plt.ylim(0, 125)
            
    plt.savefig('images/viscosity_y_pred.png')

    return True

def parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="")
    parser.add_argument("--leave_out_idx", type=int, default=4)
    parser.add_argument("--seed", type=int, default=47)
    return parser

def main():

    args = parser().parse_args()
    
    create_vectorized_data_from_pdb()
    
    # Loder uses randomize
    pl.seed_everything(args.seed)
    random.seed(args.seed)

    #train()
    train_LOOCV(args.leave_out_idx, args.seed)

    BRIDGE_UMAP()

    #after_sweep_analysis()

if(__name__=='__main__'):
    main()