
import yaml

import numpy
import scipy
import torch

from torch.nn.functional import cosine_similarity

from model import interface_predictor

from packman.constants import THREE_LETTER_TO_ONE_LETTER

from umap import UMAP

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text


# Load config.yaml
with open('config.yaml', 'r') as file:
    yaml_input = yaml.safe_load(file)
# Storage
storage_location = yaml_input['storage']


# To put everything on GPU
if(torch.cuda.is_available()):
    use_device = 'cuda:'+str(torch.cuda.current_device())
    #torch.multiprocessing.set_start_method('spawn')
else:
    use_device = 'cpu'


def get_aa_embeddings():
    model = interface_predictor.load_from_checkpoint("models/interface_pred_epoch=0289-val_loss=0.455-val_f1=0.840.ckpt").to(use_device)
    # disable randomness, dropout, etc...
    model.eval()

    all_token_elements = sorted(THREE_LETTER_TO_ONE_LETTER.keys())
    aa_to_int_token = {i:numi for numi,i in enumerate(all_token_elements)}

    excluded_aa = ['UNK', 'ASX', 'GLX', 'PYL', 'SEC', 'XAA', 'XLE']
    
    aa_seq = []
    aa_embeddings_seq = []
    for i in aa_to_int_token:
        if(i not in excluded_aa):
            aa_seq.append(i)
            aa_embeddings_seq.append( model.residue_embeddings( torch.tensor(aa_to_int_token[i], device=use_device) ) )

    aa_embeddings_seq = torch.vstack( aa_embeddings_seq )
    return aa_seq, aa_embeddings_seq

def get_cosine_similarity(aa_seq, aa_embeddings_seq):

    #dot_prod = torch.mm(aa_embeddings_seq, aa_embeddings_seq.T).detach().cpu().numpy()
    dot_prod = cosine_similarity( aa_embeddings_seq[None,:,:], aa_embeddings_seq[:,None,:], dim=-1).detach().cpu().numpy()

    dot_prod = dot_prod

    fig, ax = plt.subplots()

    min_mask = numpy.triu( numpy.zeros(dot_prod.shape) + numpy.min(dot_prod) )
    
    im = ax.imshow(numpy.tril(dot_prod) + min_mask, cmap='binary')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(numpy.arange(len(aa_seq)), labels=aa_seq, rotation=90)
    ax.set_yticks(numpy.arange(len(aa_seq)), labels=aa_seq)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    ax.title.set_text('Cosine Similarity')
    plt.savefig('images/aa_cosine_similarity.png', dpi=300)
    plt.close()
    return True

def get_euclidean_distance(aa_seq, aa_embeddings_seq):

    def min_max_normalize(data):
        min_val = numpy.min(data)
        max_val = numpy.max(data)
        return (data - min_val) / (max_val - min_val)
    
    def mean_std_normalize(data):
        mean = numpy.mean(data)
        std = numpy.std(data)
        return (data - mean) / std
    
    def log_normalization(data):
        return numpy.log10(data)

    aa_embeddings_seq = aa_embeddings_seq.detach().cpu().numpy()

    dist = scipy.spatial.distance.cdist(aa_embeddings_seq, aa_embeddings_seq)

    dist = log_normalization(dist)

    min_mask = numpy.triu( numpy.zeros(dist.shape) + numpy.min(dist) )

    fig, ax = plt.subplots()

    im = ax.imshow(numpy.tril(dist)+min_mask, cmap='binary')
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(numpy.arange(len(aa_seq)), labels=aa_seq, rotation=90)
    ax.set_yticks(numpy.arange(len(aa_seq)), labels=aa_seq)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.title.set_text('Log10(Euclidean Distance)')
    plt.savefig('images/aa_distance_similarity.png', dpi=300)
    plt.close()

def get_embedding_clustering(aa_seq, aa_embeddings_seq, epochs=50):
    aa_types = ['Non-polar, aliphatic R groups', 'Aromatic R groups', 'Polar, Uncharged R groups', 'Positively charged R groups', 'Negatively charged R groups']
    aa_colors = {'GLY':aa_types[0], 'ALA':aa_types[0], 'VAL':aa_types[0], 'LEU':aa_types[0], 'MET':aa_types[0], 'ILE':aa_types[0], 'PHE':aa_types[1], 'TYR':aa_types[1], 'TRP':aa_types[1], 'SER':aa_types[2], 'THR':aa_types[2], 'CYS':aa_types[2], 'PRO':aa_types[2], 'ASN':aa_types[2], 'GLN':aa_types[2], 'LYS':aa_types[3], 'ARG':aa_types[3], 'HIS': aa_types[3], 'ASP': aa_types[4], 'GLU':aa_types[4] }

    # Set the color map to match the number of species
    z = range(1,len(aa_types))
    hot = plt.get_cmap('hot')
    cNorm  = colors.Normalize(vmin=0, vmax=len(aa_types))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)
    
    aa_embeddings_seq = aa_embeddings_seq.detach().cpu().numpy()

    umap_data = UMAP(n_components=3, min_dist=0, n_neighbors=2, metric='cosine', init='pca', n_epochs=epochs, verbose=True).fit_transform( aa_embeddings_seq )

    fig = plt.figure(figsize=(7,8))
    sp1 = fig.add_subplot(111)
    for numi,i in enumerate(aa_types):
        local_umap_data = []
        for numj,j in enumerate(aa_seq):
            if(i == aa_colors[j]):
                local_umap_data.append(umap_data[numj])
                
        local_umap_data = numpy.array(local_umap_data)
        sp1.scatter(local_umap_data[:, 0], local_umap_data[:, 1], color=scalarMap.to_rgba(numi), label=i, alpha=0.4)
    
    texts = [plt.text(umap_data[i][0], umap_data[i][1], aa_seq[i], ha='center', va='center') for i in range(len(aa_seq))]
    adjust_text(texts)

    sp1.legend()
    sp1.title.set_text('Amino Acid Embedding UMAP Analysis')

    plt.savefig('images/aa_embeddings_cluster.png')
    plt.close()

def main():
    aa_seq, aa_embeddings_seq = get_aa_embeddings()

    get_cosine_similarity(aa_seq, aa_embeddings_seq)

    get_euclidean_distance(aa_seq, aa_embeddings_seq)

    get_embedding_clustering(aa_seq, aa_embeddings_seq)

    print('done')

if(__name__=='__main__'):
    main()