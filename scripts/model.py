import os
import yaml

import torch
from torch.nn import Embedding, LeakyReLU, Dropout, BCELoss, Linear

import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR


from torch_geometric.nn import GCNConv, norm, GATv2Conv
from torch_geometric.utils import to_undirected, batched_negative_sampling

import pytorch_lightning as pl

from torchmetrics import F1Score, Accuracy
from torchmetrics.classification import BinaryHammingDistance

from packman.constants import THREE_LETTER_TO_ONE_LETTER

#https://medium.com/the-modern-scientist/graph-neural-networks-series-part-3-node-embedding-36613cc967d5

# 
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:10000"
# Input config file
with open('config.yaml', 'r') as file:
    yaml_input = yaml.safe_load(file)


# Device
if(torch.cuda.is_available()):
    use_device = 'cuda:'+str(torch.cuda.current_device())
    import os
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    #torch.cuda.empty_cache()
else:
    use_device = 'cpu'

'''
##################################################################################################
#                                  1. Single Value Regression                                    #
##################################################################################################
'''

class BridgeEncoder(pl.LightningModule):
    def __init__(self, embedding_dim=150, hidden_dim=400, heads=3, lr=3e-4, class_1_threshold=0.75, lr_gamma=0.98):
        super(BridgeEncoder, self).__init__()
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.class_1_threshold = class_1_threshold
        self.heads = heads

        # Residue embedding
        total_tokens = len(THREE_LETTER_TO_ONE_LETTER.keys())
        self.residue_embeddings = Embedding( num_embeddings=total_tokens+1, embedding_dim=embedding_dim, device=use_device, max_norm=1, padding_idx=yaml_input['padding_idx'] )

        # Activation
        #self.activation = ReLU(inplace=True)
        self.activation = LeakyReLU(negative_slope=0.1, inplace=True)

        # Norm & misc (It is established that LayerNorm works better thab BatchNorm1d for this model)
        self.norm1 = norm.LayerNorm(hidden_dim, mode='node')
        self.norm2 = norm.LayerNorm(hidden_dim, mode='node')

        #self.norm1 = BatchNorm1d(hidden_dim)
        #self.norm2 = BatchNorm1d(hidden_dim)

        self.dropout = Dropout(p=0.05)

        # Loss
        #self.cosine_loss_fn = CosineEmbeddingLoss()
        #self.mse_loss = MSELoss()
        self.bce_loss = BCELoss()

        # Metrices
        self.binary_hamming_distance = BinaryHammingDistance(threshold=self.class_1_threshold)
        self.f1_score = F1Score(num_classes=2, task='binary', threshold=self.class_1_threshold)
        self.accuracy = Accuracy(num_classes=2, task='binary', threshold=self.class_1_threshold)

        # Example
        self.conv_first = GATv2Conv(embedding_dim, hidden_dim, heads=self.heads, concat=False, add_self_loops=True, bias=False)
        #self.conv_first = GATv2Conv(embedding_dim, hidden_dim, add_self_loops=True, bias=False)
        self.conv_mid = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, bias=False)
        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)

       
    def forward(self, graph_1, graph_2):
        
        x_1 = self.residue_embeddings(graph_1.x)
        x_2 = self.residue_embeddings(graph_2.x)

        edges_1 = to_undirected(graph_1.edge_index)
        edges_2 = to_undirected(graph_2.edge_index)

        # Dont forget to make graph undirected. Layer norm might be useless because we are independently calculating loss then taking mean but lets try.

        # Layer 1
        x_1_out = self.conv_first(x_1, edges_1)
        x_2_out = self.conv_first(x_2, edges_2)

            # Norm
        x_1_out = self.norm1(x_1_out)
        x_2_out = self.norm1(x_2_out)

            # ReLU
        x_1_out = self.activation(x_1_out)
        x_2_out = self.activation(x_2_out)

            # Dropout
        x_1_out = self.dropout(x_1_out)
        x_2_out = self.dropout(x_2_out)

        # Layer 2
        x_1_out = self.conv_mid(x_1_out, edges_1)
        x_2_out = self.conv_mid(x_2_out, edges_2)

            # Norm
        x_1_out = self.norm2(x_1_out)
        x_2_out = self.norm2(x_2_out)

            # ReLU
        x_1_out = self.activation(x_1_out)
        x_2_out = self.activation(x_2_out)

            # Dropout
        x_1_out = self.dropout(x_1_out)
        x_2_out = self.dropout(x_2_out)

        # Layer 3
        x_1_out = self.fc1(x_1_out)
        x_2_out = self.fc1(x_2_out)

        # ReLU
        x_1_out = self.activation(x_1_out)
        x_2_out = self.activation(x_2_out)

        # Layer 4
        x_1_out = self.fc2(x_1_out)
        x_2_out = self.fc2(x_2_out)

        return x_1_out, x_2_out
    
    def core_process(self, batch, name):
        graph_1, graph_2, interface, example_name = batch
        
        x_out_1, x_out_2 = self.forward(graph_1, graph_2)

        # Negative sampling (Ideally it should be batched_negative_sampling instead of negative_sampling but batched_negative_sampling has some problems in indexing)
        negative_edges = batched_negative_sampling( edge_index=interface.edge_index, batch=(graph_1.batch, graph_2.batch), method='dense' )
        #negative_edges = negative_sampling( edge_index=interface.edge_index, num_nodes=(x_out_1.size(0), x_out_2.size(0)) )
        positive_edges = interface.edge_index

        # Concat positive and negative edge indices.
        all_edges = torch.cat([positive_edges, negative_edges], dim=-1).long()

        # Label for positive edges: 1, for negative edges: 0.
        target = torch.cat( [torch.ones(positive_edges.shape[1], device=use_device) , torch.zeros(negative_edges.shape[1], device=use_device)], dim=0)
        
        paired_product = (x_out_1[all_edges[0]] * x_out_2[all_edges[1]]).sum(dim=-1)
        
        pred = torch.sigmoid(paired_product)
        pred_binary = (pred >= self.class_1_threshold) * 1

        try:
            loss = self.bce_loss(pred, target)
    
            self.log(name+'_loss', loss, batch_size=interface.num_graphs)
            self.log(name+'_hamming_distance', self.binary_hamming_distance(pred_binary, target), batch_size=interface.num_graphs)
            self.log(name+'_f1', self.f1_score(pred_binary, target), batch_size=interface.num_graphs)
            self.log(name+'_accuracy', self.accuracy(pred_binary, target), batch_size=interface.num_graphs)
        except:
            loss = None

        return loss

    def training_step(self, batch):
        loss = self.core_process(batch, 'train')
        return loss
        
    def validation_step(self, batch):
        loss = self.core_process(batch, 'val')
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": ExponentialLR(optimizer, gamma=self.lr_gamma)} }
