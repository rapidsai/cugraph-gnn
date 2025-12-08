import os
import warnings

import torch
from torch.nn import Linear, Dropout, LayerNorm
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, GCNConv, SAGEConv, to_hetero

import pylibwholegraph.torch as wgth

from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_attr_dim, heads=1):
        super().__init__()
        
        self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=edge_attr_dim, concat=False, heads=heads)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.norm1 = LayerNorm(hidden_channels)
        self.lin1 = Linear(hidden_channels)
        
        self.lin2 = Linear(hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)
        
        self.dropout = Dropout(p=0.5)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr) + self.lin1(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = self.dropout(x)
        x = x.relu()

        x = self.lin2(x).relu()

        return x

class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_nodes, edge_attr_dim, metadata, learn_embeddings=False):
        super().__init__()
        
        self.learn_embeddings = learn_embeddings
        self.hidden_channels = hidden_channels

        self.paper_lin = Linear(num_features['paper'], hidden_channels)
        self.paper_norm = LayerNorm(hidden_channels)

        self.embeddings = {}
        if self.learn_embeddings:
            global_comm = wgth.get_global_communicator()
            for node_type in num_nodes:
                wg_node_emb = wgth.create_embedding(
                    global_comm,
                    "distributed",
                    "cpu",
                    torch.float32,
                    [num_nodes[node_type], hidden_channels],
                    cache_policy=None,
                    random_init=True,
                )
                self.embeddings[node_type] = wgth.embedding.WholeMemoryEmbeddingModule(wg_node_emb)
        else:
            self.mp = torch.nn.Sequential([
                SAGEConv(hidden_channels, hidden_channels),
                LayerNorm(hidden_channels),
                Dropout(p=0.5),
            ])
            self.mp = to_hetero(self.mp, metadata=metadata)
        
        self.encoder = Encoder(in_channels=hidden_channels, hidden_channels=hidden_channels, edge_attr_dim=edge_attr_dim)

    def forward(self, batch, edge_attr):
        x_paper = self.paper_lin(batch['paper'].x)
        x_paper = self.paper_norm(x_paper)
        
        if self.learn_embeddings:
            x_dict = {
                'paper': x_paper + self.embeddings['paper'](batch['paper'].n_id),
                'author': self.embeddings['author'](batch['author'].n_id),
                'institution': self.embeddings['institution'](batch['institution'].n_id),
                'field_of_study': self.embeddings['field_of_study'](batch['field_of_study'].n_id),
            }
        else:
            # have to obtain embeddings through message passing
            x_dict = {
                'paper': x_paper,
                'author': torch.zeros(batch['author'].n_id.numel(), self.hidden_channels),
                'institution': torch.zeros(batch['institution'].n_id.numel(), self.hidden_channels),
                'field': torch.zeros(batch['field'].n_id.numel(), self.hidden_channels),
            }
            x_dict = self.mp(x_dict, batch.edge_index_dict)
            x_dict['paper'] = x_paper

        x_dict = self.encoder(x_dict, batch.edge_index_dict, edge_attr)
        return x_dict

def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm
    rmm.reinitialize(
        devices=local_rank,
        managed_memory=True,
        pool_allocator=False,
    )

    from pylibwholegraph.torch.initialize import init as wm_init
    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())
    
    import cupy
    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator
    cupy.cuda.set_allocator(rmm_cupy_allocator)
    
    from pylibcugraph.comms import cugraph_comms_init
    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )

    torch.cuda.set_device(local_rank)

def train(feature_store, train_loader, model, optimizer, wm_optimizer):
    model.train()
    total_loss = total_examples = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch, {etype: feature_store[etype, 'x', None][eid] for etype, eid in batch.e_id_dict.items()})
        
        #loss = F.binary_cross_entropy_with_logits(out, batch['paper'].y)
        #loss.backward()
        #optimizer.step()
        #total_loss += loss.item() * batch['paper'].y.numel()
        #total_examples += batch['paper'].y.numel()
        break
    #return total_loss / total_examples
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn_embeddings', action='store_true')
    parser.add_argument('--dataset_root', type=str, default='datasets')
    args = parser.parse_args()

    torch.distributed.init_process_group(backend='nccl')

    if "LOCAL_RANK" not in os.environ:
        warnings.warn("This script should be run with 'torchrun`.  Exiting.")
        exit()

    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(local_rank)
    world_size = torch.distributed.get_world_size()

    if global_rank == 0:
        from pylibcugraph.comms import (
            cugraph_comms_create_unique_id,
        )
        cugraph_id = [cugraph_comms_create_unique_id()]
    else:
        cugraph_id = [None]
    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
    cugraph_id = cugraph_id[0]
    init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)

    from cugraph_pyg.data import FeatureStore, GraphStore
    feature_store = FeatureStore()
    graph_store = GraphStore()

    torch.distributed.barrier()
    if global_rank == 0:
        print('loading dataset...')
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-mag', root=args.dataset_root)
        data = dataset[0]

        # have to use "dict" here because OGB doesn't use the updated PyG API
        ei = data.edge_index_dict
        num_nodes = data.num_nodes_dict

        # add nodes
        print('adding nodes...')
        node_counts = torch.tensor([num_nodes['paper'], num_nodes['author'], num_nodes['institution'], num_nodes['field_of_study']], device='cuda', dtype=torch.int64)
        torch.distributed.broadcast(node_counts, src=0)

        # add edges
        print('adding edges...')
        graph_store[('paper', 'cites', 'paper'), 'coo', False, (num_nodes['paper'], num_nodes['paper'])] = ei['paper', 'cites', 'paper']
        graph_store[('author', 'writes', 'paper'), 'coo', False, (num_nodes['author'], num_nodes['paper'])] = ei['author', 'writes', 'paper']
        graph_store[('author','affiliated_with','institution'), 'coo', False, (num_nodes['author'], num_nodes['institution'])] = ei['author', 'affiliated_with', 'institution']
        graph_store[('paper', 'has_topic', 'field_of_study'), 'coo', False, (num_nodes['paper'], num_nodes['field_of_study'])] = ei['paper', 'has_topic', 'field_of_study']

        # add reverse edges
        print('adding reverse edges...')
        for edge_type in [('paper', 'cites', 'paper'), ('author', 'writes', 'paper'), ('author','affiliated_with','institution'), ('paper', 'has_topic', 'field_of_study')]:
            graph_store[(edge_type[2], 'rev_' + edge_type[1], edge_type[0]), 'coo', False, (num_nodes[edge_type[2]], num_nodes[edge_type[0]])] = ei[edge_type].flip(0)
        
        # add features
        print('adding features...')
        feature_store['paper', 'x', None] = data.x_dict['paper']

        del data
        del dataset
    else:
        from cugraph_pyg.tensor import empty
        # add nodes
        num_nodes = {}
        node_counts = torch.tensor([0, 0, 0, 0], device='cuda', dtype=torch.int64)
        torch.distributed.broadcast(node_counts, src=0, device=device)
        num_nodes['paper'] = node_counts[0]
        num_nodes['author'] = node_counts[1]
        num_nodes['institution'] = node_counts[2]
        num_nodes['field_of_study'] = node_counts[3]

        # add edges
        graph_store[('paper', 'cites', 'paper'), 'coo', False, (num_nodes['paper'], num_nodes['paper'])] = empty(dim=2)
        graph_store[('author', 'writes', 'paper'), 'coo', False, (num_nodes['author'], num_nodes['paper'])] = empty(dim=2)
        graph_store[('author','affiliated_with','institution'), 'coo', False, (num_nodes['author'], num_nodes['institution'])] = empty(dim=2)
        graph_store[('paper', 'has_topic', 'field_of_study'), 'coo', False, (num_nodes['paper'], num_nodes['field_of_study'])] = empty(dim=2)

        # add reverse edges
        for edge_type in [('paper', 'cites', 'paper'), ('author', 'writes', 'paper'), ('author','affiliated_with','institution'), ('paper', 'has_topic', 'field_of_study')]:
            graph_store[(edge_type[2], 'rev_' + edge_type[1], edge_type[0]), 'coo', False, (num_nodes[edge_type[2]], num_nodes[edge_type[0]])] = empty(dim=2)

        # add features
        feature_store['paper', 'x', None] = empty(dim=2)

    torch.distributed.barrier()

    from pylibcugraph import betweenness_centrality
    vx, vy = betweenness_centrality(
        resource_handle=graph_store._resource_handle,
        graph=graph_store._graph,
        k=100,
        random_state=62 + global_rank,
        normalized=True,
        include_endpoints=False,
        do_expensive_check=False,
    )
    
    _, i = torch.sort(vx)
    vy = torch.as_tensor(vy[i])
    offsets = torch.tensor(sorted(graph_store._vertex_offsets.values())[1:], device='cuda', dtype=torch.int64)
    vtypes = sorted(graph_store._vertex_offsets.keys())
    centralities = {vtypes[i]: t for i, t in enumerate(torch.tensor_split(vy, offsets))}

    for etype in graph_store.get_all_edge_attrs():
        src, dst = graph_store[etype]
        feature_store[etype, 'x', None] = (centralities[vtypes[src]] + centralities[vtypes[dst]]) / 2.0


    """
    model = Classifier(
        hidden_channels=args.hidden_channels,
        num_features={'paper': feature_store['paper', 'x', None].shape[1], 'author': 0, 'institution': 0, 'field_of_study': 0},
        num_nodes=num_nodes,
        edge_attr_dim=edge_attr_dim,
        metadata=metadata,
        learn_embeddings=args.learn_embeddings,
    )

    train(feature_store, train_loader, model, optimizer, wm_optimizer)
    """