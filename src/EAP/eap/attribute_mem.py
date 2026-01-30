from typing import Callable, List, Union, Optional
from functools import partial
import torch
import gc
from torch import Tensor
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import einsum
from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

# Maximum sequence length for tokenization
FORCE_MAX_LENGTH = 256

def get_npos_input_lengths(model, inputs):
    tokenized = model.tokenizer(inputs, padding='longest', return_tensors='pt', truncation=True, max_length=FORCE_MAX_LENGTH,add_special_tokens=True)
    n_pos = 1 + tokenized.attention_mask.size(1)
    input_lengths = 1 + tokenized.attention_mask.sum(1)
    return n_pos, input_lengths

def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores):
    target_device = scores.device 
    
    # Allocates d_model-sized buffer for compatibility with both attention and MLP activations
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=target_device, dtype=model.cfg.dtype)

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
    
    def activation_hook(index, activations, hook, add:bool=True):
        acts = activations.detach()
        if not add:
            acts = -acts
        try:
            # Fills only the valid dimensions (handles varying sizes across component types)
            dim = acts.shape[-1]
            activation_difference[:, :, index, :dim] += acts
        except RuntimeError as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e
    
    def gradient_hook(fwd_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        grads = gradients.detach()
        try:
            if isinstance(fwd_index, slice):
                fwd_index = fwd_index.start
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
                
            # Extracts matching dimensions for gradient computation
            dim = grads.shape[-1]
            act_slice = activation_difference[:, :, :fwd_index, :dim]
            
            s = einsum(act_slice, grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
            s = s.squeeze(1)
            try:
                scores[:fwd_index, bwd_index] += s
            except:
                print("here")

        except RuntimeError as e:
            print(hook.name, activation_difference.size(), grads.size())
            raise e

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        fwd_index =  graph.forward_index(node)
        if not isinstance(node, LogitNode):
            fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
            fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        if not isinstance(node, InputNode):
            if isinstance(node, AttentionNode):
                for i, letter in enumerate('qkv'):
                    bwd_index = graph.backward_index(node, qkv=letter)
                    bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, fwd_index, bwd_index)))
            else:
                bwd_index = graph.backward_index(node)
                bwd_hooks.append((node.in_hook, partial(gradient_hook, fwd_index, bwd_index)))
            
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference

def get_scores(model: HookedTransformer, graph: Graph, dataset, metric: Callable[[Tensor], Tensor]):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    # Disables attention result caching to reduce memory usage
    model.cfg.use_attn_result = False
    
    total_items = 0
    
    for clean, corrupted, label in tqdm(dataset, desc="Attributing"):
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        
        batch_size = len(clean)
        
        # Tokenize with truncation
        clean_tokenized = model.tokenizer(
            clean, 
            padding='longest', 
            truncation=True, 
            max_length=FORCE_MAX_LENGTH, 
            return_tensors='pt', 
            add_special_tokens=True
        )
        corrupted_tokenized = model.tokenizer(
            corrupted, 
            padding='longest', 
            truncation=True, 
            max_length=FORCE_MAX_LENGTH, 
            return_tensors='pt', 
            add_special_tokens=True
        )
        
        # Align sequence lengths between clean and corrupted inputs
        clean_len = clean_tokenized['input_ids'].size(1)
        corrupted_len = corrupted_tokenized['input_ids'].size(1)
        if clean_len != corrupted_len:
            max_len = max(clean_len, corrupted_len)
            pad_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
            def pad_tensor(t, target_len):
                curr = t.size(1)
                if curr < target_len:
                    return torch.nn.functional.pad(t, (0, target_len - curr), value=pad_id)
                return t
            def pad_mask(t, target_len):
                curr = t.size(1)
                if curr < target_len:
                    return torch.nn.functional.pad(t, (0, target_len - curr), value=0)
                return t
            clean_tokenized['input_ids'] = pad_tensor(clean_tokenized['input_ids'], max_len)
            clean_tokenized['attention_mask'] = pad_mask(clean_tokenized['attention_mask'], max_len)
            corrupted_tokenized['input_ids'] = pad_tensor(corrupted_tokenized['input_ids'], max_len)
            corrupted_tokenized['attention_mask'] = pad_mask(corrupted_tokenized['attention_mask'], max_len)
        
        clean_input_ids = clean_tokenized['input_ids'].to("cuda")
        corrupted_input_ids = corrupted_tokenized['input_ids'].to("cuda")
        input_lengths = clean_tokenized['attention_mask'].sum(1).to("cuda")
        
        n_pos = clean_input_ids.size(1)
        total_items += batch_size
        
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            corrupted_logits = model(corrupted_input_ids)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_input_ids)
            metric_value = metric(logits, corrupted_logits, input_lengths, label)
            
            # Triggers backward pass to compute gradients through hooks
            metric_value.backward()
            
        del fwd_hooks_corrupted
        del fwd_hooks_clean
        del bwd_hooks
        del activation_difference
        del logits
        del corrupted_logits
        del metric_value
        
        model.zero_grad(set_to_none=True)

    scores /= total_items
    return scores

# Integrated gradients variant (not currently used)
def get_scores_ig(model: HookedTransformer, graph: Graph, dataset, metric: Callable[[Tensor], Tensor], steps=30):
    pass

allowed_aggregations = {'sum', 'mean', 'l2'}        
def attribute(model: HookedTransformer, graph: Graph, dataset, metric: Callable[[Tensor], Tensor], aggregation='sum', integrated_gradients: Optional[int]=None):
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

    if integrated_gradients is None:
        scores = get_scores(model, graph, dataset, metric)
    else:
        assert integrated_gradients > 0
        pass 
        
    scores = scores.float().cpu().numpy()

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        edge.score = scores[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)]