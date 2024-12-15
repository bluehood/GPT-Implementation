def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_structure(model, config):
    """Print detailed structure of the GPT model and parameter counts for each component"""
    def count_parameters_in_module(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print("\nGPT Model Structure Analysis")
    print("=" * 50)
    print(f"Total Parameters: {count_parameters(model):,}")
    print("\nConfiguration:")
    print(f"Vocabulary Size: {config.vocab_size}")
    print(f"Sequence Length: {config.block_size}")
    print(f"Embedding Dimension: {config.n_embd}")
    print(f"Number of Layers: {config.n_layer}")
    print(f"Number of Heads: {config.num_heads}")
    print(f"Dropout: {config.dropout}")
    
    print("\nComponent-wise Parameter Count:")
    print("-" * 50)
    
    # Token Embeddings
    embed_params = count_parameters_in_module(model.tok_emb)
    print(f"Token Embeddings: {embed_params:,} parameters")
    print(f"  Shape: {config.vocab_size} x {config.n_embd}")
    
    # Position Embeddings
    pos_params = config.block_size * config.n_embd
    print(f"Position Embeddings: {pos_params:,} parameters")
    print(f"  Shape: {config.block_size} x {config.n_embd}")
    
    # Transformer Blocks
    print("\nTransformer Blocks:")
    for i, block in enumerate(model.blocks):
        block_params = count_parameters_in_module(block)
        print(f"\nBlock {i+1}: {block_params:,} parameters")
        
        # Attention parameters
        att_params = count_parameters_in_module(block.attn)
        print(f"  Multi-Head Attention: {att_params:,} parameters")
        qkv_params = sum(p.numel() for p in block.attn.query_projs.parameters()) + \
                    sum(p.numel() for p in block.attn.key_projs.parameters()) + \
                    sum(p.numel() for p in block.attn.value_projs.parameters())
        print(f"    Q/K/V Projections: {qkv_params:,} parameters")
        proj_params = count_parameters_in_module(block.attn.proj)
        print(f"    Output Projection: {proj_params:,} parameters")
        
        # MLP parameters
        ffwd_params = count_parameters_in_module(block.ffwd)
        print(f"  FeedForward: {ffwd_params:,} parameters")
        
        # LayerNorm parameters
        ln_params = count_parameters_in_module(block.ln1) + count_parameters_in_module(block.ln2)
        print(f"  LayerNorm: {ln_params:,} parameters")
    
    # Final LayerNorm and Output
    final_ln_params = count_parameters_in_module(model.ln_f)
    print(f"\nFinal LayerNorm: {final_ln_params:,} parameters")
    
    head_params = count_parameters_in_module(model.head)
    print(f"Output Head: {head_params:,} parameters")
    print(f"  Shape: {config.n_embd} x {config.vocab_size}")
    
    # Memory Usage Estimation
    param_bytes = count_parameters(model) * 4  # 4 bytes per float32 parameter
    print(f"\nApproximate Memory Usage:")
    print(f"Parameters: {param_bytes / (1024*1024):.2f} MB")
    
    # Training memory will be higher due to gradients and optimizer states
    print(f"Training (with gradients + optimizer states): ~{param_bytes * 4 / (1024*1024):.2f} MB")