import torch
import json
from pathlib import Path
from model import GPT, GPTConfig

class Tokenizer:
    def __init__(self, vocab_path):
        # Load vocabulary from JSON file
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
            
        # Create reverse vocabulary (id to token)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
    def encode(self, text):
        # Split text into words and convert to token ids
        # This is a simple implementation - you might want to add more preprocessing
        words = text.split()
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
    def decode(self, token_ids):
        # Convert token ids back to words and join them
        words = [self.id_to_token.get(id, '<UNK>') for id in token_ids]
        # Clean up special tokens and join words
        words = [word for word in words if word not in ['<PAD>', '<UNK>', '<START>', '<END>']]
        return ' '.join(words)
        
    def __len__(self):
        return len(self.vocab)

def load_model(checkpoint_path, config):
    """Load a trained GPT model from checkpoint"""
    model = GPT(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def generate_text(model, tokenizer, prompt="", max_tokens=100, temperature=0.8, top_k=50, device='cpu'):
    """Generate text from a prompt"""
    model.eval()
    model = model.to(device)
    
    # Encode prompt if provided
    if prompt:
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
        context = context.unsqueeze(0)  # Add batch dimension
    else:
        # Start with START token if no prompt
        context = torch.tensor([[tokenizer.vocab['<START>']]], dtype=torch.long, device=device)
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text

def main():
    # Settings
    checkpoint_path = './models/checkpoints/checkpoint_epoch_9.pt'
    tokenizer_path = './models/alice_in_wonderland_tokeniser.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = Tokenizer(tokenizer_path)
    
    # Create config and load model
    config = GPTConfig(
        vocab_size=len(tokenizer),
        block_size=128,
        n_layer=6,
        n_embd=384,
        num_heads=6,
        dropout=0.1
    )
    
    model = load_model(checkpoint_path, config)
    
    # Generate from different prompts
    prompts = [
        "Alice was",
        "The rabbit",
        "Down the",
        ""  # Empty prompt to generate from scratch
    ]
    
    print("\nGenerating text samples examples:\n")
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 50)
        generated_text = generate_text(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=100,
            temperature=0.8,
            device=device
        )
        print(generated_text)
        print("-" * 50)

    while True:
        prompt = input("\nPrompt > ")
        generated_text = generate_text(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=100,
            temperature=0.8,
            device=device
        )
        print("-" * 50)
        print(generated_text)
        print()

def debug():
    import debugpy
    debugpy.listen(('127.0.0.1', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    return

if __name__ == "__main__":
    debug()
    main()