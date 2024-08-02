import torch
from transformers import AutoConfig, AutoModelForMaskedLM


class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Redefine layers according to the original model's architecture
        self.embedding_one = torch.nn.Embedding(30522, 768)  
        self.encoder = torch.nn.GRU(768, 768, num_layers=12)
        self.decoder = torch.nn.Linear(768, 30522)
        
    def forward(self, input_ids):
        x = self.embedding_one(input_ids)
        x, _ = self.encoder(x)
        logits = self.decoder(x)
        return logits

# Load the model state dict from model.pth
state_dict = torch.load('./trained_model/Saligned-mawps-single/model.pth')

# Remove the 'model.' prefix if necessary and map keys to match the CustomModel
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('model.', '')  # Adjust key names as needed
    new_state_dict[new_key] = v

# Load the state dict into the CustomModel
model = CustomModel()
model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to ignore mismatched keys

# Save the modified model in the Hugging Face format
model_path = "./trained_model/Saligned-mawps-single/pytorch_model.bin"
torch.save(model.state_dict(), model_path)
