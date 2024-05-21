import math

import torch
import transformers


class Transformer(torch.nn.Module):
    """
    A PyTorch module for a custom transformer-based model.

    This class encapsulates a transformer model, typically for a classification task.
    It allows for the transformer's parameters to be frozen and adds a linear layer
    on top of the hidden states output by the transformer.

    Attributes:
        transformer (nn.Module): The transformer model from Hugging Face's library.
        fc (nn.Linear): Linear layer to map from hidden states to output dimension.
    """
    def __init__(self, transformer, freeze):
        """
        Initializes the Transformer model.

        Args:
            transformer (nn.Module): A pre-trained transformer model from Hugging Face.
            output_dim (int): The dimension of the output layer, typically the number of classes.
            freeze (bool): If True, the parameters of the transformer will not be updated during training.
        """
        super().__init__()  # Initialize the parent class, nn.Module.
        self.transformer = transformer  # Store the transformer model.
        hidden_dim = transformer.config.hidden_size  # Extract the hidden layer size from the transformer configuration.
        
        self.fc1 = torch.nn.Linear(hidden_dim+543, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512,256)
        self.fc4 = torch.nn.Linear(256,128)
        self.fc5 = torch.nn.Linear(128, 64)
        self.fc6 = torch.nn.Linear(64, 32)
        self.fc7 = torch.nn.Linear(32,1)# Create a linear layer for classification.

        # If freezing is requested, disable gradient calculations for all transformer parameters.
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False  # Freeze the parameters.

    def forward(self, ids, chord_vector, attention_mask):
        """
        Defines the forward pass of the model.

        Args:
            ids (Tensor): Input token ids, shaped [batch size, sequence length].

        Returns:
            Tensor: The prediction scores for each class, shaped [batch size, output dim].
        """
        # Pass the input through the transformer model.
        output = self.transformer(ids, attention_mask = attention_mask, output_attentions=True)
        hidden = output.last_hidden_state  # Extract the last hidden states.
        # hidden shape: [batch size, sequence length, hidden dimension]

        # sattention = output.attentions[-1]  # Get the last layer's attention values.
        # attention shape: [batch size, number of heads, sequence length, sequence length]

        cls_hidden = hidden[:, 0, :]  # Extract the [CLS] token's hidden state (first token).
        
        x = torch.cat([cls_hidden, chord_vector], dim=1)
        
        x = torch.nn.functional.relu(self.fc1(x)) # Pass the [CLS] hidden state through the linear layer.
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = torch.nn.functional.relu(self.fc5(x))
        x = torch.nn.functional.relu(self.fc6(x))
        x = torch.nn.functional.tanh(self.fc7(x))
        # prediction shape: [batch size, output dimension]

        return x  # Return the final prediction scores.

def load_model(freeze=True):
    transformer = transformers.AutoModel.from_pretrained('distilbert-base-uncased')
    model = Transformer(transformer, freeze)
    return model