import math

import numpy as np
import torch

from src.encoder_model import load_encoder_model


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = torch.nn.LSTM(543, 773, batch_first=True)
        self.linear1 = torch.nn.Linear(773, 1024)
        self.linear2 = torch.nn.Linear(1024, 543)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self,x, hidden_state, cell_state):
        output, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        output = output.squeeze(1)
        x = self.linear1(output)
        x = self.linear2(x)
        x = self.softmax(x)
        
        return x, hidden_state, cell_state

class Transformer2Seq(torch.nn.Module):
    def __init__(self):
        super(Transformer2Seq, self).__init__()
        self.encoder = load_encoder_model(True)
        self.decoder = Decoder()
        
    def forward(self, text, attention_masks, start_label, label_vector, teacher_force_ratio = 0.7):
        genre, cls = self.encoder(text, attention_masks)
        hidden_state = torch.cat((cls,genre), dim=1).unsqueeze(0)
        cell_state = torch.cat((cls,genre), dim=1).unsqueeze(0)
        chord_vector = start_label.unsqueeze(1)
        output_chords = torch.zeros_like(label_vector)
        
        action_log_proba = []
       
        for i in range(3):
            output, hidden_state, cell_state = self.decoder(chord_vector, hidden_state, cell_state)
            action_distribution = torch.distributions.categorical.Categorical(probs=output)
            
            action = action_distribution.sample()
            
            log_proba = action_distribution.log_prob(action)
            
            action_log_proba.append(log_proba)
            
            output_chords[:,i,action] = 1.0
            
            if i<2:
                if np.random.rand() <= teacher_force_ratio:
                    chord_vector = output.unsqueeze(1)
                else:
                    chord_vector = label_vector[:,i+1,:].unsqueeze(1)
            
        action_log_proba = torch.stack(action_log_proba, dim=1)
        return output_chords, genre, action_log_proba

def load_model():
    model = Transformer2Seq()
    return model