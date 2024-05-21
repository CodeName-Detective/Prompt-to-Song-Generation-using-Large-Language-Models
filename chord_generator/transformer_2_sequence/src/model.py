import math

import numpy as np
import torch

from src.encoder_model import load_encoder_model


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = torch.nn.LSTM(1316, hidden_size=128, batch_first=True)
        self.linear1 = torch.nn.Linear(128, 1024)
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
        state_vector = torch.cat((cls,genre), dim=1)
        
        hidden_state = torch.zeros(1, start_label.shape[0], 128).to('cuda')  # 1 layer, batch size 64, hidden size 128
        cell_state = torch.zeros(1, start_label.shape[0], 128).to('cuda')
        
        chord_vector = start_label
        output_chords = torch.zeros_like(label_vector)
        
        for i in range(3):
            if len(chord_vector.shape) == 2:
                chord_vector = torch.cat((chord_vector, state_vector), dim=1).unsqueeze(1)
            else:
                chord_vector = torch.cat((chord_vector, state_vector.unsqueeze(1)), dim=2)
                
            output, hidden_state, cell_state = self.decoder(chord_vector, hidden_state, cell_state)
            output_chords[:,i,:] = output
            
            if i<2:
                if np.random.rand() <= teacher_force_ratio:
                    chord_vector = output.unsqueeze(1)
                else:
                    chord_vector = label_vector[:,i+1,:].unsqueeze(1)
        return output_chords, genre

def load_model():
    model = Transformer2Seq()
    return model