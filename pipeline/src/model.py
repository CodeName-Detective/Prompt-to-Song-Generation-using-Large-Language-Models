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
    def forward(self, text, attention_masks, start_label):
        genre, cls = self.encoder(text, attention_masks)
        hidden_state = torch.cat((cls,genre), dim=1).unsqueeze(0)
        cell_state = torch.cat((cls,genre), dim=1).unsqueeze(0)
        chord_vector = start_label.unsqueeze(1)
        output_chords = []
       
        for _ in range(3):
            output, hidden_state, cell_state = self.decoder(chord_vector, hidden_state, cell_state)
            output_chords.append(output)
            
            chord_vector = output.unsqueeze(1)
        return output_chords, genre

def load_model(rhlf=False):
    model = Transformer2Seq()
    if not rhlf:
        mode_state_dict = torch.load('artifacts/decoder_tf_2_seq.pth')
    else:
        mode_state_dict = torch.load('artifacts/decoder_rhlf.pth')
    model.decoder.load_state_dict(mode_state_dict)
    return model