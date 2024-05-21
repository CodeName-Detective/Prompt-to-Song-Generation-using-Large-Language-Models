import pickle
import torch
import transformers
import numpy as np

from src.model import load_model

if torch.backends.mps.is_available():
    DEVICE = torch.device(device='mps')
elif torch.cuda.is_available():
    DEVICE = torch.device(device='cuda')
else:
    DEVICE = torch.device(device='cpu')

with open('artifacts/chords_mapping.pkl', "rb") as f:
        chords_map = pickle.load(f)

genre_map = {0: 'Metal', 1: 'rock', 2: 'rap', 3: 'pop', 4: 'country'}

def get_output(input_lyrics, start_chord, rhlf = False):
    
    #Initiate Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    tokens = tokenizer(input_lyrics, truncation=True, padding=True)
    
    input_ids = torch.tensor(tokens["input_ids"]).unsqueeze(0).to(DEVICE)
    
    attention_mask = torch.tensor(np.array(tokens["attention_mask"])).unsqueeze(0).to(DEVICE)
    
    start_label = [0] * len(chords_map['chord_to_id'])
    start_label[chords_map['chord_to_id'][start_chord]] = 1.0
    
    start_label = torch.tensor(np.array(start_label), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    model = load_model(rhlf).to(DEVICE)
    
    
    output_chords, genre = model(input_ids, attention_mask, start_label)
    
    final_chord_output = []
    
    if not rhlf:
        for chord in output_chords:
                final_chord_output.append(chords_map['id_to_chord'][torch.argmax(chord.squeeze(0)+torch.randn(len(chords_map['chord_to_id'])).to(DEVICE)).item()])
    else:
        for chord in output_chords:
            action_dist = torch.distributions.categorical.Categorical(probs=chord.squeeze(0))
            action = action_dist.sample()
            final_chord_output.append(chords_map['id_to_chord'][action.item()])
    
    genre = genre_map[torch.argmax(genre).item()]
    
    return final_chord_output, genre