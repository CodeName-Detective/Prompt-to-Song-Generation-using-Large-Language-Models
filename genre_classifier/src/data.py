import ast

import datasets
import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split

genre_map = {'Metal':0, 'rock':1, 'rap':2, 'pop':3, 'country':4}

def tokenize_and_numericalize_example(example, tokenizer):
    tokens = tokenizer(example["Lyric"], truncation=True, padding=True)
    label = example['genre']
    return {"ids": tokens["input_ids"], "label": label, 'attention_mask': tokens["attention_mask"]}


def get_collate_fn(pad_index):
    """
    Creates a collate function tailored for padding sequences.

    Args:
        pad_index (int): The index used for padding sequences (typically the index of the [PAD] token).

    Returns:
        Function: A custom collate function for a DataLoader.
    """

    # Define the actual collate function that will be used by the DataLoader.
    def collate_fn(batch):
        """
        Processes a batch of data.

        Args:
            batch (list of dicts): The data batch, where each item is expected to have
                                   'ids' (token indices) and 'label' (target label).

        Returns:
            dict: A batch where 'ids' are padded and 'label' are stacked into tensors.
        """
        # Extract 'ids' from each item in the batch and pad them to have the same length.
        batch_ids = [item["ids"] for item in batch]  # Extract token indices for all items in the batch.
        
        batch_attention_masks = [item["attention_mask"] for item in batch]
        
        batch_ids = torch.nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )  # Pad sequences to the longest one in the batch.
        
        batch_attention_masks = torch.nn.utils.rnn.pad_sequence(
            batch_attention_masks, padding_value=pad_index, batch_first=True
        )  # Pad sequences to the longest one in the batch.

        # Extract 'label' from each item in the batch and convert them into a tensor.
        batch_label = [item["label"] for item in batch]  # Extract labels for all items in the batch.
        batch_label = torch.stack(batch_label)  # Stack all labels into a single tensor.

        # Combine the processed ids and labels back into a batch dictionary.
        batch = {"ids": batch_ids, "label": batch_label, "attention_masks": batch_attention_masks}

        # Return the processed batch.
        return batch

    # Return the collate function to be used with a DataLoader.
    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    """
    Creates and returns a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int): The number of samples to load in each batch.
        pad_index (int): The index used for padding sequences for uniform length.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to False.

    Returns:
        DataLoader: A PyTorch DataLoader instance that yields batches of data.
    """

    # Create a collate function that will be used to pad the sequences in the batches.
    # The 'get_collate_fn' function creates a custom collate function that uses the provided
    # 'pad_index' to pad all sequences to the same length.
    collate_fn = get_collate_fn(pad_index)

    # Create the DataLoader instance. This will be used to load batches of data from the dataset.
    # The DataLoader uses the custom collate function defined above to handle variable-length sequences,
    # and it can shuffle the data every epoch if required.
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,          # The dataset from which to load data.
        batch_size=batch_size,    # The number of samples per batch.
        collate_fn=collate_fn,    # The function used to merge individual samples into batches.
        shuffle=shuffle,          # Whether to shuffle the data at the start of each epoch.
    )

    # Return the created DataLoader.
    return data_loader

def get_data(tokenizer_name, batch_size):
    #Get Data
    df = pd.read_csv('data/cleaned_lyrics.csv')[['Lyric', 'genre']]
    df['genre'] = df['genre'].replace(genre_map)
    
    train_data, test_data, _, _ = train_test_split(df, df['genre'], test_size=0.20, random_state=42, stratify=df['genre'])
    
    train_data = datasets.Dataset.from_pandas(train_data)
    test_data = datasets.Dataset.from_pandas(test_data)
    
    
    #Initiate Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=512)
    
    
    # Tokenize the data
    train_data = train_data.map(tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer})
    test_data = test_data.map(tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer})
    
    pad_index = tokenizer.pad_token_id
    
    # Convert arrays to torch Tensor
    train_data = train_data.with_format(type="torch", columns=["ids", "label", "attention_mask"])
    test_data = test_data.with_format(type="torch", columns=["ids", "label", "attention_mask"])
    
    # Create Dataloaders
    train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
    test_data_loader = get_data_loader(test_data, batch_size, pad_index)
    
    return train_data_loader, test_data_loader