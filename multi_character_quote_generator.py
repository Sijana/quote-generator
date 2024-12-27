import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from tqdm import tqdm
import random

class ConversationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.inputs = []

        # Group conversations by some criteria (e.g., location, episode)
        grouped_conversations = self._group_conversations(df)

        for conversation in grouped_conversations:
            # Create a conversation prompt
            prompt = self._create_conversation_prompt(conversation)

            # Tokenize the entire conversation
            encodings = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )

            self.inputs.append({
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze()
            })

    def _group_conversations(self, df):
        """
        Group lines into conversations based on location or other contextual clues
        """
        # Group by location and sort by some criteria (e.g., original order)
        grouped = df.sort_values(['Location', 'Episode', 'Season'])

        # Split into conversation chunks
        conversations = []
        current_conversation = []

        for _, row in grouped.iterrows():
            # If conversation is getting too long, start a new one
            if len(current_conversation) > 10:
                conversations.append(current_conversation)
                current_conversation = []

            current_conversation.append(row)

        # Add the last conversation if not empty
        if current_conversation:
            conversations.append(current_conversation)

        return conversations

    def _create_conversation_prompt(self, conversation):
        """
        Create a formatted conversation prompt
        """
        prompt = "Conversation Context:\n"

        # Add location and setting information
        if conversation:
            location = conversation[0]['Location']
            prompt += f"Location: {location}\n"

        prompt += "\nDialogue:\n"

        # Format the conversation
        for line in conversation:
            prompt += f"{line['Character']}: {line['Line']}\n"

        prompt += "\nContinue the conversation:"

        return prompt

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

class MultiCharacterQuoteGenerator:
    def __init__(self, model_path=None):
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Load model if path is provided
        if model_path:
            self.load_model(model_path)

    def save_model(self, path, save_tokenizer=True):
        """
        Save the current model state to specified path

        Parameters:
        -----------
        path : str
            Path where the model will be saved
        save_tokenizer : bool, optional
            Whether to save the tokenizer alongside the model, by default True
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

            # Save model state dictionary
            torch.save(self.model.state_dict(), path)
            print(f"Model saved to {path}")

            # Optionally save tokenizer
            if save_tokenizer:
                tokenizer_path = path.replace('.pth', '_tokenizer')
                self.tokenizer.save_pretrained(tokenizer_path)
                print(f"Tokenizer saved to {tokenizer_path}")

        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, path, load_tokenizer=True):
        """
        Load a previously saved model

        Parameters:
        -----------
        path : str
            Path to the saved model file
        load_tokenizer : bool, optional
            Whether to load the tokenizer alongside the model, by default True

        Raises:
        -------
        FileNotFoundError
            If the specified model file does not exist
        ValueError
            If the model file is incompatible or corrupted
        """
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")

        try:
            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load state dictionary
            state_dict = torch.load(path, map_location=device)

            # Load state dictionary into model
            self.model.load_state_dict(state_dict)

            # Move model to appropriate device
            self.model.to(device)

            # Optionally load tokenizer
            if load_tokenizer:
                tokenizer_path = path.replace('.pth', '_tokenizer')
                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
                    print(f"Tokenizer loaded from {tokenizer_path}")
                except Exception as e:
                    print(f"Warning: Could not load tokenizer: {e}")

            print(f"Model successfully loaded from {path}")

        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def train(self, train_dataframe, epochs=3, batch_size=4, learning_rate=5e-5):
        # Use the new ConversationDataset
        dataset = ConversationDataset(train_dataframe, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Rest of the training method remains similar to previous implementation
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        progress_bar = tqdm(total=epochs * len(dataloader),
                            desc="Training Progress",
                            unit="batch")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )

                loss = outputs.loss
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                progress_bar.update(1)

            progress_bar.set_postfix({'Epoch': epoch + 1, 'Loss': epoch_loss / len(dataloader)})

        progress_bar.close()

    def generate_conversation(self,
                               initial_context,
                               characters,
                               num_turns=5,
                               max_length=200,
                               catchphrase=None,
                               catchphrase_character=None):
        """
        Generate a multi-character conversation with optional catchphrase

        Parameters:
        -----------
        initial_context : str
            Initial setting or context for the conversation
        characters : list
            List of characters participating in the conversation
        num_turns : int, optional
            Number of conversation turns to generate
        max_length : int, optional
            Maximum total length of the generated conversation
        catchphrase : str, optional
            Specific catchphrase to include
        catchphrase_character : str, optional
            Character associated with the catchphrase

        Returns:
        --------
        str
            Generated conversation
        """
        # Prepare the initial prompt
        prompt = f"Conversation Context: {initial_context}\n\nDialogue:\n"

        # Track catchphrase insertion
        catchphrase_inserted = False

        # Randomly start with one of the characters
        current_character = random.choice(characters)

        for turn in range(num_turns):
            # Add current character's turn marker
            prompt += f"{current_character}: "

            # Check if catchphrase should be inserted
            if not catchphrase_inserted and catchphrase and catchphrase_character:
                if current_character == catchphrase_character:
                    # Insert catchphrase
                    prompt += f"{catchphrase} "
                    catchphrase_inserted = True

            # Generate the line
            inputs = self.tokenizer(prompt, return_tensors='pt')

            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the new line
            new_line = generated_text.split(prompt)[-1].split('\n')[0].strip()
            prompt += f"{new_line}\n"

            # Switch to next character
            characters.remove(current_character)
            if not characters:
                characters = ["Ross", "Rachel", "Chandler", "Monica", "Joey", "Phoebe"]
            current_character = random.choice(characters)

        return prompt