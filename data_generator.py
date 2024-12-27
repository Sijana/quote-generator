# Import necessary libraries
import os
import re
import pandas as pd

# Function to correct typos and capitalize character names
def correct_typos_and_capitalize(file_path, typo_corrections):
    with open(file_path, 'r') as file:
        data = file.read()

    # Correct typos in text
    for typo, correction in typo_corrections.items():
        pattern = re.compile(r'\b' + re.escape(typo) + r':', re.IGNORECASE)
        data = pattern.sub(lambda x: correction + ':', data)

    # Capitalize all caps words
    data = re.sub(r'\b[A-Z]{2,}\b', lambda x: x.group().capitalize(), data)

    with open(file_path, 'w') as file:
        file.write(data)

# Example typo corrections dictionary
typo_corrections = {
    'Chan': 'Chandler',
    'Gunter': 'Gunther',
    'Mnca': 'Monica',
    'Rach': 'Rachel',
    'Phoe': 'Phoebe',
    'Racel': 'Rachel',
    'Rache': 'Rachel'
}

# Function to extract information from the text files
def extract_information(file_content, season, episode, episode_name, episode_id):
    lines = file_content.split("\n")
    data = []
    location = ""
    context = ""
    line_id = 0
    for line in lines:
        # Extract context
        if line.startswith("[Scene: "):
            context_match = re.search(r'\[Scene: (.+?)\]', line)
            if context_match:
                context = context_match.group(1).lower()
            # Extract location
            location_match = re.search(r'\[Scene: (.+?)[,.:]', line)
            if location_match:
                location = location_match.group(1).lower()
        
        # Extract character and dialogue
        match = re.match(r'(\w+): (.+)', line)
        if match:
            character = match.group(1).capitalize()  # Capitalize first character of character name
            dialogue = match.group(2)
            line_id += 1
            data.append([f"{episode_id}_{line_id}", character, dialogue, season, episode, episode_name, location, context])
    return data

def main():
    path = "friends_episodes"
    all_data = []
    episode_id = 0
    
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)

            # Correct typos and capitalize character names in the file
            correct_typos_and_capitalize(file_path, typo_corrections)

            # Extract season, episode, and episode name from the filename
            match = re.match(r'S(\d+)E(\d+) (.+)\.txt', filename)
            if match:
                episode_id += 1
                season = match.group(1)
                episode = match.group(2)
                episode_name = match.group(3)
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    episode_data = extract_information(file_content, season, episode, episode_name, episode_id)
                    all_data.extend(episode_data)
    
    # Create DataFrame
    columns = ['ID', 'Character', 'Line', 'Season', 'Episode', 'Episode Name', 'Location', 'Context']
    df = pd.DataFrame(all_data, columns=columns)
    
    # Save to CSV
    df.to_csv("friends_dialogues.csv", index=False)

if __name__ == "__main__":
    main()