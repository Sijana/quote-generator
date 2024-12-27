# Friends Nonsense Conversation Generator

The **Friends Nonsense Conversation Generator** is a fun app that generates humorous and nonsensical conversations between characters from the TV show *Friends*. Customize conversations with characters, catchphrases, and unique contexts to create entertaining scenarios!

## Features

- Select your favorite characters from *Friends*.
- Add predefined or custom catchphrases.
- Choose from predefined contexts or create your own.
- Generate multi-turn conversations with a customizable number of turns.
- Download the generated conversation as a text file.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Customization](#customization)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/friends-quote-generator.git
   cd friends-quote-generator
   ```

2. **Set Up the Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the Pre-Trained Model**:
   Place the pre-trained model file (`multi-quote_generator.pth`) in the `model/` directory. Note: The model file is not included in the repository due to size constraints.

---

## Usage

1. **Run the App**:
   ```bash
   streamlit run app.py
   ```

2. **Interact with the Interface**:
   - Select characters, contexts, and catchphrases in the sidebar.
   - Specify the number of conversation turns.
   - Click the "Generate Conversation" button to create a conversation.

3. **Download the Generated Conversation**:
   Use the "Download Conversation" button to save the output as a text file.

---

## File Descriptions

### 1. `app.py`
This is the main application file. It:
- Loads the pre-trained model.
- Provides the user interface via Streamlit.
- Handles input and generates conversations.

### 2. `multi_character_quote_generator.py`
Contains the implementation of the `MultiCharacterQuoteGenerator` class, which:
- Loads the pre-trained model.
- Generates conversations based on input parameters such as context, characters, and catchphrases.

### 3. `data_generator.py`
Utility script for generating training data for the model.

### 4. `data_analysis.py`
Analyzes and visualizes patterns in the training data.

---

## Customization

- **Add New Characters or Catchphrases**:
  Update the `CHARACTER_CATCHPHRASES` dictionary in `app.py` with additional characters and their catchphrases.

- **Add New Predefined Contexts**:
  Add more entries to the `PREDEFINED_CONTEXTS` list in `app.py`.

- **Train a Custom Model**:
  Use `data_generator.py` to prepare your dataset and retrain the model using your own conversations.

---

## Dependencies

The project requires the following Python packages:
- `streamlit`
- `torch`
- `random`

Ensure all dependencies are installed by running:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

1. **Error Loading Model**:
   - Ensure the `multi-quote_generator.pth` file is in the `model/` directory.
   - Verify the file path in the `load_model` function of `app.py`.

2. **Streamlit App Not Launching**:
   - Confirm all dependencies are installed.
   - Check for syntax errors in `app.py` or other files.

3. **Unexpected Output in Conversations**:
   - Review the input parameters for consistency.
   - Retrain the model with more data using `data_generator.py`.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

Inspired by the iconic TV show *Friends* and powered by machine learning, this project is designed for fans who want to relive the humor in a creative and interactive way!
