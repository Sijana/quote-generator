import streamlit as st
import torch
import random
import sys
import os

# Import the quote generator (assuming it's in the same directory)
from multi_character_quote_generator import MultiCharacterQuoteGenerator

# Predefined contexts
PREDEFINED_CONTEXTS = [
    "Central Perk on a busy afternoon",
    "Monica and Chandler's apartment",
    "Ross's apartment after a breakup",
    "At a coffee shop",
    "During a group hangout",
    "After a complicated relationship moment",
    "Planning a surprise party",
    "Discussing a work problem",
    "During a family gathering",
    "At a wedding or engagement party"
]

# Predefined Catchphrases

CHARACTER_CATCHPHRASES = {
    "Chandler": [
        "Could I BE any more...",
        "Could I BE any more sarcastic?",
        "Could I BE any more confused?",
        "Could I BE any more frustrated?"
    ],
    "Joey": [
        "How you doin'?",
        "Joey doesn't share food!",
        "PIVOT!",
        "Does this smell funny to you?"
    ],
    "Phoebe": [
        "Oh. My. God.",
        "Smelly cat, smelly cat...",
        "I wish I could, but I don't want to.",
        "They don't know that we know they know we know!"
    ],
    "Ross": [
        "We were on a break!",
        "PIVOT!",
        "Science is cool!",
        "Unagi!"
    ],
    "Rachel": [
        "Oh. My. God.",
        "No uterus, no opinion!",
        "I'm a shoe!",
        "Did you get my message?"
    ],
    "Monica": [
        "I know!",
        "Rules are good!",
        "Clean is good!",
        "Did you just touch something?"
    ]
}

# List of Friends characters
CHARACTERS = [
    "Ross", "Rachel", "Chandler", 
    "Monica", "Joey", "Phoebe"
]

def load_model():
    """
    Load the pre-trained model
    """
    try:
        model_path = 'model/multi-quote_generator.pth'
        generator = MultiCharacterQuoteGenerator()
        generator.load_model('model/multi-quote_generator.pth')
        
        return generator
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Friends Conversation Generator", 
        page_icon="📺",
        layout="wide"
    )

    # Title and description
    st.title("🛋️ Friends Nonsense Conversation Generator")
    st.write("Generate nonsense conversations between your favorite Friends characters!")

    # Load the model
    generator = load_model()
    if generator is None:
        st.stop()

    # Sidebar for inputs
    st.sidebar.header("Conversation Parameters")

    # Character selection
    selected_characters = st.sidebar.multiselect(
        "Select Characters",
        list(CHARACTER_CATCHPHRASES.keys()),
        default=["Ross", "Rachel", "Chandler"]
    )

    # Catchphrase selection
    catchphrase_option = st.sidebar.radio(
        "Catchphrase Option", 
        ["No Catchphrase", "Predefined", "Custom"]
    )

    # Catchphrase input logic
    catchphrase = None
    catchphrase_character = None

    if catchphrase_option == "Predefined":
        # If predefined, show catchphrases for selected characters
        available_catchphrases = []
        for char in selected_characters:
            available_catchphrases.extend(CHARACTER_CATCHPHRASES[char])
        
        if available_catchphrases:
            selected_catchphrase = st.sidebar.selectbox(
                "Select a Catchphrase", 
                available_catchphrases
            )
            
            # Find the character for the selected catchphrase
            for char, phrases in CHARACTER_CATCHPHRASES.items():
                if selected_catchphrase in phrases:
                    catchphrase = selected_catchphrase
                    catchphrase_character = char
                    break
        else:
            st.sidebar.warning("No catchphrases available for selected characters")

    elif catchphrase_option == "Custom":
        # Custom catchphrase input
        catchphrase = st.sidebar.text_input(
            "Enter Custom Catchphrase",
            placeholder="E.g., 'How you doin'?'"
        )
        catchphrase_character = st.sidebar.selectbox(
            "Select Character for Catchphrase",
            selected_characters
        )

    # Context selection
    context_type = st.sidebar.radio(
        "Choose Context Type", 
        ["Predefined", "Custom"]
    )

    # Predefined contexts
    PREDEFINED_CONTEXTS = [
        "Central Perk on a busy afternoon",
        "Monica and Chandler's apartment",
        "Ross's apartment after a breakup",
        "At a coffee shop",
        "During a group hangout",
        "After a complicated relationship moment",
        "Planning a surprise party",
        "Discussing a work problem",
        "During a family gathering",
        "At a wedding or engagement party"
    ]

    # Context input based on selection
    if context_type == "Predefined":
        context = st.sidebar.selectbox(
            "Select a Predefined Context", 
            PREDEFINED_CONTEXTS
        )
    else:
        context = st.sidebar.text_input(
            "Enter Your Custom Context", 
            placeholder="E.g., At a coffee shop discussing relationship drama"
        )

    # Number of conversation turns
    num_turns = st.sidebar.slider(
        "Number of Conversation Turns", 
        min_value=3, 
        max_value=10, 
        value=5
    )

    # Generate button
    generate_button = st.sidebar.button("Generate Conversation")

    # Main content area
    conversation_container = st.container()

    # Generate conversation when button is clicked
    if generate_button:
        # Validate inputs
        if len(selected_characters) < 2:
            st.error("Please select at least two characters")
        elif not context:
            st.error("Please provide a context")
        else:
            # Show loading spinner
            with st.spinner("Generating conversation..."):
                try:
                    # Generate conversation
                    conversation = generator.generate_conversation(
                        initial_context=context,
                        characters=selected_characters.copy(),
                        num_turns=num_turns,
                        catchphrase=catchphrase,
                        catchphrase_character=catchphrase_character
                    )

                    # Display conversation
                    conversation_container.subheader("Generated Conversation")
                    conversation_container.markdown(f"**Context:** {context}")
                    if catchphrase and catchphrase_character:
                        conversation_container.markdown(f"**Catchphrase:** *{catchphrase}* (by {catchphrase_character})")
                    conversation_container.text(conversation)

                    # Optional: Copy to clipboard button
                    st.sidebar.download_button(
                        label="Download Conversation",
                        data=conversation,
                        file_name="friends_conversation.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Error generating conversation: {e}")

    # Additional information
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 Tip: Mix and match characters, contexts, and catchphrases "
        "to create unique conversation scenarios!"
    )

if __name__ == "__main__":
    main()