Recipe Generator using T5 Model and Indian Food Dataset

📌 Project Overview

This project leverages a T5-based generative model to create Indian food recipes based on user-input ingredients. It uses a cleaned Indian food dataset to analyze common ingredients and cooking methods, enhancing recipe generation.

🚀 Features

Generates recipes based on user-provided ingredients.

Suggests additional ingredients based on common usage in similar recipes.

Uses a pre-trained T5 model for natural recipe generation.

Provides step-by-step cooking instructions in an easy-to-read format.

Interactive UI with Gradio for easy access.

📂 Dataset

Dataset: Cleaned_Indian_Food_Dataset.csv

Columns: ingredients, steps or directions

Purpose: Extract common ingredient usage and cooking patterns.

🔧 Installation

1️⃣ Clone the Repository

git clone https://github.com/yourusername/recipe-generator-t5.git
cd recipe-generator-t5

2️⃣ Install Dependencies

pip install gradio pandas transformers flax

3️⃣ Download Model & Tokenizer

The script will automatically load the T5 model from Hugging Face:

from transformers import FlaxAutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

🏃‍♂️ Running the Application

Start the Gradio interface by running:

python app.py

The interface will launch in your browser.

🛠 How It Works

1️⃣ Input Ingredients

Enter ingredients separated by commas (e.g., rice, turmeric, chicken).

2️⃣ Recipe Generation

The model suggests complementary ingredients.

Generates a step-by-step recipe using common cooking techniques.

3️⃣ Output Format

Ingredients List

Cooking Instructions

🎯 Example Usage

Input:

rice, turmeric, chicken

Output:

Recipe using rice, turmeric, chicken, cumin:

[INGREDIENTS]:
Rice, Turmeric, Chicken, Cumin

[DIRECTIONS]:
Step 1: Heat oil in a pan.
Step 2: Add cumin and sauté.
Step 3: Cook chicken until golden brown.
Step 4: Add turmeric and rice, mix well.
Step 5: Simmer until rice is cooked.

🌟 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

Happy Cooking! 🍽️

