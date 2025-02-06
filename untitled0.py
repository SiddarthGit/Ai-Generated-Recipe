import gradio as gr
import pandas as pd
from transformers import FlaxAutoModelForSeq2SeqLM, AutoTokenizer
import os
import warnings
from collections import Counter

# Disable warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load the pre-trained generative model and tokenizer
MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

# Load the Indian food dataset
data_path = r"C:\Users\nsidd\Cleaned_Indian_Food_Dataset.csv"
df = pd.read_csv(data_path)

# Inspecting common ingredients and cooking methods in the dataset
def extract_common_methods_and_ingredients(df):
    ingredient_usage = {}
    for _, row in df.iterrows():
        ingredients = row.get("ingredients", "")
        steps = row.get("steps") or row.get("directions")
        if not isinstance(ingredients, str) or not isinstance(steps, str):
            continue
        ingredient_list = [ingredient.strip() for ingredient in ingredients.split(",")]
        for ingredient in ingredient_list:
            if ingredient not in ingredient_usage:
                ingredient_usage[ingredient] = []
            ingredient_usage[ingredient].extend(steps.split("--"))
    return ingredient_usage

# Get the common usage of each ingredient to guide generation
ingredient_usage = extract_common_methods_and_ingredients(df)

# Refined generation settings
generation_kwargs = {
    "max_length": 500,
    "min_length": 150,
    "num_beams": 2,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9
}

# Special tokens and mapping for formatting
special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}

def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")
    return text

def target_postprocessing(texts, special_tokens):
    if not isinstance(texts, list):
        texts = [texts]
    
    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens)
        for k, v in tokens_map.items():
            text = text.replace(k, v)
        new_texts.append(text)

    return new_texts

# Function to generate a recipe
def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = [f"items: {inp}" for inp in _inputs]
    inputs = tokenizer(
        inputs, 
        max_length=256, 
        padding="max_length", 
        truncation=True, 
        return_tensors="jax"
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    output_ids = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        **generation_kwargs
    )
    generated = output_ids.sequences
    generated_recipe = target_postprocessing(
        tokenizer.batch_decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        special_tokens
    )
    
    return generated_recipe

# Function to suggest additional ingredients based on similar recipes
def suggest_ingredients(ingredient_list):
    try:
        # Filter dataset for recipes that contain any of the ingredients provided
        similar_dishes = df[df['ingredients'].str.contains('|'.join(ingredient_list), case=False, na=False)]
        
        # Collect ingredients from these similar recipes
        all_ingredients = []
        for ingredients in similar_dishes['ingredients']:
            if pd.isna(ingredients):
                continue
            all_ingredients.extend(ingredient.strip() for ingredient in ingredients.split(","))
        
        # Count most common ingredients that aren't already in the original list
        common_ingredients = [item for item, count in Counter(all_ingredients).items() 
                              if item.lower() not in map(str.lower, ingredient_list) and count > 1]
        
        return common_ingredients[:3]  # Return top 3 suggested ingredients
    except Exception as e:
        print(f"Error in suggesting ingredients: {e}")
        return []  # Return an empty list in case of error

# Function to generate a recipe based on input and additional ingredients
def generate_recipe_from_ingredients(ingredients):
    # Clean and format input ingredients
    ingredient_list = [ingredient.strip().capitalize() for ingredient in ingredients.split(",")]

    # Suggest additional complementary ingredients
    additional_ingredients = suggest_ingredients(ingredient_list)
    full_ingredient_list = ingredient_list + additional_ingredients

    # Create a prompt based on common usage patterns
    prompt = f"Recipe using: {', '.join(full_ingredient_list)}"
    usage_instructions = []
    for ingredient in full_ingredient_list:
        if ingredient.lower() in ingredient_usage:
            # Use common steps for this ingredient from the dataset
            common_steps = list(set(ingredient_usage[ingredient.lower()]))[:5]  # Get top 5 common steps
            usage_instructions.extend(common_steps)
    if usage_instructions:
        prompt += "\nInstructions:\n" + " ".join(usage_instructions)

    # Generate recipe instructions
    generated = generation_function([prompt])

    # Format output as step-by-step instructions
    result = f"Recipe using {', '.join(full_ingredient_list)}:\n\n"
    result += "[INGREDIENTS]:\n" + ", ".join(full_ingredient_list) + "\n\n"
    result += "[DIRECTIONS]:\n"
    step_counter = 1  # Step counter for instructions

    for text in generated:
        sections = text.split("\n")
        for section in sections:
            steps = section.split("--")  # Split on step delimiters
            for step in steps:
                if step.strip():  # Skip empty steps
                    result += f"  Step {step_counter}: {step.strip().capitalize()}.\n"
                    step_counter += 1

    return result

# Gradio interface
with gr.Blocks() as interface:
    ingredients_input = gr.Textbox(
        lines=3, 
        placeholder="Enter ingredients separated by commas (e.g., rice, turmeric, chicken)",
        label="Ingredients"
    )

    generate_button = gr.Button("Generate Recipe")
    output_text = gr.Textbox(label="Recipe", lines=20)

    # Function to handle generate button click
    def on_generate(ingredients):
        recipe = generate_recipe_from_ingredients(ingredients)
        return recipe

    generate_button.click(on_generate, inputs=ingredients_input, outputs=output_text)

# Launch the interface
interface.launch()