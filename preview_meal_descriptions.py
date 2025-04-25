import requests
import logging
from dotenv import load_dotenv
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load environment variables from .env file
load_dotenv()

# Load Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_ANON_KEY environment variable is not set")

# Load OpenAI API key from environment variables

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def generate_embedding(text):
    """
    Generate an embedding for the given text using OpenAI's text-embedding-ada-002 model.
    """
    response = client.embeddings.create(input=text,
    model="text-embedding-ada-002")
    return response.data[0].embedding

def fetch_meals_and_prepare_descriptions(limit=5):
    """
    Fetch meals and their related data from Supabase, and prepare descriptions for embedding.
    """
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
    }

    # Fetch meals
    meals_response = requests.get(f"{SUPABASE_URL}/rest/v1/meals", headers=headers, params={"select": "id,name,category", "limit": limit})
    if meals_response.status_code != 200:
        raise ValueError(f"Failed to fetch meals: {meals_response.text}")
    meals = meals_response.json()

    data_to_embed = []

    for meal in meals:
        meal_id = meal["id"]
        meal_name = meal["name"]
        category_id = meal["category"]  # Updated to match the correct column name

        # Ensure category_id is treated as a list of values
        if not isinstance(category_id, list):
            category_id = [category_id]

        # Fetch category names directly using the category array of IDs
        category_ids = ','.join(category_id)  # Join IDs directly
        category_response = requests.get(f"{SUPABASE_URL}/rest/v1/meal_categories", headers=headers, params={"select": "name", "id": f"in.({category_ids})"})
        if category_response.status_code != 200:
            raise ValueError(f"Failed to fetch categories for meal {meal_id}: {category_response.text}")
        category_names = [category["name"] for category in category_response.json()]

        # Fetch ingredients (if available in the database)
        ingredients_response = requests.get(f"{SUPABASE_URL}/rest/v1/meals", headers=headers, params={"select": "ingredients", "id": f"eq.{meal_id}"})
        if ingredients_response.status_code == 200:
            ingredients_data = ingredients_response.json()[0].get("ingredients", [])
            ingredients = [
                f"{ingredient['amount']} {ingredient['unit']} {ingredient['item']}".strip()
                for ingredient in ingredients_data
                if ingredient['item'] and ingredient['amount'] is not None
            ] if isinstance(ingredients_data, list) else []
        else:
            ingredients = []

        # Generate enhanced description
        description = f"{meal_name} is a meal in the categories: {', '.join(category_names)}. "
        if ingredients:
            description += f"It is made with ingredients such as {', '.join(ingredients)}."

        # Append to the data to embed
        data_to_embed.append({
            "meal_name": meal_name,
            "category_names": category_names,
            "ingredients": ingredients,
            "description": description
        })

    return data_to_embed

# Fetch meals without embeddings from Supabase
headers = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
}
meals_response = requests.get(f"{SUPABASE_URL}/rest/v1/meals", headers=headers, params={"select": "id,name,category", "embedding": "is.null"})
if meals_response.status_code != 200:
    raise ValueError(f"Failed to fetch meals without embeddings: {meals_response.text}")
meals_without_embeddings = meals_response.json()

for meal in meals_without_embeddings:
    meal_id = meal["id"]
    meal_name = meal["name"]
    category_id = meal["category"]

    # Fetch category names
    category_ids = ','.join(category_id) if isinstance(category_id, list) else category_id
    category_response = requests.get(f"{SUPABASE_URL}/rest/v1/meal_categories", headers=headers, params={"select": "name", "id": f"in.({category_ids})"})
    category_names = [category["name"] for category in category_response.json()]

    # Generate description
    description = f"{meal_name} is a meal in the categories: {', '.join(category_names)}."

    # Generate embedding
    embedding = generate_embedding(description)

    # Save embedding to Supabase
    update_response = requests.patch(f"{SUPABASE_URL}/rest/v1/meals", headers=headers, json={"embedding": embedding}, params={"id": f"eq.{meal_id}"})
    if update_response.status_code != 200:
        raise ValueError(f"Failed to update embedding for meal {meal_id}: {update_response.text}")

# Fetch and display the data
data_to_embed = fetch_meals_and_prepare_descriptions()
print(json.dumps(data_to_embed, indent=4))