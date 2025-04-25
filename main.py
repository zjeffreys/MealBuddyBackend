from fastapi import FastAPI, Query
import numpy as np
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY or not OPENAI_API_KEY:
    raise ValueError("Environment variables SUPABASE_URL, SUPABASE_ANON_KEY, or OPENAI_API_KEY are not set")

headers = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
}

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

@app.post("/generate_meal_embeddings")
def generate_meal_embeddings():
    """
    Generate embeddings for meals without embeddings in the database.
    """
    meals_response = requests.get(f"{SUPABASE_URL}/rest/v1/meals", headers=headers, params={"select": "id,name,category", "embedding": "is.null"})
    if meals_response.status_code != 200:
        return {"error": f"Failed to fetch meals: {meals_response.text}"}

    meals_without_embeddings = meals_response.json()

    for meal in meals_without_embeddings:
        meal_id = meal["id"]
        meal_name = meal["name"]
        category_id = meal["category"]

        # Fetch category names
        category_response = requests.get(f"{SUPABASE_URL}/rest/v1/meal_categories", headers=headers, params={"select": "name", "id": f"eq.{category_id}"})
        if category_response.status_code != 200:
            continue

        category_name = category_response.json()[0]["name"]

        # Generate description
        description = f"{meal_name} is a meal in the category: {category_name}."

        # Generate embedding
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=description
        )
        embedding = response.data[0].embedding

        # Save embedding to Supabase
        update_response = requests.patch(f"{SUPABASE_URL}/rest/v1/meals", headers=headers, json={"embedding": embedding}, params={"id": f"eq.{meal_id}"})
        if update_response.status_code not in [200, 204]:
            continue

    return {"message": "Embeddings generated successfully"}

@app.get("/search_meals")
def search_meals(query: str):
    """
    Search for meals using the provided query and return the best matches based on embeddings.
    """
    # Generate embedding for the query
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = np.array(response.data[0].embedding)

    # Fetch all meals with embeddings
    meals_response = requests.get(f"{SUPABASE_URL}/rest/v1/meals", headers=headers, params={"select": "id,name,embedding"})
    if meals_response.status_code != 200:
        return {"error": f"Failed to fetch meals: {meals_response.text}"}

    meals = meals_response.json()

    # Calculate similarity scores
    results = []
    for meal in meals:
        if not meal.get("embedding"):
            continue
        meal_embedding = np.array(meal["embedding"])
        similarity = np.dot(query_embedding, meal_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(meal_embedding))
        results.append({"id": meal["id"], "name": meal["name"], "similarity": similarity})

    # Sort results by similarity
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    return results[:5]