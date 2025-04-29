from fastapi import FastAPI, Query, HTTPException, Request, Form
import numpy as np
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
import stripe
import logging
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging to include debug information
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn")

# Load environment variables
load_dotenv()

app = FastAPI()

# Update CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://mealbuddyfrontend.com", 
        "https://meal-buddy-gold.vercel.app",
        "https://mealprepme.com"
    ],  # Add your frontend origins here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
YOUR_DOMAIN = os.getenv('YOUR_DOMAIN')

# Debugging: Log the Stripe Secret Key to ensure it is loaded
if stripe.api_key:
    logger.info("Stripe Secret Key loaded successfully.")
else:
    logger.error("Stripe Secret Key is not set. Check your .env file.")

if not SUPABASE_URL or not SUPABASE_ANON_KEY or not OPENAI_API_KEY:
    raise ValueError("Environment variables SUPABASE_URL, SUPABASE_ANON_KEY, or OPENAI_API_KEY are not set")

headers = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
}

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown complete.")

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

@app.post("/create-checkout-session")
async def create_checkout_session(body: dict):
    try:
        product_id = body.get("product_id")
        coupon_code = body.get("coupon_code")

        if not product_id:
            raise HTTPException(status_code=400, detail="Missing product_id in request body.")

        logger.info(f"Received product_id: {product_id}")
        if coupon_code:
            logger.info(f"Received coupon_code: {coupon_code}")

        # Step 1: Retrieve product prices
        logger.info("Retrieving prices from Stripe.")
        prices = stripe.Price.list(
            product=product_id,
            expand=['data.product']
        )
        logger.info(f"Prices retrieved: {prices}")

        # Step 2: Create checkout session
        logger.info("Creating checkout session.")
        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    'price': prices.data[0].id,
                    'quantity': 1,
                },
            ],
            discounts=[
                {
                    'coupon': coupon_code
                }
            ] if coupon_code else None,
            allow_promotion_codes=True,  # Enable promo code input at checkout
            mode='subscription',
            success_url=YOUR_DOMAIN +
            '?success=true&session_id={CHECKOUT_SESSION_ID}',
            cancel_url=YOUR_DOMAIN + '?canceled=true',
        )
        logger.info(f"Checkout session created: {checkout_session}")

        return JSONResponse(content={"url": checkout_session.url})
    except Exception as e:
        logger.error(f"Error during create-checkout-session process: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-portal-session")
async def customer_portal(body: dict):
    try:
        checkout_session_id = body.get('session_id')
        checkout_session = stripe.checkout.Session.retrieve(checkout_session_id)

        return_url = YOUR_DOMAIN

        portal_session = stripe.billing_portal.Session.create(
            customer=checkout_session.customer,
            return_url=return_url,
        )
        return JSONResponse(content={"url": portal_session.url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook")
async def webhook_received(request: Request):
    webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    if not webhook_secret:
        logger.error("STRIPE_WEBHOOK_SECRET environment variable is not set")
        raise HTTPException(status_code=500, detail="Webhook secret not configured.")

    payload = await request.body()  # Ensure raw body is used
    sig_header = request.headers.get('stripe-signature')

    if not sig_header:
        logger.error("Missing stripe-signature header.")
        raise HTTPException(status_code=400, detail="Missing stripe-signature header.")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig_header, secret=webhook_secret
        )
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload.")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature.")

    event_type = event['type']
    data_object = event['data']['object']

    logger.info(f"Received event type: {event_type}")

    try:
        if event_type == 'checkout.session.completed':
            # Add or update subscription in the user_subscriptions table
            session_id = data_object['id']
            customer_id = data_object['customer']
            subscription_id = data_object.get('subscription', None)
            start_date = data_object.get('created', None)

            # Check if the subscription already exists
            subscription_check_response = requests.get(f"{SUPABASE_URL}/rest/v1/user_subscriptions", headers=headers, params={"id": f"eq.{subscription_id}"})

            if subscription_check_response.status_code == 200 and subscription_check_response.json():
                logger.info(f"Subscription already exists: {subscription_id}")
            else:
                # Add new subscription to the user_subscriptions table
                subscription_data = {
                    "id": subscription_id,
                    "user_id": customer_id,
                    "start_date": start_date,
                    "status": "active"
                }
                subscription_add_response = requests.post(f"{SUPABASE_URL}/rest/v1/user_subscriptions", headers=headers, json=subscription_data)

                if subscription_add_response.status_code not in [200, 201]:
                    logger.error(f"Failed to add subscription to Supabase: {subscription_add_response.text}")
                    raise HTTPException(status_code=500, detail="Failed to add subscription to database.")

                logger.info(f"Subscription added to Supabase: {subscription_data}")

        elif event_type == 'customer.subscription.trial_will_end':
            logger.info('Subscription trial will end')
        elif event_type == 'customer.subscription.created':
            logger.info(f'Subscription created: {data_object}')
        elif event_type == 'customer.subscription.updated':
            logger.info(f'Subscription updated: {data_object}')
        elif event_type == 'customer.subscription.deleted':
            logger.info(f'Subscription canceled: {data_object}')
        elif event_type == 'customer.created':
            # Add or update user in the users table
            user_id = data_object['id']
            user_email = data_object.get('email', 'unknown')

            # Check if the user already exists
            user_check_response = requests.get(f"{SUPABASE_URL}/rest/v1/users", headers=headers, params={"id": f"eq.{user_id}"})

            if user_check_response.status_code == 200 and user_check_response.json():
                logger.info(f"User already exists: {user_id}")
            else:
                # Add new user to the users table
                user_data = {
                    "id": user_id,
                    "email": user_email,
                    "created_at": data_object.get('created', None)
                }
                user_add_response = requests.post(f"{SUPABASE_URL}/rest/v1/users", headers=headers, json=user_data)

                if user_add_response.status_code not in [200, 201]:
                    logger.error(f"Failed to add user to Supabase: {user_add_response.text}")
                    raise HTTPException(status_code=500, detail="Failed to add user to database.")

                logger.info(f"User added to Supabase: {user_data}")
        else:
            logger.warning(f"Unhandled event type: {event_type}")
    except Exception as e:
        logger.error(f"Error handling event {event_type}: {e}")
        raise HTTPException(status_code=500, detail="Error handling event.")

    return {"status": "success"}

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)