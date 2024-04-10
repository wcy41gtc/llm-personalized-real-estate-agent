import numpy as np
import pandas as pd
import random
import openai
import re
from openai import OpenAI
import lancedb
from lancedb.pydantic import vector, LanceModel

prompt = """
You are a real estate agent, there are many properties on the market for sale, including condo, apartment, house, ranch, and mansion.
The listing must include: neighborhood name, property type, price, size, number of bedrooms, number of bathrooms, description, neighborhood description.
Use the following format:

Neighborhood name:
Property type:
Price:
Size:
Number of bedrooms:
Number of bathrooms:
Description:
Neighborhood Description:

Generate a description for a {} listed for sale:
"""

property_types = ["condo", "apartment", "house", "ranch", "mansion"]

def generate_listing_description(prompt):
    """
    Using custom prompt to generate property listings
    from gpt-3.5-turbo model. 
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
            temperature = 1.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        return ""

def generate_listings(num_listings, openai_api_key, prompt, property_types):
    """
    generate a number (num_listings) of listings using generate_listing_description
    parse the response string into a dictionary
    storm the listing dictionaries into a Pandas dataframe
    """
    listings = []
    success = False
    keys = ["neighborhood name",
            "property type",
            "price",
            "size",
            "number of bedrooms",
            "number of bathrooms",
            "description",
            "neighborhood description"]
    for i in range(num_listings):
        property_type = random.choice(property_types)
        prompt = prompt.format(property_type)
        listing_str = generate_listing_description(prompt).lower()
        listing_dict = {key: "" for key in keys}
        current_key = None
        for line in listing_str.split('\n'):
            line_key = next((key for key in keys if line.startswith(key + ":")), None)
            if line_key:
                current_key = line_key
                listing_dict[current_key] = line.split(": ", 1)[1].strip()
            elif current_key:
                listing_dict[current_key] += " " + line.strip()
        listings.append(listing_dict)
    listings_df = pd.DataFrame(listings)
    return listings_df

def get_embedding(text):
    text = text.replace("\n", " ")
    model = "text-embedding-ada-002"
    return client.embeddings.create(input=[text], model=model).data[0].embedding

class PropertyListings(LanceModel):
    neighborhood_name: str
    property_type: str
    price: str
    size: str
    num_bedrooms: int
    num_bathrooms: float
    description: str
    neighborhood_description: str
    combined_description: str
    ada_embeddings: vector(1536)

average_buyer_prompt = """
You are a real estate buyer, you are interetsed in buying either a house, a condo, or an apartment, but you can only choose one.
A real estate agent is helping you choosing a property listing that best suits your preferences.

Answer the following questions:

Questions:
What kind of property are you interested in buying?
What kind of neighborhood would you like to live in?
How many bedrooms and bathrooms do you need?
What amenities would you like?

Answers:
"""

high_net_value_buyer_prompt = """
You are a high net value real estate buyer, you are interested in buying a mansion or a ranch, but you can only choose one.
A real estate agent is helping you choosing a property listing that best suits your preferences.

Answer the following questions:

Questions:
What kind of property are you interested in buying?
What kind of neighborhood would you like to live in?
How many bedrooms and bathrooms do you need?
What amenities would you like?

Answers:
"""

# create a list fo buyer prompts
buyer_prompt = [average_buyer_prompt, high_net_value_buyer_prompt]

client = OpenAI(api_key=openai_api_key)

def generate_buyer_preference(prompt):
    """
    Using custom prompt to generate property listings
    from gpt-3.5-turbo model. 
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
            temperature = 1.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        return ""
def clean_buyer_preference(response):
    if "\n\n" in response:
        response = response.split("\n\n")
    if "\n" in response:
        response = response.split("\n")
    if "?" in response:
        response = [s.strip() for s in response if "?" not in s]
    result_response = " ".join(response)
    # remove numerical bullet points in result_response
    result_response = re.sub(r'\d+\. ', '', result_response)
    # remove "I [text] " pattern
    cleaned_response = re.sub(r'I \w+ ', '', result_response)
    return cleaned_response

rag_prompt = """
You are a real estate agent.
Genearte a tailored description based on the context below, highlight the specific preferences in the context.
Do not change factual information including name, neighborhood, amenities, and location.

Context: 

{}

---

Description: 

{}

Tailored description:
"""