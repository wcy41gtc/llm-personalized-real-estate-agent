import numpy as np
import pandas as pd #
import random
import openai
from openai import OpenAI
import re
import gradio as gr
import lancedb
from lancedb.pydantic import vector, LanceModel


############### Customized Prompts for LLM ################
property_types = ["condo", "apartment", "house", "ranch", "mansion"]
agent_prompt = """
You are a real estate agent, there are many properties on the market for sale, including condo, apartment, house, ranch, and mansion.
The listing must include: neighborhood name, property type, price, size, number of bedrooms, number of bathrooms, description, neighborhood description.
Use the following format:

Neighborhood name (be creative):
Property type:
Price:
Size:
Number of bedrooms:
Number of bathrooms:
Description:
Neighborhood Description (be creative):

Generate a description for a {} listed for sale:
"""
buyer_prompt = """
You are a real estate buyer, you are interetsed in buying a {}.
A real estate agent is helping you choosing a property listing that best suits your preferences.

Answer the following questions:

Questions:
What kind of property are you interested in buying?
What kind of neighborhood would you like to live in?
How many bedrooms and bathrooms do you need?
What amenities would you like?

Answers:
"""
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
############### ************************* ################

with open('openai_api_key.txt', 'r') as file:
    openai_api_key = file.read()
client = OpenAI(api_key=openai_api_key)

###############          Methods          ################
def generate_listing_description(agent_prompt):
    """
    Using custom prompt to generate property listings
    from gpt-3.5-turbo model. 
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": agent_prompt,
                }
            ],
            model="gpt-3.5-turbo",
            temperature = 0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        return ""

def generate_listings(num_listings, agent_prompt, property_types):
    """
    Generate a number (num_listings) of listings using generate_listing_description
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
        property_type = property_types[int(np.mod(i,5))]
        prompt = agent_prompt.format(property_type)
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
    """
    Get embedding for input text using openai text-embedding-ada-002 model
    """
    text = text.replace("\n", " ")
    model = "text-embedding-ada-002"
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def generate_buyer_preference():
    """
    Using custom prompt to generate property listings
    from gpt-3.5-turbo model. 
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": buyer_prompt.format(random.choice(property_types)),
                }
            ],
            model="gpt-3.5-turbo",
            temperature = 0.5
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
    # remove "-" in result_response
    result_response = re.sub(r'- ', '', result_response)
    # remove "I [text] " pattern
    cleaned_response = re.sub(r'I \w+ ', '', result_response)
    return cleaned_response

def search_listings(table, preference, num:int):
    """
    inputs:
        listings_df (Pandas DataFrame): listings database
        query (str): customer preference
        num (int): number of recommendations
    outputs:
        reco_df (Pandas DataFrame): recommendations 
    
    """
    cleaned_preference = clean_buyer_preference(preference)
    embedding = get_embedding(cleaned_preference)
    reco_df = table.search(embedding).limit(num).to_pandas()
    return reco_df

def generate_custom_listing_description(preference, num):
    reco_df = search_listings(table, preference, num)
    agent_reco = generate_listing_description(rag_prompt.format(preference, reco_df.iloc[0]["combined_description"]))
    reco_df.drop(['combined_description', 'ada_embeddings', '_distance'], axis=1, inplace=True)
    return agent_reco, reco_df

###############          LanceDB class          ################

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

###############          Gradio App          ################
def app():
    import sys
    num_listings = 20
    # generate some listings
    try:
        print("Reading listings from pre-generated file...")
        listings_df = pd.read_pickle("./listings_with_embedding.pkl")
        print("Reading successfull.")
    except:
        print("Pre-generated listings not found, generating {} listings using openai gpt-3.5-turbo...".format(num_listings))
        listings_df = generate_listings(num_listings, agent_prompt, property_types)
        print("Generated {} listings.".format(num_listings))
        # concatenate description and neighborhood description into one column and calculating embeddings
        print("Getting embeddings for listings using text-embedding-ada-002 model.")
        listings_df["combined_description"] = listings_df["description"].str.cat(listings_df["neighborhood description"], sep=" ")
        listings_df["ada_embeddings"] = listings_df["combined_description"].apply(get_embedding)
        print("Generated embeddings.")
        listings_df.to_pickle("./listings_with_embedding.pkl")

    # create vector data base using lancedb
    print("Creating lanceDB vector database...")
    db = lancedb.connect("./.lancedb")
    table_name = "property_listings"
    global table
    table = db.create_table(table_name, schema = PropertyListings, mode="overwrite")

    # update column name to match lancedb schema
    listings_df.rename(columns={"neighborhood name": "neighborhood_name",
                                "property type": "property_type",
                                "number of bedrooms": "num_bedrooms",
                                "number of bathrooms": "num_bathrooms",
                                "neighborhood description": "neighborhood_description"
                               }, inplace=True)
    table.add(listings_df)
    print("Database creation successful.")
    
    def startup():
    # Automatically generate a sample buyer preference
        initial_preference = generate_buyer_preference()
        return initial_preference
        
    def update_results_label(value):
        label = f"Showing Top {value} Results"
        return gr.Dataframe(label=label, show_label=True)
    
    def close_app():
        print("Closing the application.")
        demo.close()
        sys.exit()

    initial_pref = startup()

    listings_df = pd.read_csv("./listings.csv",index_col=0)
    print("Starting Gradio app...")
    with gr.Blocks() as demo:
        with gr.Row():
            client_pref = gr.Textbox(value=initial_pref, label="Client Preference",show_label=True)
        with gr.Row():
            b1 = gr.Button("Generate Client Preferences")
            b2 = gr.Button("Submit")
            num = gr.Slider(minimum=1, maximum=10, label="Number of Results", step=1)
            b3 = gr.Button("Close App")
        with gr.Row():
            agent_reco = gr.Textbox(value="", label="Agent Recommendation",show_label=True)
        with gr.Row():
            reco_list = gr.Dataframe(label=f"Showing Top {num.value} Results", show_label=True)
        with gr.Row():
            avail_listings = gr.Dataframe(listings_df, label="Available Listings", show_label=True)
        num.change(update_results_label, inputs=num, outputs=reco_list)
        b1.click(generate_buyer_preference, outputs=client_pref)
        b2.click(generate_custom_listing_description, inputs=[client_pref, num], outputs=[agent_reco, reco_list])
        b3.click(close_app)
    
    demo.launch()

###############          Main          ################
def main():
    app()

if __name__ == "__main__":
    main()
    