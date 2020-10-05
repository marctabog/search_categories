#web frameworks
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import uvicorn
import aiohttp
import asyncio

import os
import sys
import base64 
	
# search frameworks

import fasttext.util
import pandas as pd
import numpy as np
from scipy import spatial
import pickle

# Define functions

def get_fasttext_sent_embeed(sent):
    words = sent.split(' ')
    embeedings = np.array([ft.get_word_vector(x) for x in words])
    sent_embeed = embeedings.mean(axis=0)
    return sent_embeed

def extract_category(x,categories):
    for cat in categories:
        if cat in x:
            return cat

def perform_search(request,embeddings,df,n_results=10):
    #request_embedding = get_fasttext_sent_embeed(request)
    request_embedding = dico[request]
    scores = []
    for item in embeddings:
        scores.append(spatial.distance.cosine(request_embedding, item))

    top_n = np.array(scores).argsort()[:10]
    results_to_display = []
    for item in top_n:
        res= df['text'].iloc()[item]
        category = extract_category(res,categories)
        #print('Category = {}'.format(category))
        subcategory = res.replace(category,'')
        #print('Sub category = {}'.format(subcategory))
        results_to_display.append((category,subcategory))
        #print('')
        
    show_df = pd.DataFrame(results_to_display)
    show_df.columns = ['Category','Sub category']
    return show_df

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

# Define variables
#ft = fasttext.load_model('cc.en.300.bin')
df = pd.read_csv('data_model.csv')

with open('embeddings.pickle', 'rb') as handle:
    embeddings = pickle.load(handle)
with open('dico_general.pickle', 'rb') as handle:
    dico_general = pickle.load(handle)

with open('dico.pickle', 'rb') as handle:
    dico = pickle.load(handle)
categories = list(dico_general.values())

app = Starlette()

@app.route("/classify-url", methods = ["GET"])
async def classify_url(request):
    req = request.query_params["request"]
    perform_search(req,embeddings,df)
    return predict_image_from_bytes(req)

def predict_image_from_bytes(request):
    
    return HTMLResponse(
        """
        <html>
            <body>
                <p> Prediction: <b> %s </b> </p>
            </body>
        </html>
        """ %(request))
        
@app.route("/")
def form(request):
        return HTMLResponse(
            """
            <h1> Category / Subcategory prediction test </h1>
            <br>
            <u> Please enter your request </u>
            <form action = "/classify-url" method="get">
                1. <input name="request" size="60"><br><p>
                2. <input type="submit" value="Search">
            </form>
            """)
        
@app.route("/form")
def redirect_to_homepage(request):
        return RedirectResponse("/")
        
if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008)) 
        uvicorn.run(app, host = "0.0.0.0", port = port)
