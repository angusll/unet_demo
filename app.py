import os
import glob
import sys
import base64
import json
import tensorflow as tf
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from datetime import datetime
from google.cloud import aiplatform

def endpoint_predict_sample(project: str, location: str, instances: list, endpoint: str):
    
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint)

    prediction = endpoint.predict(instances=instances)
    print(prediction)
    return prediction

def post_process(y_hat):
    img = np.squeeze(y_hat)
    return img

def request_model_api(img_array):

    img = base64.urlsafe_b64encode(img_array.ravel()).decode()# urlsafe_b64encode for tf.io.decode_base64
    request_data = [{'image':{'b64':img}}]
    
    with st.spinner("Sending post request..."):

#        response = requests.post(api_url, json.dumps(request_data), headers=headers,timeout=120)
        pred_response = endpoint_predict_sample(project="1056569535305",
                                        location="asia-east1",
                                        endpoint = "7937066577659166720",
                                        instances=request_data)
    st.success('Done!')

    try: 
        pred_bytes = pred_response.predictions
        mask = tf.reshape(tf.io.decode_raw(base64.urlsafe_b64decode(pred_bytes),tf.float32),[320,320])
        return  mask
    except:
        return st.write('Request Time out or error occured') # temp escape from time out or api error



@st.cache
def load_model(fp):
    if not os.path.exists(fp):
        download_model(fp)
    try:
        model = tf.keras.models.load_model(fp)    
    except ValueError:
        model = tf.keras.models.load_model(fp, compile=False)
        model.compile(loss='rmse') # dummy loss
    return model 

@st.cache
def download_model(fp):
    url = f'https://storage.googleapis.com/scancer/model_weights/{fp}'
    r = requests.get(url, allow_redirects=True)
    with open(fp , 'wb') as f:
        f.write(r.content)


def send_resposne(reaction=""):

    from google.cloud import bigquery
    # https://cloud.google.com/bigquery/docs/samples/bigquery-table-insert-rows#bigquery_table_insert_rows-python
    # Construct a BigQuery client object.
    client = bigquery.Client()

    table_id = 'scancer.unet_app_metrics.metrics'
    rows_to_insert = [
        {u"reaction": reaction},
    ]

    errors = client.insert_rows_json(table_id, rows_to_insert)  # Make an API request.
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))


def show_input_output(tile_raw,output)

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(tile_raw)
    axes[1].imshow(output)

    axes[0].set_title("Input")
    axes[1].set_title("Prediction")

    axes[0].axis(AXIS)
    axes[1].axis(AXIS)

    st.write(fig)

    
# config
AXIS = 'off'

# main
st.title("AE13")
models = ["H&E 128x128 Baseline", 
            "H&E 128x128 Baseline2",
            "IHC model api"]
model_di = {"H&E 128x128 Baseline":"test_deploy_model.h5",
            "H&E 128x128 Baseline2":"epoch_100_0.0840_128_128_this_works.h5",
            "IHC model api":"IHC model api"}

# defaults to model with API for faster init time
MODEL_FP = st.selectbox("Select model", models, index=2)
MODEL_FP = model_di[MODEL_FP]


sample_img = np.array(Image.open('sample_data.png'))
tile_raw = sample_img[...,:3]
st.image(tile_raw, caption='Sample Data', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

if st.button("Use sample data to demo"):
    tile = sample_img
    output = request_model_api(tile)
    show_input_output(tile_raw,output)
    
file_obj = st.file_uploader("Upload tile", type=['png','jpg'])

if file_obj is not None:
    # io
    tile = Image.open(file_obj)
    tile_raw = tile.copy()
    tile = tile.resize((500,500))
    tile = np.array(tile)[:,:,:3]# remove alpha channel
    output = request_model_api(tile)
    show_input_output(tile_raw,output)
