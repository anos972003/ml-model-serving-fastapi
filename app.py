from scripts.data_model import NLPDataOutput,NLPDataInput,ImageDataInput,ImageDataOutput
from scripts import s3
from fastapi import FastAPI
from fastapi import Request
from typing import Union
import uvicorn
import time
import os
from transformers import pipeline 
import torch
from transformers import AutoImageProcessor #-> like tokenizer
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import warnings

warnings.filterwarnings('ignore')

model_ckpt = 'google/vit-base-patch16-224-in21k'
image_processor= AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)

app=FastAPI()
device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

##### Download ML Models####
force_download=False

model_name='tinybert-sentiment-analysis/'
local_path='ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
sentiment_model=pipeline("text-classification",model=local_path,device=device)


model_name='tinybert-disaster-tweets/'
local_path='ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
twitter_model=pipeline("text-classification",model=local_path,device=device)


model_name='vit-human-pose-classification/'
local_path='ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
pose_model=pipeline("image-classification",model=local_path,device=device, image_processor=image_processor)

####### Download Ends ######



@app.get("/")
def read_root():
    return "ehllo im up"

@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data : NLPDataInput):
    start = time.time()
    output=sentiment_model(data.text)
    end=time.time()
    prediction_time = end-start

    labels= [x['label'] for x in output]
    scores= [x['score'] for x in output]
    output=NLPDataOutput(model_name="tinybert-sentiment-analysis",
                         text=data.text,
                         labels=labels,
                         scores=scores,
                         prediction_time=prediction_time)
    return output

@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data : NLPDataInput):
    start=time.time()
    output=twitter_model(data.text)
    end=time.time()
    prediction_time=int((end-start)*1000)
    labels= [x['label'] for x in output]
    scores= [x['score'] for x in output]
    output=NLPDataOutput(model_name="tinybert-disaster-tweet",
                         text=data.text,
                         labels=labels,
                         scores=scores,
                         prediction_time=prediction_time)
    return output

@app.post("/api/v1/pose_classifier")
def pose_classifier(data : ImageDataInput):
    start=time.time()
    image_urls=[str(url)for url in data.url]
    output=pose_model(image_urls)
    end=time.time()
    prediction_time=int((end-start)*1000)
    labels= [x[0]['label'] for x in output]
    scores= [x[0]['score'] for x in output]
    output=ImageDataOutput(model_name="vit-pose-analysis",
                         url=data.url,
                         labels=labels,
                         scores=scores,
                         prediction_time=prediction_time)
    return output



if __name__=="__main__":
    uvicorn.run(app="app:app", port=8000, reload=True)