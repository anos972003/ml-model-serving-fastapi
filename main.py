from fastapi import FastAPI
from fastapi import Request
from typing import Union

app=FastAPI()

@app.get("/")
def read_root():
    #return {"hell": 'world'}
    return "hello here"


@app.get("/items/{item_id}")
def read_item(item_id: int, q:Union[str,None]=None):
    return {"item_id":item_id, "q":q}



@app.get('/get_sentiment/{text},{ip}')
def get_sentiment(text:str,ip:str, user_id:Union[str,None]=None):
    return{"ip":ip
            ,'text': text,
           'sentiment':'positive'
           ,'user_id':user_id}

@app.post("/get_twitter_sentiment")
def get_twitter_sentiment(text : str, ip:str, user_id:Union[str,None]=None):
    return{"ip":ip
            ,'text': text,
           'sentiment':'normal'
           ,'user_id':user_id}

@app.post("/get_twitter_sentiment_v2")
async def get_twitter_sentiment_v2(request:Request):
    data=await request.json()

    text=data.get('text')
    ip = data.get('ip')
    user_id = data.get("user_id")
    return {"ip":ip
            ,'text': text,
           'sentiment':'normal'
           ,'user_id':user_id}