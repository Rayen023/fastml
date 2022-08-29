# Bring in dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
#pydantic guarantees that the fields of the resultant model instance will conform to the field types defined on the model.

app = FastAPI()

class Scoring_item(BaseModel) : 
    YearsAtCompany : float
    EmployeeSatisfaction : float
    Position : str 
    Salary : int

with open('rfmodel.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item : Scoring_item):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction" : float(yhat)}