import uvicorn
from models.model import *
from contextlib import asynccontextmanager
from fastapi import FastAPI, Body
from joblib import load
import pandas as pd

pd.options.mode.chained_assignment = None

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models['clf'] = load('models/new_clf_th042_auc_7953.joblib')
    models['metarf'] = ProcessMetarMetaf()
    yield
    models['clf'] = None

app = FastAPI(title='FastAPI', 
                description='API for flight delay prediction', 
                version='1.0',
                lifespan=lifespan,
                # debug=True,
                )

@app.post('/predict')
async def get_prediction(data):
    data = pd.read_json(data)
    data = models['metarf'].transform(data)
    proba = models['clf'].predict_proba(data)[:,1].tolist()
    # predictions = (proba > PREDICT_THRESHOLD).astype(int).tolist()
    return {
        # 'predictions': predictions,
        'proba': proba,
    }

if __name__ == '__main__':
    uvicorn.run(uvicorn.run(app, host='127.0.0.1', workers=1))