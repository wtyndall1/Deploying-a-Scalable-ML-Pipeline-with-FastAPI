import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


PROJECT_PATH = os.getcwd()

path = os.path.join(PROJECT_PATH, "model", "encoder.pkl")  # TODO: enter the path for the saved encoder  - done
encoder = load_model(path)

path = os.path.join(PROJECT_PATH, "model", "model.pkl")  # TODO: enter the path for the saved model  - done
model = load_model(path)

# TODO: create a RESTful API using FastAPI - done
app = FastAPI(title="Census Income Classifier API")

# TODO: create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    return {"message": "Hello from the Census Income Classifier API."}


# TODO: create a POST on a different path that does model inference - done
@app.post("/data/")
async def post_inference(data: Data):
      # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
      # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
      # The data has names with hyphens and Python does not allow those as variable names.
      # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
          # do not need to pass lb as input
    )
    _inference = inference(model, data_processed)  # your code here to predict the result using data_processed - done
    return {"result": apply_label(_inference)}