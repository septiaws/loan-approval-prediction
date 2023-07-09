from fastapi import FastAPI
from fastapi import Request
import pickle

app = FastAPI()

@app.get('/')
async def root ():
    response = {
        "status": 200,
    }
    return response


def load_model():
    try:
        pickle_file = open('../models/production_model.pkl', 'rb')
        prediction = pickle.load(pickle_file)
        return prediction
    except Exception as e:
        response = {
            "status": 204,
            "message": str(e)
        }
        return response

@app.get('/check')
async def check():
    model = load_model()
    return model

@app.post('/predict')
async def predict (data: Request):

    # load request
    data = await data.json()
    
    ApplicantIncome = data["ApplicantIncome"]
    Gender = data["Gender"]
    Married = data["Married"]
    Dependents = data["Dependents"]
    Education = data["Education"]
    Self_Employed = data["Self_Employed"]
    Property_Area = data["Property_Area"]
    CoapplicantIncome = data["CoapplicantIncome"]
    LoanAmount = data["LoanAmount"]
    Loan_Amount_Term = data["Loan_Amount_Term"]
    Credit_History = data["Credit_History"]
    

    model = load_model()
    labels = ['Pengajuan Diterima', 'Pengajuan Ditolak']
    try:
        prediction = model.predict([[ApplicantIncome, Gender, Married, Dependents, Education, Self_Employed, Property_Area, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History]])
        result = labels[prediction[0]]
        response = {
            "status": 200,
            "prediction": result
        }
    except Exception as e:
        response = {
            "status": 204,
            "message": str(e)
        }
    return response

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8000)
    