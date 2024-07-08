import dash
import io
import time
import uvicorn
import pandas as pd
import dash_bootstrap_components as dbc 
from fastapi import FastAPI
from employee_data_HLF import test_run,employee_api_mail
from dash import dcc
from dash import html
from fastapi import FastAPI, HTTPException, UploadFile, File


fast_app = FastAPI()

@fast_app.get("/")
async def root():
    return {"message": "Welcome to Timesheet Analysis!"}

@fast_app.post("/timesheet/")
async def admin(timesheet_file: UploadFile = File(...),month: str = ''):
    content = await timesheet_file.read()  # Read the content of the uploaded file
    # content2 = await admin_file.read()
    # df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
    # emp_df = pd.read_excel(io.BytesIO(content2), engine='openpyxl')
    message = employee_api_mail(content,month)
    
    return {"File_path":message}

if __name__ == '__main__':
    uvicorn.run(fast_app, host="127.0.0.1", port=8001)