import dash
import io
import time
import uvicorn
import pandas as pd
import dash_bootstrap_components as dbc 
from fastapi import FastAPI
# from dash_app_selenium import test_run
# from dash_app_selenium_new_format import test_run
from dash_app_selenium_new_format_department import test_run
from dash import dcc
from dash import html
from fastapi import FastAPI, HTTPException, UploadFile, File



fast_app = FastAPI()

@fast_app.get("/")
async def root():
    return {"message": "Welcome to Timesheet Analysis!"}

@fast_app.post("/timesheet/")
async def admin(timesheet_file: UploadFile = File(...),admin_file:UploadFile = File(...), month: str = ''):
    content = await timesheet_file.read()  # Read the content of the uploaded file
    content2 = await admin_file.read()
    df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
    df = df[df['DEPARTMENT'] != 'Office Boy']
    df = df[df['DEPARTMENT'] != 'OPS']
    
    
    exclude_emp = ['ASIF AHMED MOHAMMAD','PARDHA SARADHI. KATTA','TATI SASI KRISHNA','KOTI AZMIRA',
                'KHAJA SALEEMUDDIN','M CHENNAKESHAVULU','MOHAMMED SHOUKAT ALI','VENKAYALA V. UDAY KIRAN','SYED IMTIAZ','MOHD AKHEEL','NAZIMA ERSHAD']
    
    df = df[~df['EMPLOYEE NAME'].isin(exclude_emp)].reset_index(drop = True)
    df = df[~df['EMPLOYEE NAME'].isin(exclude_emp)].reset_index(drop = True)    
    emp_df = pd.read_excel(io.BytesIO(content2), engine='openpyxl')
    message = test_run(df,emp_df,month)
    
    return {"File_path":message}

if __name__ == '__main__':
    uvicorn.run(fast_app, host="127.0.0.1", port=8000)