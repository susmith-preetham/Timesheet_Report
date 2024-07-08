import dash
import io
import ssl
import time
import math
import base64
import smtplib
import warnings
import threading
import multiprocessing
import os, signal
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc 
from email_table_process_HLF import emp_table_data
from tqdm import tqdm
from dash import dcc
from dash import html
from datetime import datetime
from datetime import timedelta
from dash.dependencies import Input, Output
from selenium import webdriver
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pretty_html_table import build_table
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.print_page_options import PrintOptions
warnings.filterwarnings('ignore')



def removing_unnecessary_values(df):
    columns = []
    current_date_column = None

    for i, column in enumerate(df.columns):
        if i == 0:
            columns.append(column)
        elif 'Unnamed' in str(column):
            columns.append(f"{current_date_column} {df.iloc[1][column]}")
        else:
            current_date_column = column
            temp_date_column = f"{column} {df.iloc[1][column]}"
            columns.append(temp_date_column)

    df.columns = columns
    df = df.drop([0, 1]).reset_index(drop=True)
    return df

def adjust_decimal_values(value):
    if value % 1 > 0.60:
        adjusted_value = value - 0.40
        adjusted_value = round(adjusted_value,2)
        number = str(adjusted_value).split('.')[0]
        decimal = str(adjusted_value).split('.')[1]
        if len(decimal)<2:
            decimal = int(decimal)/6
        else:
            decimal = int(decimal)/60
        adjusted_value = float(number) + decimal
        adjusted_value = round(adjusted_value,2)
    else:
        adjusted_value = value
        adjusted_value = round(adjusted_value,2)
        if len(str(adjusted_value))<=2:
            return adjusted_value
        else:
            number = str(adjusted_value).split('.')[0]
            decimal = str(adjusted_value).split('.')[1]
            if len(decimal)<2:
                decimal = int(decimal)/6
            else:
                decimal = int(decimal)/60
            adjusted_value = float(number) + decimal
            adjusted_value = round(adjusted_value,2)
    return adjusted_value

def float_to_string(value):
    if len(str(value))<=2:
        value = str(value) + ".00"
    elif len(str(value))==4 and str(value).startswith('1'):
        value = str(value)
        value = value + '0'
    elif len(str(value))==3:
        value = str(value)
        value = value + '0'
    else:
        value = str(value)
    return value

def calculate_time_difference(time_str1, time_str2):
    time_str1 = float_to_string(time_str1)
    time_str2 = float_to_string(time_str2)
    # Assuming time values are in the format "HH.MM"
    time_format = "%H.%M"

    # Convert time strings to datetime objects
    time1 = pd.to_datetime(time_str1, format=time_format)
    time2 = pd.to_datetime(time_str2, format=time_format)

    # Calculate time difference
    time_difference = time2 - time1

    # Extract hours and minutes from the timedelta object
    hours = time_difference.seconds // 3600
    minutes = (time_difference.seconds % 3600) // 60

    total_time = hours + minutes / 60
    return round(total_time,2)

def columns_name_change(dataframe):
    col_lst = dataframe.columns.to_list()
    new_col = []

    # Remove "00:00:00" from column names
    for col in col_lst:
        new_col.append(col.replace("00:00:00", "").replace(" nan", ""))

    dataframe.columns = new_col
    return dataframe

def decimal_correct(value):
    number = str(value).split('.')[0]
    decimal = str(value).split('.')[1]
    if len(decimal) == 1:
        decimal = decimal +"0"
    value = number +  "."+ decimal
    return value

def min_to_hrs(value):
    if value<0:
        value = abs(value)
        if len(str(value))<=2:
            hrs_value = round(value,2)
        else:
            number = str(value).split('.')[0]
            decimal = str(value).split('.')[1]
            if len(str(decimal))==1:
                decimal = str(decimal) + "0"
            hrs_value = int(number) + (int(decimal)/60)
    else:
        if len(str(value))<=2:
            hrs_value = round(value,2)
        else:
            number = str(value).split('.')[0]
            decimal = str(value).split('.')[1]
            if len(str(decimal))==1:
                decimal = str(decimal) + "0"
            hrs_value = int(number) + (int(decimal)/60)
            
    return hrs_value

def hrs_to_min(value):
    if value<0:
        value = abs(value)
        if len(str(value))<=2:
            adjusted_value = value
            adjusted_value = round(adjusted_value,2)
        else:
            number = str(value).split('.')[0]
            decimal = str(value).split('.')[1]
            decimal = float("0." + decimal)*0.6
            adjusted_value = int(number)+decimal
            adjusted_value = round(adjusted_value,2)
            adjusted_value = -adjusted_value
    else:
        if len(str(value))<=2:
            adjusted_value = value
            adjusted_value = round(adjusted_value,2)
        else:
            number = str(value).split('.')[0]
            decimal = str(value).split('.')[1]
            decimal = float("0." + decimal)*0.6
            adjusted_value = int(number)+decimal
            adjusted_value = round(adjusted_value,2)
            adjusted_value = adjusted_value       
    return adjusted_value

def half_day_in(in_time_hlf):
    if 12.3<=in_time_hlf<=13.3:
        in_time = 9
    elif in_time_hlf>13.3:
        in_time = min_to_hrs(in_time_hlf)
        in_time = (in_time - 13.5) + 9
        in_time = hrs_to_min(in_time)
        
    else:
        in_time = in_time_hlf
    return in_time

def half_day_out(out_time_hlf):
    if 12.3<=out_time_hlf<=14.3:
        out_time = 18.3
    elif out_time_hlf<12.3:
        out_time = min_to_hrs(out_time_hlf)
        out_time = (12.5-out_time_hlf) + 18.5
        out_time = hrs_to_min(out_time)
        
    else:
        out_time = out_time_hlf
    return out_time

def create_dataframe(dataframe):
    dataframe["Grand Total Actual"] = np.nan
    dataframe["Grand Total Final"] = np.nan
    dataframe["Grand Total Grace Period"] = np.nan
    dataframe["Grand Total Worked"] = np.nan    
    dataframe.rename(columns=lambda x: x.strip() if isinstance(x, str) else x, inplace=True) # added remove if needed
    # print("create_dataframe",dataframe.columns)
    
    # Melt the original dataframe to transform it into a long format
    # melted_df = pd.melt(dataframe, id_vars=['EMPLOYEE NAME'], var_name='datetime_type', value_name='value')
    melted_df = pd.melt(dataframe, id_vars=['EMPLOYEE NAME','EMPLOYEE MAIL'], var_name='datetime_type', value_name='value') # changed

    # Extract date, time, and type from the 'datetime_type' column
    melted_df[['date', 'type']] = melted_df['datetime_type'].str.split(' ', n=1, expand=True)

    # Pivot the melted dataframe to get the desired format
    # final_df = melted_df.pivot_table(index=['EMPLOYEE NAME', 'date'], columns='type', values='value', aggfunc='first').reset_index()
    final_df = melted_df.pivot_table(index=['EMPLOYEE NAME','EMPLOYEE MAIL', 'date'], columns='type', values='value', aggfunc='first').reset_index() # changed

    # Rename columns
    final_df.columns.name = None
    # final_df.columns = ['EMPLOYEE NAME', 'DATE', 'IN TIME', 'OUT TIME', 'TOTAL']
    final_df.columns = ['EMPLOYEE NAME', 'EMPLOYEE MAIL', 'DATE', 'IN TIME', 'OUT TIME', 'TOTAL'] # changed
    
    final_df["TOTAL"] = final_df.apply(lambda row: 0 if row['IN TIME'] == 'L' else row['TOTAL'], axis=1)
    final_df["IN TIME"] = final_df["IN TIME"].apply(lambda x: np.nan if x == 'L' else x)
    final_df["OUT TIME"] = final_df["OUT TIME"].apply(lambda x: np.nan if x == 'L' else x)
    final_df["IN TIME"] = final_df["IN TIME"].astype(float)
    final_df["OUT TIME"] = final_df["OUT TIME"].astype(float)
    final_df["IN TIME"] = final_df["IN TIME"].apply(half_day_in)
    final_df["OUT TIME"] = final_df["OUT TIME"].apply(half_day_out)
    final_df["DATE"] = pd.to_datetime(final_df["DATE"])
    final_df["IN TIME"] = final_df["IN TIME"].astype(float)
    final_df["OUT TIME"] = final_df["OUT TIME"].astype(float)
    final_df["TOTAL"] = final_df["TOTAL"].astype(float)  
    final_df["TOTAL"] = final_df["TOTAL"].apply(lambda x : 0 if x<0 else x)
    # final_df["TOTAL"] = final_df["TOTAL"].apply(lambda x : adjust_decimal_values(x))
    final_df = final_df.dropna(thresh=0.85*len(final_df.columns))
    final_df["TOTAL"] = final_df.apply(lambda x : calculate_time_difference(x['IN TIME'],x['OUT TIME']),axis = 1)
    
    # Merge with the original 'Grand Total' columns
    final_df = final_df.merge(dataframe[['EMPLOYEE NAME', 'Grand Total Worked', 'Grand Total Grace Period', 'Grand Total Actual', 'Grand Total Final']], on='EMPLOYEE NAME', how='left')
    
    final_df["Grand Total Worked"] = final_df.groupby(["EMPLOYEE NAME"])["TOTAL"].transform('sum')
    final_df["Grand Total Actual"] = len(final_df["DATE"].unique())*9.5
    final_df["Grand Total Final"] = final_df["Grand Total Actual"] - final_df["Grand Total Worked"]
    return final_df

def add_week_no(dataframe):
    dataframe = dataframe.sort_values(by='DATE')

    # Extract year and month from the "DATE" column
    dataframe['YEAR'] = dataframe['DATE'].dt.year
    dataframe['MONTH'] = dataframe['DATE'].dt.month

    # Calculate week number within each month, considering the starting day of the week
    dataframe['WEEK'] = (dataframe['DATE'] - dataframe.groupby(['YEAR', 'MONTH'])['DATE'].transform('min')).dt.days // 7 + 1

    # Create the "WEEK" column with the desired format
    dataframe['WEEK'] = 'week' + dataframe['WEEK'].astype(str)

    # Drop the temporary columns if you don't need them
    dataframe = dataframe.drop(['YEAR', 'MONTH'], axis=1)
    return dataframe
def single_employee(dataframe,Employee_name):
    dataframe = dataframe[dataframe["EMPLOYEE NAME"] == Employee_name]
    return dataframe

def average_time(series):
    # Filter out np.nan values
    valid_times = [time for time in series if not math.isnan(time)]

    # Check if there are valid times
    if not valid_times:
        return np.nan  # Return np.nan if there are no valid times

    # Convert each time to minutes
    minutes_list = [(int(time) * 60) + (time % 1 * 100) for time in valid_times]

    # Calculate the average in minutes
    average_minutes = sum(minutes_list) / len(minutes_list)

    # Convert the average back to the original format (hours and minutes)
    average_hours = int(average_minutes / 60)
    average_minutes_remainder = int(average_minutes % 60)
    average_time = f"{average_hours:02d}:{average_minutes_remainder:02d}"

    return average_time
def new_columns(dataframe):
    dataframe["WEEK_AVG"] = dataframe.groupby(["EMPLOYEE NAME", "WEEK"])["TOTAL"].transform('mean')
    dataframe["WEEK_IN_TIME_AVG"] = dataframe.groupby(["EMPLOYEE NAME", "WEEK"])["IN TIME"].transform(average_time)
    dataframe["WEEK_OUT_TIME_AVG"] = dataframe.groupby(["EMPLOYEE NAME", "WEEK"])["OUT TIME"].transform(average_time)
    dataframe['Day'] = dataframe['DATE'].dt.day_name()
    dataframe["WEEK_TOTAL"] = dataframe.groupby(["EMPLOYEE NAME","WEEK"])["TOTAL"].transform('sum')
    dataframe["Average In Time"] = average_time(dataframe["IN TIME"])
    dataframe["Average Out Time"] = average_time(dataframe["OUT TIME"])
    dataframe["Average In Time"] = pd.to_datetime(dataframe["Average In Time"], format='%H:%M').dt.time
    dataframe["Average Out Time"] = pd.to_datetime(dataframe["Average Out Time"], format='%H:%M').dt.time
    dataframe["Avg Grand Total Worked"] = dataframe["Grand Total Worked"].mean()
    dataframe["Avg Grand Total Compensation"] = dataframe["Grand Total Final"].mean()
    dataframe["Avg Per day Work"] = dataframe["TOTAL"].mean()
    dataframe["AVG Day IN TIME"] = dataframe.groupby(["Day"])["IN TIME"].transform(average_time)
    dataframe["AVG Day OUT TIME"] = dataframe.groupby(["Day"])["OUT TIME"].transform(average_time)
    dataframe["AVG Date IN TIME"] = dataframe.groupby(["DATE"])["IN TIME"].transform(average_time)
    dataframe["AVG Date OUT TIME"] = dataframe.groupby(["DATE"])["OUT TIME"].transform(average_time)
    dataframe.reset_index(drop=True,inplace=True)
    return dataframe


def overall_avg_df(dataframe):
    df_avg_vals = dataframe[["Average In Time","Average Out Time","Avg Grand Total Worked","Grand Total Actual","Avg Per day Work","Avg Grand Total Compensation"]].drop_duplicates().reset_index(drop = True)
    df_avg_vals["Total Work Hrs"] = dataframe["Grand Total Worked"].sum()
    df_avg_vals["Total Actual Work Hrs"] = dataframe["Grand Total Actual"].sum()
    df_avg_vals["Total Pending Hrs"] = df_avg_vals["Total Actual Work Hrs"] - df_avg_vals["Total Work Hrs"]
    df_avg_vals["Actual In Time"] = "09:00"
    df_avg_vals["Actual In Time"] = pd.to_datetime(df_avg_vals["Actual In Time"], format='%H:%M').dt.time
    df_avg_vals["Actual Out Time"] = "18:30"
    df_avg_vals["Actual Out Time"] = pd.to_datetime(df_avg_vals["Actual Out Time"], format='%H:%M').dt.time
    df_avg_vals["Actual In Time_temp"] = df_avg_vals["Actual In Time"].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))
    df_avg_vals["Average In Time_temp"] = df_avg_vals["Average In Time"].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))
    # Calculate time difference
    df_avg_vals["In Time Difference"] = df_avg_vals["Average In Time_temp"] - df_avg_vals["Actual In Time_temp"]
    # Convert timedelta64 to hours and minutes
    df_avg_vals["In Time Difference"] = df_avg_vals["In Time Difference"].apply(lambda x: f"{int(x.seconds // 3600)} hr {int((x.seconds % 3600) // 60)} min")
    
    df_avg_vals["Actual Out Time_temp"] = df_avg_vals["Actual Out Time"].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))
    df_avg_vals["Average Out Time_temp"] = df_avg_vals["Average Out Time"].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))
    
    if df_avg_vals["Average Out Time_temp"][0] > df_avg_vals["Actual Out Time_temp"][0]:
        df_avg_vals["Out Time Difference"] = df_avg_vals["Average Out Time_temp"] - df_avg_vals["Actual Out Time_temp"]
    else:
        df_avg_vals["Out Time Difference"] = df_avg_vals["Actual Out Time_temp"] - df_avg_vals["Average Out Time_temp"]
    # Calculate time difference
    # df_avg_vals["Out Time Difference"] = df_avg_vals["Average Out Time_temp"] - df_avg_vals["Actual Out Time_temp"]

    # Convert timedelta64 to hours and minutes
    df_avg_vals["Out Time Difference"] = df_avg_vals["Out Time Difference"].apply(lambda x: f"{int(x.seconds // 3600)} hr {int((x.seconds % 3600) // 60)} min")
    
    new_column_order = ['Average In Time', 'Actual In Time', 'In Time Difference', 
                    'Average Out Time', 'Actual Out Time', 'Out Time Difference',
                    'Avg Grand Total Worked','Grand Total Actual', 'Avg Per day Work',
                    'Avg Grand Total Compensation', 'Total Work Hrs', 'Total Actual Work Hrs',
                    'Total Pending Hrs']
    
    df_avg_vals = df_avg_vals.iloc[:, df_avg_vals.columns.get_indexer(new_column_order)]
    
    return df_avg_vals
def day_wise_df(dataframe):
    day_df = dataframe[["Day","AVG Day IN TIME","AVG Day OUT TIME"]].drop_duplicates().reset_index(drop=True)
    return day_df
def date_wise_df(dataframe):
    date_df = dataframe[["DATE","AVG Date IN TIME","AVG Date OUT TIME"]].drop_duplicates().reset_index(drop=True)
    return date_df


def apply_all_functions(dataframe,employee_name):
    # Assuming 'df' is your DataFrame
    dataframe.rename(columns=lambda x: x.strip() if isinstance(x, str) else x, inplace=True) # added  this line because of some extra spaces that were present

    dataframe = removing_unnecessary_values(dataframe)
    dataframe = columns_name_change(dataframe)
    dataframe = create_dataframe(dataframe)
    dataframe = add_week_no(dataframe)
    dataframe = single_employee(dataframe,employee_name)
    dataframe = new_columns(dataframe)
    overall_avg = overall_avg_df(dataframe)
    day_wise = day_wise_df(dataframe)
    date_wise = date_wise_df(dataframe)
    return dataframe, overall_avg, day_wise, date_wise

def filter_emp_name(dataframe):
    dataframe.rename(columns=lambda x: x.strip() if isinstance(x, str) else x, inplace=True)
    dataframe = removing_unnecessary_values(dataframe)
    dataframe = columns_name_change(dataframe)
    dataframe = create_dataframe(dataframe)
    return dataframe

# def hrs_to_min(value):
#     if len(str(value))<=2:
#         adjusted_value = value
#         adjusted_value = round(adjusted_value,2)
#     else:
#         number = str(value).split('.')[0]
#         decimal = str(value).split('.')[1]
#         decimal = float("0." + decimal)*0.6
#         adjusted_value = int(number)+decimal
#         adjusted_value = round(adjusted_value,2)
#     return adjusted_value

def float_to_hrs_min(time_in_hrs_min):
    # hours_time, minutes_time = map(int, str(time_in_hrs_min).split('.'))
    hours_time = str(time_in_hrs_min).split('.')[0]
    try:
        minutes_time = str(time_in_hrs_min).split('.')[1]
    except:
        minutes_time = 0
    if len(str(minutes_time))==1:
        minutes_time = str(minutes_time) + "0"
    time_in_hrs_min = f"{hours_time} hrs and {minutes_time} min"
    return time_in_hrs_min

def plot_values(values):
    formatted_values = []
    for i in values:
        i = str(i)
        if '.' in i:
            number = i.split('.')[0]
            decimal = i.split('.')[1]
            if len(decimal) == 1:
                decimal = decimal + '0'
            else:
                decimal = decimal
            final_number = number + '.' + decimal
            formatted_values.append(final_number)
        else:
            formatted_values.append(i)
    return formatted_values

def count_working_days(year, month, day):
    # Get today's date
    today = datetime(year, month, day).date()
    
    # Calculate the number of days remaining in the month
    days_remaining = (datetime(year, month+1, 1).date() - today).days
    
    # Initialize a counter for working days
    working_days = 0
    
    # Iterate over each day remaining in the month
    for i in range(days_remaining):
        # Get the date of the current day
        current_date = today + timedelta(days=i)
        
        # Check if the current day is a weekday (0 = Monday, 6 = Sunday)
        if current_date.weekday() < 5: # Monday to Friday
            working_days += 1
    
    return working_days

def weeks_remaining_in_month():
    # Get the current date
    current_date = datetime.now()
    
    # Get the last day of the month
    last_day_of_month = datetime(current_date.year, current_date.month, 1) + timedelta(days=32)
    last_day_of_month = last_day_of_month.replace(day=1) - timedelta(days=1)
    
    # Calculate the remaining days in the month
    remaining_days = (last_day_of_month - current_date).days + 1
    
    # Calculate the number of remaining weeks
    remaining_weeks = remaining_days // 7
    
    return remaining_weeks

def working_days_in_last_week():
    # Get the current date
    current_date = datetime.now()
    
    # Get the last day of the month
    last_day_of_month = datetime(current_date.year, current_date.month, 1) + timedelta(days=32)
    last_day_of_month = last_day_of_month.replace(day=1) - timedelta(days=1)
    
    # Get the last week of the month
    last_week_start = last_day_of_month - timedelta(days=last_day_of_month.weekday())
    last_week_end = last_day_of_month
    
    # Calculate the number of working days in the last week
    working_days_count = 0
    for i in range((last_week_end - last_week_start).days + 1):
        day = last_week_start + timedelta(days=i)
        if day.weekday() < 5:  # Monday to Friday
            working_days_count += 1
    
    return working_days_count

def individual_employee(content,employee):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
    # Applying Functions to Dataframe
    df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
    df.drop(['DEPARTMENT'],axis = 1, inplace = True)
    df.rename(columns={'EMPLOYEE OFFICIAL EMAIL ID':'EMPLOYEE MAIL'}, inplace=True) # added remove if needed

    # print("individeal_employee",df.columns)
    # df = pd.read_excel(excel_path)
    final_df, Avg_df, day_df, date_df = apply_all_functions(df,employee)
    
    exclude_emp = ['ASIF AHMED MOHAMMAD','PARDHA SARADHI. KATTA','TATI SASI KRISHNA','KOTI AZMIRA',
                'KHAJA SALEEMUDDIN','M CHENNAKESHAVULU','MOHAMMED SHOUKAT ALI','VENKAYALA V. UDAY KIRAN','SYED IMTIAZ','MOHD AKHEEL','NAZIMA ERSHAD']
    final_df = final_df[~final_df['EMPLOYEE NAME'].isin(exclude_emp)].reset_index(drop = True)
    
    
    employee_table = final_df[['EMPLOYEE NAME', 'DATE', 'IN TIME', 'OUT TIME', 'TOTAL',
        'Grand Total Worked', 'Grand Total Actual',
        'Grand Total Final', 'WEEK', 'WEEK_AVG', 'WEEK_IN_TIME_AVG',
        'WEEK_OUT_TIME_AVG', 'Day', 'WEEK_TOTAL', 'Average In Time',
        'Average Out Time', 'Avg Per day Work', 'AVG Day IN TIME',
        'AVG Day OUT TIME']]
    employee_table["Grand Total Actual"] = len(employee_table["DATE"].unique())*9.5
    employee_table["Grand Total Final"] = employee_table["Grand Total Actual"] - employee_table["Grand Total Worked"]
    employee_count = len(final_df["EMPLOYEE NAME"].unique())

    time_object_in = employee_table["Average In Time"].drop_duplicates().values.tolist()[0]
    in_time_obj = str(time_object_in)[0:5]
    time_object_out = employee_table["Average Out Time"].drop_duplicates().values.tolist()[0]
    out_time_obj = str(time_object_out)[0:5]
    hour_part_in = time_object_in.hour
    minute_part_in = time_object_in.minute
    hour_part_out = time_object_out.hour
    minute_part_out = time_object_out.minute
    
    if len(str(minute_part_in))<2:
        minute_part_str = '0' + str(minute_part_in)
    else: 
        minute_part_str = minute_part_in
        
    compensation_time = employee_table["Grand Total Final"].drop_duplicates().values.tolist()[0]
    # compensation_time = hrs_to_min(compensation_time)
    compensation_time = round(compensation_time, 2)
    compensation_time_text = hrs_to_min(abs(compensation_time))
    # hours_time, minutes_time = map(int, str(compensation_time_text).split('.'))
    hours_time = str(compensation_time_text).split('.')[0]
    try:
        minutes_time = str(compensation_time_text).split('.')[1]
    except:
        minutes_time = 0
    if len(str(minutes_time))==1:
        minutes_time = str(minutes_time) + "0"
    compensation_time_text = f"{hours_time} hrs and {minutes_time} min"
    
    compensation_per_day = round(compensation_time/5,2)
    compensation_per_day = hrs_to_min(abs(compensation_per_day))
    # hours_day, minutes_day = map(int, str(compensation_per_day).split('.'))
    hours_day = str(compensation_per_day).split('.')[0]
    try:
        minutes_day = str(compensation_per_day).split('.')[1]
    except:
        minutes_day = 0
    if len(str(minutes_day))==1:
        minutes_time = str(minutes_day) + "0"
    compensation_per_day = f"{hours_day} hrs and {minutes_day} min"
    
    if len(str(compensation_per_day))<=2:
        compensation_per_day = str(compensation_per_day) + ".00"
    # elif len(str(compensation_per_day))==4 and str(compensation_per_day).startswith('1'):
    #     compensation_per_day = str(compensation_per_day)
    #     compensation_per_day = compensation_per_day + '0'
    elif len(str(compensation_per_day))==3:
        compensation_per_day = str(compensation_per_day)
        compensation_per_day = compensation_per_day + '0'
    else:
        compensation_per_day = str(compensation_per_day)


    # best_login_time = sorted(employee_table["IN TIME"].values.tolist())[0]
    
    best_login_table = employee_table[employee_table["IN TIME"] == employee_table["IN TIME"].min()]
    best_login_time = best_login_table['IN TIME'].iloc[0]
    best_login_date = best_login_table['DATE'].iloc[0].date()
    
    # print(best_login_time)
    
    best_logout_table = employee_table[employee_table["OUT TIME"] == employee_table["OUT TIME"].max()]
    best_logout_time = best_logout_table['OUT TIME'].iloc[0]
    best_logout_date = best_logout_table['DATE'].iloc[0].date()
        
    day_count = len(final_df['DATE'].unique())
    in_count = len(final_df["IN TIME"])

    if hour_part_in < 9:
        in_df = employee_table[["Average In Time"]].drop_duplicates().reset_index(drop = True)
        in_df["Actual In Time"] = "09:00"
        in_df["Actual In Time"] = pd.to_datetime(in_df["Actual In Time"], format='%H:%M').dt.time
        in_df["Actual In Time_temp"] = in_df["Actual In Time"].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))
        in_df["Average In Time_temp"] = in_df["Average In Time"].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))
        # Calculate time difference
        in_df["In Time Difference"] = in_df["Actual In Time_temp"] - in_df["Average In Time_temp"]
        # Convert timedelta64 to hours and minutes
        in_df["In Time Difference"] = in_df["In Time Difference"].apply(lambda x: f"{int(x.seconds // 3600)} hr {int((x.seconds % 3600) // 60)} min")
        in_df = in_df.drop(['Actual In Time_temp','Average In Time_temp'],axis = 1)
    else:
        in_df = employee_table[["Average In Time"]].drop_duplicates().reset_index(drop = True)
        in_df["Actual In Time"] = "09:00"
        in_df["Actual In Time"] = pd.to_datetime(in_df["Actual In Time"], format='%H:%M').dt.time
        in_df["Actual In Time_temp"] = in_df["Actual In Time"].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))
        in_df["Average In Time_temp"] = in_df["Average In Time"].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))
        # Calculate time difference
        in_df["In Time Difference"] = in_df["Average In Time_temp"] - in_df["Actual In Time_temp"]
        # Convert timedelta64 to hours and minutes
        in_df["In Time Difference"] = in_df["In Time Difference"].apply(lambda x: f"{int(x.seconds // 3600)} hr {int((x.seconds % 3600) // 60)} min")
        in_df = in_df.drop(['Actual In Time_temp','Average In Time_temp'],axis = 1)
        
    out_df = Avg_df[["Average Out Time","Actual Out Time","Out Time Difference"]]

    work_hrs = Avg_df[["Avg Per day Work"]]
    work_hrs["Actual Per day Work"] = 9.5
    work_hrs["Time Difference"] = work_hrs["Actual Per day Work"] - work_hrs["Avg Per day Work"]
    work_hrs = work_hrs.round(2)
    # work_hrs = work_hrs.applymap(hrs_to_min)

    Total_work = pd.DataFrame({"Total Work Hrs":final_df["TOTAL"].sum(),
                # "Total Actual Work Hrs":in_count*9.5
                "Total Actual Work Hrs": employee_table["Grand Total Actual"][0]
                },index = [0])
    Total_work["Total Pending Hrs"] = Total_work["Total Actual Work Hrs"] - Total_work["Total Work Hrs"]
    Total_work = Total_work.round(2)
    # Total_work = Total_work.applymap(hrs_to_min)

    day_df["AVG Day IN TIME"] = day_df["AVG Day IN TIME"].astype('str')
    day_df["AVG Day OUT TIME"] = day_df["AVG Day OUT TIME"].astype('str')
    day_df['DAYTIME_IN'] = pd.to_datetime("2024-01-15" + ' ' + day_df['AVG Day IN TIME'])
    day_df['DAYTIME_OUT'] = pd.to_datetime("2024-01-15"  + ' ' + day_df['AVG Day OUT TIME'])

    # Extract the time component and convert it to string
    day_df['AVG Day IN TIME'] = day_df['DAYTIME_IN'].dt.time.astype(str)
    day_df['AVG Day OUT TIME'] = day_df['DAYTIME_OUT'].dt.time.astype(str)
    day_df.drop(["AVG Day IN TIME","AVG Day OUT TIME"],axis=1,inplace=True)

    day_df['DAYTIME_IN'] = pd.to_datetime(day_df['DAYTIME_IN'])

    # Extract Date and Time columns
    day_df['DATE'] = day_df['DAYTIME_IN'].dt.date
    day_df['TIME_IN'] = day_df['DAYTIME_IN'].dt.time

    # Sort DataFrame by time
    day_df = day_df.sort_values(by='TIME_IN')

    # Convert time values to datetime format
    day_df['IN_TIME_DATETIME'] = pd.to_datetime(day_df['TIME_IN'].astype(str))

    # Group by Date and aggregate Time (using the first time value of each date)
    grouped_data_IN = day_df.groupby('Day')['IN_TIME_DATETIME'].first().reset_index()
    grouped_data_IN['Day'] = grouped_data_IN['Day'].astype('category')
    custom_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Convert 'Day' to categorical with custom order
    grouped_data_IN['Day'] = pd.Categorical(grouped_data_IN['Day'], categories=custom_order, ordered=True)

    # Sort the DataFrame by 'Day'
    grouped_data_IN = grouped_data_IN.sort_values(by='Day')


    day_df['DAYTIME_OUT'] = pd.to_datetime(day_df['DAYTIME_OUT'])

    # Extract Date and Time columns
    day_df['DATE'] = day_df['DAYTIME_OUT'].dt.date
    day_df['TIME_OUT'] = day_df['DAYTIME_OUT'].dt.time

    # Sort DataFrame by time
    day_df = day_df.sort_values(by='TIME_OUT')

    # Convert time values to datetime format
    day_df['OUT_TIME_DATETIME'] = pd.to_datetime(day_df['TIME_OUT'].astype(str))

    # Group by Date and aggregate Time (using the first time value of each date)
    grouped_data_OUT = day_df.groupby('Day')['OUT_TIME_DATETIME'].first().reset_index()
    grouped_data_OUT['Day'] = grouped_data_OUT['Day'].astype('category')
    custom_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Convert 'Day' to categorical with custom order
    grouped_data_OUT['Day'] = pd.Categorical(grouped_data_OUT['Day'], categories=custom_order, ordered=True)

    # Sort the DataFrame by 'Day'
    grouped_data_OUT = grouped_data_OUT.sort_values(by='Day')

    date_df["DATE"] = date_df["DATE"].astype('str')
    date_df["AVG Date IN TIME"] = date_df["AVG Date IN TIME"].astype('str')
    date_df["AVG Date OUT TIME"] = date_df["AVG Date OUT TIME"].astype('str')
    date_df['DATETIME_IN'] = pd.to_datetime(date_df['DATE'] + ' ' + date_df['AVG Date IN TIME'])
    date_df['DATETIME_OUT'] = pd.to_datetime(date_df['DATE'] + ' ' + date_df['AVG Date OUT TIME'])

    # Extract the time component and convert it to string
    date_df['AVG Date IN TIME'] = date_df['DATETIME_IN'].dt.time.astype(str)
    date_df['AVG Date OUT TIME'] = date_df['DATETIME_OUT'].dt.time.astype(str)
    date_df.drop(["DATE","AVG Date IN TIME","AVG Date OUT TIME"],axis=1,inplace=True)

    bin_edges_in = [4,8.30,9, 9.30, 10, 10.30, 19]
    bin_edges_out = [9, 18.30, 19, 19.30, 20,20.30,23.50]
    bin_names_in = ['before 8:30','8:30 - 9:00','9:00 - 9:30', '9:30 - 10:00', '10:00 - 10:30', '10:30 and above']
    bin_names_out = ['before 18:30', '18:30 - 19:00', '19:00 - 19:30', '19:30 - 20:00', '20:00 - 20:30', '20:30 and above']
    final_df['bin_in'] = pd.cut(final_df['IN TIME'], bins=bin_edges_in, labels=bin_names_in)
    final_df['bin_out'] = pd.cut(final_df['OUT TIME'], bins=bin_edges_out, labels=bin_names_out)

    intime_df = final_df.groupby('bin_in').size().reset_index(name='count')
    outtime_df = final_df.groupby('bin_out').size().reset_index(name='count')
    
    current_date = datetime.today().date()
    year = current_date.year
    month = current_date.month
    day = current_date.day
    remaining_working_days = count_working_days(year, month, day)


    if compensation_time>1:
        colors = {
        'background': 'white', #'#edc2c2',#'#f0f8ff',  # Light blue background ff7a7a ed9595 ff7d7d e85d5d
        'dark background':'#ff9c9c', #'#eba9a9',
        'text': '#cc0000',  # Black text color
        'text_card' : '#cc0000',
        'plotly_text': '#000000',  # Plotly default blue color 
        'table_color': '#f58484',
        'card_color': '#ff9c9c', #'#eba9a9',
        # 'banner_color':'#ff9c9c',
        'banner_color':'white',
        'plot_color': 'red'
        }
        # colors = {
        # 'background': '#c2d9ed',#'#f0f8ff',  # Light blue background
        # 'dark background':'#a9cceb',
        # 'text': '#cc0000',  # Black text color
        # 'plotly_text': '#000000',  # Plotly default blue color 
        # 'table_color': 'lightblue',
        # 'card_color': '#a9cceb'
        # }
    else:
        # colors = {
        # 'background': '#c2edc9',#'#f0f8ff',  # Light blue background
        # 'dark background':'#a9ebad',
        # 'text': 'green',  # Black text color
        # 'plotly_text': '#000000',
        # 'table_color': '#84f597',# Plotly default blue color 
        # 'card_color': '#a9ebad'
        #  }
        colors = {
        'background': 'white',#'#f0f8ff',  # Light blue background
        'dark background':'#a9cceb',
        'text': 'green',  # Black text color
        'text_card' : 'black',
        'plotly_text': '#000000',
        'table_color': 'lightblue',# Plotly default blue color 
        'card_color': '#a9cceb',
        'banner_color':'white',
        'plot_color': '#3498db'
        }
    if compensation_time<0:
        bar_colors = ['#3498db', '#2ecc71', '#f53333']
    else:
        bar_colors = ['#3498db', '#2ecc71', '#f53333']   #007d02
        
    if hour_part_in < 12:
        time_am_pm_in =  "AM"
    else:
        time_am_pm_in =  "PM"
    
    if hour_part_out >=12:
        time_am_pm_out = "PM"
    else:
        time_am_pm_out = "AM"
        

    # In time Statement
    if hour_part_in < 9:
        intime_statement = f"* Your average login time is **{in_time_obj}**, it's fantastic to observe you logging in before office timings. Your punctuality sets a positive tone for the day. Keep up the excellent work!"
    elif hour_part_in == 9 and minute_part_in < 15:
        intime_statement = f"* Your average login time is **{in_time_obj}**, we recommend you to follow office login time, punctuality is crucial for a smooth work environment and  Your cooperation in this regard is greatly appreciated."
    elif hour_part_in == 9 and 15 <= minute_part_in < 60:
        intime_statement = f"* Your average login time is **{in_time_obj}**, strictly adhere to office login time. Login must be completed before **09:00**. Punctuality is crucial for a smooth work environment. Your cooperation is appreciated."
    else:
        intime_statement = f"* Your average login time is **{in_time_obj}**, effective immediately, you are required to login before **09:00**. Your average login time, consistently beyond **10:00**, needs urgent correction. This is a non-negotiable directive for better adherence to office timings."

    # Out time statement
    # if hour_part_out < 18 or (hour_part_out == 18 and minute_part_out < 30):
    #     outtime_statement = "Your average Logout time is before **18:30**"
    # elif hour_part_out == 18 and 30 <= minute_part_out < 60:
    #     outtime_statement = "Your average Logout time is between **18:30 and 19:00**"
    # elif hour_part_out == 19 and minute_part_out < 30:
    #     outtime_statement = "Your average Logout time is between **19:00 and 19:30**"
    # else:
    #     outtime_statement = "Your average Logout time is after **19:30**"
    
    outtime_statement = f"* Your average logout time is **{out_time_obj}** for the last **{day_count}** Working Days."

    # Best Login time statement
    if best_login_time<=9:
        if len(str(best_login_time))<=2:
            best_login_time = str(best_login_time) + ":00"
        elif len(str(best_login_time))==4 and str(best_login_time).startswith('1'):
            best_login_time = str(best_login_time).replace('.',':')
            best_login_time = best_login_time + '0'
        elif len(str(best_login_time))==3:
            best_login_time = str(best_login_time).replace('.',':')
            best_login_time = best_login_time + '0'
        else:
            best_login_time = str(best_login_time).replace('.',':')
            
        hrs_bst, min_bst = best_login_time.split(':')[0],best_login_time.split(':')[1]
        if len(hrs_bst)==1:
            hrs_bst = "0" +  hrs_bst
            best_login_time = hrs_bst + ":" + min_bst
        else:
            best_login_time = hrs_bst + ":" + min_bst
        
        best_intime_statement = f"* Your best login time, **{best_login_time}** on **{best_login_date}**, is absolutely impressive. Keep up the outstanding effort you're doing great!"
    elif best_login_time>9 and best_login_time<9.15:
        
        if len(str(best_login_time))<=2:
            best_login_time = str(best_login_time) + ":00"
        elif len(str(best_login_time))==4 and str(best_login_time).startswith('1'):
            best_login_time = str(best_login_time).replace('.',':')
            best_login_time = best_login_time + '0'
        elif len(str(best_login_time))==3:
            best_login_time = str(best_login_time).replace('.',':')
            best_login_time = best_login_time + '0'
        else:
            best_login_time = str(best_login_time).replace('.',':')
            
        hrs_bst, min_bst = best_login_time.split(':')[0],best_login_time.split(':')[1]
        if len(hrs_bst)==1:
            hrs_bst = "0" +  hrs_bst
            best_login_time = hrs_bst + ":" + min_bst
        else:
            best_login_time = hrs_bst + ":" + min_bst
            
        best_intime_statement = f"* Your best login time was **{best_login_time}** on **{best_login_date}**, which is falling short of suggested Login."
    elif best_login_time>9 and best_login_time<9.30:
        
        if len(str(best_login_time))<=2:
            best_login_time = str(best_login_time) + ":00"
        elif len(str(best_login_time))==4 and str(best_login_time).startswith('1'):
            best_login_time = str(best_login_time).replace('.',':')
            best_login_time = best_login_time + '0'
        elif len(str(best_login_time))==3:
            best_login_time = str(best_login_time).replace('.',':')
            best_login_time = best_login_time + '0'
        else:
            best_login_time = str(best_login_time).replace('.',':')
        hrs_bst, min_bst = best_login_time.split(':')[0],best_login_time.split(':')[1]
        if len(hrs_bst)==1:
            hrs_bst = "0" +  hrs_bst
            best_login_time = hrs_bst + ":" + min_bst
        else:
            best_login_time = hrs_bst + ":" + min_bst
            
        best_intime_statement = f"* Your Best login Time was **{best_login_time}** on **{best_login_date}**, which is falling short of suggested login."
    elif best_login_time>9.30 and best_login_time<10:
        if len(str(best_login_time))<=2:
            best_login_time = str(best_login_time) + ":00"
        elif len(str(best_login_time))==4 and str(best_login_time).startswith('1'):
            best_login_time = str(best_login_time).replace('.',':')
            best_login_time = best_login_time + '0'
        elif len(str(best_login_time))==3:
            best_login_time = str(best_login_time).replace('.',':')
            best_login_time = best_login_time + '0' 
        else:
            best_login_time = str(best_login_time).replace('.',':')
        hrs_bst, min_bst = best_login_time.split(':')[0],best_login_time.split(':')[1]
        if len(hrs_bst)==1:
            hrs_bst = "0" +  hrs_bst
            best_login_time = hrs_bst + ":" + min_bst
        else:
            best_login_time = hrs_bst + ":" + min_bst
        best_intime_statement = f"* Your Best login Time was **{best_login_time}** on **{best_login_date}**, strictly adhere to office login time, no exceptions."
    else:
        if len(str(best_login_time))<=2:
            best_login_time = str(best_login_time) + ":00"
        elif len(str(best_login_time))==4 and str(best_login_time).startswith('1'):
            best_login_time = str(best_login_time).replace('.',':')
            best_login_time = best_login_time + '0'
        elif len(str(best_login_time))==3:
            best_login_time = str(best_login_time).replace('.',':')
            best_login_time = best_login_time + '0'
        else:
            best_login_time = str(best_login_time).replace('.',':')
        hrs_bst, min_bst = best_login_time.split(':')[0],best_login_time.split(':')[1]
        if len(hrs_bst)==1:
            hrs_bst = "0" +  hrs_bst
            best_login_time = hrs_bst + ":" + min_bst
        else:
            best_login_time = hrs_bst + ":" + min_bst
            
        best_intime_statement = f"* Under strict orders, adhere to office timings. Your best login time was **{best_login_time}** on **{best_login_date}**, which does not fall within the suggested login. Punctuality is non-negotiable."
        
    if len(str(best_logout_time))<=2:
        best_logout_time = str(best_logout_time) + ":00"
    elif len(str(best_logout_time))==4 and str(best_logout_time).startswith('1'):
        best_logout_time = str(best_logout_time).replace('.',':')
        best_logout_time = best_logout_time + '0'
    elif len(str(best_logout_time))==3:
        best_logout_time = str(best_logout_time).replace('.',':')
        best_logout_time = best_logout_time + '0'
    else:
        best_logout_time = str(best_logout_time).replace('.',':')
    hrs_bst, min_bst = best_logout_time.split(':')[0],best_logout_time.split(':')[1]
    if len(hrs_bst)==1:
        hrs_bst = "0" +  hrs_bst
        best_logout_time = hrs_bst + ":" + min_bst
    else:
        best_logout_time = hrs_bst + ":" + min_bst
        
    best_outtime_statement = f"* Your belated logout time was **{best_logout_time}** on **{best_logout_date}**."
    # Compensation statement
    if compensation_time < 0:
        compensation_statement = f"* Your dedication is commendable! You've put in **{compensation_time_text}** of overtime. Keep up the excellent work!"
    elif compensation_time<=2:
        compensation_statement = f"* Your compensation hours stand at **{compensation_time_text}** , a manageable amount that can be easily compensated. Let's work together to address this effectively."
    elif compensation_time>2 and compensation_time<5:
        compensation_statement = f"* Your compensation hours are **{compensation_time_text}**, and it's crucial to improve. Consider putting in extra effort to compensate for this shortfall."
    else:
        compensation_statement = f"* Your compensation hours are **{compensation_time_text}**, a level that does not align with our company's policies. Immediate action is required to rectify this situation. Adjust your schedule accordingly and adhere to company standards."

    # Compensation Planning
    if compensation_time < 0:
        planning = f"* Your willingness to put in **{str(compensation_time_text)}** extra effort is truly commendable and greatly benefits the company. Your dedication is an inspiration to us all. Keep up the fantastic work!"
    elif 0 < compensation_time < 2.1:
        planning = f"* To fulfill your compensation hours target of **{str(compensation_time_text)}**, consider dedicating an additional **{str(compensation_time_text)}** today or the following day. Your commitment and effort are truly valued."

    else:
        current_date = datetime.today().date()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        remaining_working_days = count_working_days(year, month, day)
        compensation_time = round(compensation_time, 2)
        compensation_time_text = hrs_to_min(abs(compensation_time))
        # hours_time, minutes_time = map(int, str(compensation_time_text).split('.'))
        hours_time = str(compensation_time_text).split('.')[0]
        try:
            minutes_time = str(compensation_time_text).split('.')[1]
        except:
            minutes_time = 0
        if len(str(minutes_time))==1:
            minutes_time = str(minutes_time) + "0"
        compensation_time_text = f"{hours_time} hrs and {minutes_time} min"


        remaining_weeks = weeks_remaining_in_month()
        # remaining_weeks = 1
        

        working_days_last = working_days_in_last_week()

        if remaining_working_days < 5:
            working_days = remaining_working_days
        else:
            working_days = 5
        # start_time = datetime.strptime(f"{hour_part_in}:{minute_part_in}", "%H:%M")
        work_duration = timedelta(hours=compensation_time/working_days+9.5)

        compensation_per_day_num = round(compensation_time/working_days,2)
        if compensation_per_day_num >3:
            compensation_per_day_num = 3
        else:
            compensation_per_day_num = compensation_per_day_num
            
        compensation_per_day = hrs_to_min(abs(compensation_per_day_num))
        hours_day = str(compensation_per_day).split('.')[0]
        try:
            minutes_day = str(compensation_per_day).split('.')[1]
        except:
            minutes_day = 0
        if len(str(minutes_day))==1:
            minutes_time = str(minutes_day) + "0"
        compensation_per_day = f"{hours_day} hrs and {minutes_day} min"

        if len(str(compensation_per_day))<=2:
            compensation_per_day = str(compensation_per_day) + ".00"
        # elif len(str(compensation_per_day))==4 and str(compensation_per_day).startswith('1'):
        #     compensation_per_day = str(compensation_per_day)
        #     compensation_per_day = compensation_per_day + '0'
        elif len(str(compensation_per_day))==3:
            compensation_per_day = str(compensation_per_day)
            compensation_per_day = compensation_per_day + '0'
        else:
            compensation_per_day = str(compensation_per_day)
            

        ####

        compensation_per_lst_num = round(compensation_time/working_days_last,2)
        if compensation_per_lst_num >=3:
            compensation_per_lst_num = 3
        else:
            compensation_per_lst_num = compensation_per_lst_num
            
        compensation_per_day_lst = hrs_to_min(abs(compensation_per_lst_num))
        hours_day_lst = str(compensation_per_day_lst).split('.')[0]
        try:
            minutes_day_lst = str(compensation_per_day_lst).split('.')[1]
        except:
            minutes_day_lst = 0
        if len(str(minutes_day_lst))==1:
            minutes_day_lst = str(minutes_day_lst) + "0"
        compensation_per_day_lst = f"{hours_day_lst} hrs and {minutes_day_lst} min"

        if len(str(compensation_per_day_lst))<=2:
            compensation_per_day_lst = str(compensation_per_day_lst) + ".00"
        # elif len(str(compensation_per_day))==4 and str(compensation_per_day).startswith('1'):
        #     compensation_per_day = str(compensation_per_day)
        #     compensation_per_day = compensation_per_day + '0'
        elif len(str(compensation_per_day_lst))==3:
            compensation_per_day_lst = str(compensation_per_day_lst)
            compensation_per_day_lst = compensation_per_day_lst + '0'
        else:
            compensation_per_day_lst = str(compensation_per_day_lst)

        #####


        if compensation_per_day_num < 1.5:
            ear_in_time = "09:00"
        else:
            ear_in_time = "08:00"
            
        reg_time = datetime.strptime(ear_in_time, "%H:%M")

        if compensation_per_day_num < 3:
            reg_end_time = reg_time + work_duration
        else:
            reg_end_time = datetime.strptime("20:30", "%H:%M")

        remaining_comp_time = compensation_time - (compensation_per_day_num*working_days)
        remaining_comp_per_lst_wk = round(remaining_comp_time/working_days_last,2)

        remaining_comp_time_lst_wk = compensation_time - (compensation_per_day_num*working_days_last)
        remaining_comp_time_lst_wk = round(remaining_comp_time_lst_wk,2)

        remaining_comp_time_fin_wk = compensation_time - (compensation_per_lst_num*working_days_last)
        remaining_comp_time_fin_wk = round(remaining_comp_time_fin_wk,2)

        # final_comp_time = remaining_comp_time - (remaining_comp_per_lst_wk*working_days_last)
        final_comp_time = remaining_comp_time - (3*working_days_last)


        final_comp_time = round(final_comp_time,2)
        remaining_comp_time = round(remaining_comp_time, 2)
        remaining_comp_time_text = hrs_to_min(abs(remaining_comp_time))
        # hours_time, minutes_time = map(int, str(compensation_time_text).split('.'))
        hours_time_cmp = str(remaining_comp_time_text).split('.')[0]
        try:
            minutes_time_cmp = str(remaining_comp_time_text).split('.')[1]
        except:
            minutes_time_cmp = 0
        if len(str(minutes_time_cmp))==1:
            minutes_time_cmp = str(minutes_time_cmp) + "0"
        remaining_comp_time_text = f"{hours_time_cmp} hrs and {minutes_time_cmp} min"

        remaining_comp_per_lst_wk_text = hrs_to_min(abs(remaining_comp_per_lst_wk))
        # hours_time, minutes_time = map(int, str(compensation_time_text).split('.'))
        hours_time_cmp_lst = str(remaining_comp_per_lst_wk_text).split('.')[0]
        try:
            minutes_time_cmp_lst = str(remaining_comp_per_lst_wk_text).split('.')[1]
        except:
            minutes_time_cmp_lst = 0
        if len(str(minutes_time_cmp_lst))==1:
            minutes_time_cmp_lst = str(minutes_time_cmp_lst) + "0"
        remaining_comp_per_lst_wk_text = f"{hours_time_cmp_lst} hrs and {minutes_time_cmp_lst} min"

        final_comp_per_lst_wk_text = hrs_to_min(abs(final_comp_time))
        # hours_time, minutes_time = map(int, str(compensation_time_text).split('.'))
        hours_time_cmp_fin = str(final_comp_per_lst_wk_text).split('.')[0]
        try:
            minutes_time_cmp_fin = str(final_comp_per_lst_wk_text).split('.')[1]
        except:
            minutes_time_cmp_fin = 0
        if len(str(minutes_time_cmp_fin))==1:
            minutes_time_cmp_fin = str(minutes_time_cmp_fin) + "0"
        final_comp_per_lst_wk_text = f"{hours_time_cmp_fin} hrs and {minutes_time_cmp_fin} min"

        comp_time_lst_wk = round(compensation_time/working_days_last,2)

        if comp_time_lst_wk>=3:
            comp_time_lst_wk = 3
        else:
            comp_time_lst_wk = comp_time_lst_wk

        rem_comp_final = compensation_time - (comp_time_lst_wk * working_days_last)
        rem_comp_final = round(rem_comp_final,2)



        final_lst_wk_text = hrs_to_min(abs(comp_time_lst_wk))
        # hours_time, minutes_time = map(int, str(compensation_time_text).split('.'))
        hours_fin = str(final_lst_wk_text).split('.')[0]
        try:
            minutes_fin = str(final_lst_wk_text).split('.')[1]
        except:
            minutes_fin = 0
        if len(str(minutes_fin))==1:
            minutes_fin = str(minutes_fin) + "0"
        final_lst_wk_text = f"{hours_fin} hrs and {minutes_fin} min"

        rem_lst_wk_text = hrs_to_min(abs(rem_comp_final))
        # hours_time, minutes_time = map(int, str(compensation_time_text).split('.'))
        hours_fin_rem = str(rem_lst_wk_text).split('.')[0]
        try:
            minutes_fin_rem = str(rem_lst_wk_text).split('.')[1]
        except:
            minutes_fin_rem = 0
        if len(str(minutes_fin_rem))==1:
            minutes_fin_rem = str(minutes_fin_rem) + "0"
        rem_lst_wk_text = f"{hours_fin_rem} hrs and {minutes_fin_rem} min"

        remaining_comp_time_fin_wk_txt = hrs_to_min(abs(remaining_comp_time_fin_wk))
        # hours_time, minutes_time = map(int, str(compensation_time_text).split('.'))
        hours_fin_rem_fi = str(remaining_comp_time_fin_wk_txt).split('.')[0]
        try:
            minutes_fin_rem_fi = str(remaining_comp_time_fin_wk_txt).split('.')[1]
        except:
            minutes_fin_rem_fi = 0
        if len(str(minutes_fin_rem_fi))==1:
            minutes_fin_rem_fi = str(minutes_fin_rem_fi) + "0"
        remaining_comp_time_fin_wk_txt = f"{hours_fin_rem_fi} hrs and {minutes_fin_rem_fi} min"

        formatted_reg_time = reg_end_time.strftime("%H:%M")

        if remaining_comp_time > 0 and remaining_weeks > 1:
            print(1)
            planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{compensation_per_day}** per day for the next **{working_days}** working days. And the remaining **{remaining_comp_time_text}** in following weeks.
            \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
            \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
        elif remaining_comp_time > 0 and remaining_weeks == 1:
            if remaining_comp_per_lst_wk < 3:
                if compensation_per_day_num>3:
                    print(2.1)
                    planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{compensation_per_day}** per day for the next **{working_days}** working days. And the remaining **{remaining_comp_time_text}** in following week as **{remaining_comp_per_lst_wk_text}** per day for **{working_days_last}** days in last week.
                    \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
                    \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
                else:
                    print(2.1)
                    planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{compensation_per_day}** per day for the next **{working_days}** working days.
                    \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
                    \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
            else:
                if final_comp_time >1:
                    print(3)
                    planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{compensation_per_day}** per day for the next **{working_days}** working days. And the remaining **{remaining_comp_time_text}** in following week as **3 hrs and 0 min** per day for **{working_days_last}** working days in last week and for the balance **{final_comp_per_lst_wk_text}** Salary will be deducted or any other instructions given by the management will be taken.
                    \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
                    \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
                else:
                    print(3.1)
                    planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{compensation_per_day}** per day for the next **{working_days}** working days. And the remaining **{remaining_comp_time_text}** in following week as **3 hrs and 0 min** per day for **{working_days_last}** working days in last week and for the balance **{final_comp_per_lst_wk_text}** further information will be given by management.
                    \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
                    \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
        elif remaining_comp_time > 0 and remaining_weeks == 0:
            if rem_comp_final>1:
                print(4)
                planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{final_lst_wk_text}** per day for the next **{working_days_last}** working days. And for the remaining **{rem_lst_wk_text}** Salary will be deducted or any other instructions given by the management will be taken.
                    \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
                    \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
            elif rem_comp_final<0:
                print(4.1)
                planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{final_lst_wk_text}** per day for the next **{working_days_last}** working days.
                    \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
                    \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
            else:
                print(4.2)
                planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{final_lst_wk_text}** per day for the next **{working_days_last}** working days. And for the remaining **{rem_lst_wk_text}** further information will be given by management.
                    \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
                    \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
        else:
            if remaining_weeks >= 1:
                print(5)
                planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{compensation_per_day}** per day for the next **{working_days}** working days.
                        \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
                        \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
            else:
                if remaining_comp_time_fin_wk<=0:
                    print(6)
                    planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{compensation_per_day_lst}** per day for the next **{working_days_last}** working days.
                            \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
                            \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
                else:
                    if remaining_comp_time_fin_wk>1:
                        print(7)
                        planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{compensation_per_day_lst}** per day for the next **{working_days_last}** working days. And for the remaining **{remaining_comp_time_fin_wk_txt}** Salary will be deducted or any other instructions given by the management will be taken.
                                \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
                                \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
                    else:
                        print(7.1)
                        planning = f"""* To fulfill your compensation hours target of **{compensation_time_text}**, consider dedicating an additional **{compensation_per_day_lst}** per day for the next **{working_days_last}** working days. And for the remaining **{remaining_comp_time_fin_wk_txt}** further information will be given by management.
                                \n* Considering your target, we highly recommend logging in at **{ear_in_time}** and logout after **{formatted_reg_time}** to complete your compensation hours efficiently. 
                                \n* Your commitment and effort are greatly appreciated as we work towards meeting our goals together."""
            
            
    emp_name = employee
    # Component 1 
    Header_component = html.H1(emp_name, style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '20px',"font-family": "Optima", "font-size": "44px","font-weight": "bold"})

    # Graph 1
    fig1 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in in_df.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color=colors['table_color'],  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[in_df[col] for col in in_df.columns],height = 35))
    ])
    fig1.update_layout(
        title='In-Time',
        height=120,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor=colors['dark background'],
        font_color=colors['plotly_text'],
        font=dict(
            family='Optima',
            size=16  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.9
    )

    # Graph 2
    fig2 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in out_df.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color=colors['table_color'],  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[out_df[col] for col in out_df.columns],height = 35))
    ])

    fig2.update_layout(
        title='Out-Time',
        height=120,
        margin=dict(l=0, r=0, b=10, t=40),
        paper_bgcolor=colors['dark background'],
        font_color=colors['plotly_text'],
        font=dict(
            family='Optima',
            size=16  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.9
    )

    # Graph 4
    # text_values_2 = work_hrs.iloc[0, :].round(2).astype(str)
    text_values_2 = work_hrs.values.tolist()[0]
    text_values_2 = plot_values(text_values_2)
    text_values_2 = ['+' + str(abs(float(value))) if float(value) < 0 else value for value in text_values_2]
    fig4 = px.bar(x=work_hrs.iloc[0, :], y=['Average Worked Hrs','Actual Work Hrs','Pending Hrs'],
                labels={'x': 'Time in Hours', 'y': ''}, height=250, width=728,
                text=text_values_2,color=work_hrs.columns, color_discrete_sequence=bar_colors)
    fig4.update_layout(bargap=0.3, bargroupgap=0.5, title='Per Day Average Work', margin=dict(l=0, r=10, b=0, t=30),
                    showlegend=False,
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['plotly_text'],
                    font=dict(
                        family='Optima',
                        size=16  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)
    fig4.update_traces(textposition='auto', hovertemplate='<b>Time in Hours</b>: %{x}')

    # Graph 5
    # text_values_3 = Total_work.iloc[0, :].round(2).astype(str)
    text_values_3 = Total_work.values.tolist()[0]
    text_values_3 = plot_values(text_values_3)
    text_values_3 = ['+' + str(abs(float(value))) if float(value) < 0 else value for value in text_values_3]
    fig5 = px.bar(x=Total_work.iloc[0, :], y=['Total Worked Hrs','Actual Work Hrs','Pending Hrs'],
                labels={'x': 'Time in Hours', 'y': ''}, height=250, width=728,
                text=text_values_3, color=Total_work.columns, color_discrete_sequence=bar_colors)

    # Update layout
    fig5.update_layout(bargap=0.3, bargroupgap=0.5, title='Total Work',
                    showlegend=False,  # Set showlegend to False
                    margin=dict(l=0, r=10, b=0, t=30),
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['plotly_text'],
                    font=dict(
                        family='Optima',
                        size=16  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)
    fig5.update_traces(textposition='auto', hovertemplate='<b>Time in Hours</b>: %{x}')

    # Graph 6
    # fig6 = go.Figure(data=go.Scatter(x = grouped_data_IN["Day"], y = grouped_data_IN["IN_TIME_DATETIME"]),layout=go.Layout(height = 300))
    # fig6 = px.line(grouped_data_IN, x="Day", y='IN_TIME_DATETIME', markers=True, line_shape='linear', height=300)
    fig6 = go.Figure(data=go.Scatter(x = grouped_data_IN["Day"],
                                y = grouped_data_IN["IN_TIME_DATETIME"],
                                mode='lines+markers+text',
                                text=grouped_data_IN["IN_TIME_DATETIME"].apply(lambda x: str(x)[11:16]),
                                textposition='bottom center',
                                line=dict(color=colors['plot_color'])),
            
                layout=go.Layout(height=290, margin=dict(b=20, t=50, l=50, r=50)))

    # Customize the layout
    fig6.update_layout(
        title='Day Wise Average In-Time',
        xaxis_title='Day',
        yaxis_title='Time',
        # xaxis=dict(tickangle=-45),
        yaxis=dict(
            tickformat="%H:%M:%S",
        ),
        title_x=0.5, title_y=0.98,
        paper_bgcolor=colors['dark background'],
        font_color=colors['plotly_text'],
        font=dict(
                        family='Optima',
                        size=16  # Adjust the font size as needed
                    ),
    )

    # Graph 7
    # fig7 = px.line(grouped_data_OUT, x="Day", y='OUT_TIME_DATETIME', markers=True, line_shape='linear', height=300)
    # fig7 = go.Figure(data=go.Scatter(x = grouped_data_OUT["Day"], y = grouped_data_OUT["OUT_TIME_DATETIME"]),layout=go.Layout(height = 300))
    fig7 = go.Figure(data=go.Scatter(
        x=grouped_data_OUT["Day"],
        y=grouped_data_OUT["OUT_TIME_DATETIME"],
        mode='lines+markers+text',
        text=grouped_data_OUT["OUT_TIME_DATETIME"].apply(lambda x: str(x)[11:16]),
        textposition='bottom center',
        line=dict(color=colors['plot_color'])
    ), layout=go.Layout(height=290, margin=dict(b=20, t=50, l=50, r=50)))


    # Customize the layout
    fig7.update_layout(
        title='Day Wise Average Out-Time',
        xaxis_title='Day',
        yaxis_title='Time',
        # xaxis=dict(tickangle=-45),
        yaxis=dict(
            tickformat="%H:%M:%S",
        ),
        title_x=0.5, title_y=0.98,
        paper_bgcolor=colors['dark background'],
        font_color=colors['plotly_text'],
        font=dict(
                        family='Optima',
                        size=16  # Adjust the font size as needed
                    ),
    )

    # Graph 10 
    fig10 = px.bar(intime_df, x='bin_in', y='count',
                height=300, width=728,
                text='count',
                color_discrete_sequence=[colors['plot_color']]
                # color_discrete_map={'count':colors['plot_color']}
                )  # Display the count on top of each bar

    fig10.update_layout(bargap=0.3, bargroupgap=0.5, title='In-Time Report ',
                    margin=dict(l=0, r=10, b=0, t=30),
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['plotly_text'],
                    xaxis_title='Time Range',
                    yaxis_title='Count',
                    font=dict(
                        family='Optima',
                        size=16  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)

    fig10.update_traces(textangle=0,hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')

    # Graph 11
    fig11 = px.bar(outtime_df, x='bin_out', y='count',
                height=300, width=728,
                text='count',
                color_discrete_sequence=[colors['plot_color']]
                # color_discrete_map={'count':colors['plot_color']}
                )  # Display the count on top of each bar

    fig11.update_layout(bargap=0.3, bargroupgap=0.5, title='Out-Time Report',
                    margin=dict(l=0, r=10, b=0, t=30),
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['plotly_text'],
                    xaxis_title='Time Range',
                    yaxis_title='Count',
                    font=dict(
                        family='Optima',
                        size=16  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)

    fig11.update_traces(textangle=0,hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')

    # Design the app Layout

    # Define a slightly darker shade of blue
    colors['dark background'] = '#1b4965'
    
    if compensation_time>0:
        comp_pen_text = "Compensation Hrs"
    else:
        comp_pen_text = "Extra Worked Hrs"
        

    # Design the app Layout

    app.layout = html.Div(
        [
            dbc.Row([
                dbc.Col(
                    Header_component,
                    width=8,  # Set the width for the header column
                    className='mb-4 mx-auto text-center'
                ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4("Working Days", className="card-title",style = {"color": "Black", "text-align": "center","margin-bottom": "0px","margin-top": "0","font-family": "Optima", "font-size": "22px"}),
                            html.H2(day_count,style = {"color": "Black", "text-align": "center","margin-top": "0","font-family": "Optima", "font-size": "24px","font-weight": "bold"}),
                        ]
                    ),
                    style={"background-color": colors['card_color'],"height": "100px","margin-top": "0px"},
                    # Applying the sizing class to control the size of the card
                    className='mb-1 sm-1',  # Use 'sm', 'md', 'lg', or 'xl'
                ),
                width=2,  # Set the width for the graph column
                className='mb-4'

            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4(f"{comp_pen_text}", className="card-title",style = {"color": "Black", "text-align": "center","margin-bottom": "0px","margin-top": "0","font-family": "Optima", "font-size": "22px"}),
                            html.H2(compensation_time_text,style = {"color": "Black", "text-align": "center","margin-top": "0","font-family": "Optima", "font-size": "22px","font-weight": "bold"}),
                        ]
                    ),
                    style={"background-color": colors['card_color'],"height": "100px","margin-top": "0px"},
                    # Applying the sizing class to control the size of the card
                    className='mb-1 sm-1',  # Use 'sm', 'md', 'lg', or 'xl'
                ),
                width=2,  # Set the width for the graph column
                className='mb-4'

            )
                    
            ]),
            dbc.Row([
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Overview", className="card-title",style={"font-family": "Optima", "font-size": "26px", "font-weight": "bold","color": colors['text_card']}),
                                    dcc.Markdown(intime_statement),
                                    # html.P(intime_statement),
                                    dcc.Markdown(outtime_statement),
                                    # html.P(outtime_statement),
                                    dcc.Markdown(best_intime_statement),
                                    # html.P(best_intime_statement)
                                    dcc.Markdown(best_outtime_statement),
                                ]
                            ),
                            style={"background-color": colors['banner_color'],"font-family": "Optima", "font-size": "24px","width": "98%","margin": "0 auto","color":colors['text_card']},  #color=colors['dark background'], 
                            className='mb-4',
                        )
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig1, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'  # Add margin at the bottom
                ),
                dbc.Col(
                    dcc.Graph(figure=fig2, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'  # Add margin at the bottom
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig4, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,  # Adjust the width to 6 to make it centered and equal-sized
                    className='mb-4',
                    align='center'  # Align the content in the center
                ),
                dbc.Col(
                    dcc.Graph(figure=fig5, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,  # Adjust the width to 6 to make it centered and equal-sized
                    className='mb-4',
                    align='center'  # Align the content in the center
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig10,style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                    ),
                dbc.Col(
                    dcc.Graph(figure=fig11,style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig6, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                ),
                dbc.Col(
                    dcc.Graph(figure=fig7, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                )
            ]),
            dbc.Row([
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4("Suggestions", className="card-title",style={"font-family": "Optima", "font-size": "26px", "font-weight": "bold","color":colors['text_card']}),
                            # html.P(intime_statement),
                            # html.P(outtime_statement),
                            # html.P(best_intime_statement),
                            # html.P(compensation_statement),
                            dcc.Markdown(compensation_statement),
                            dcc.Markdown(planning),
                            # dbc.Button("Click me", color="primary"),
                        ]
                    ),
                    style={"background-color": colors['banner_color'],"font-family": "Optima", "font-size": "24px","color":colors['text_card']},
                    className='mb-4',
                )
            ]),
        ],
        style={
            # 'backgroundColor': colors['background'],
            'backgroundColor': colors['background'],
            'padding': '20px'  # Add padding to the entire layout
        }
    )
    
    # app.run_server(debug=False, port=8051)
    app.run_server(debug=False, host='127.0.0.1', port=8051)
    
output_folder_path = r"D:\Susmith Tasks\TImesheet_API\Employee_timesheet\Output_files"

def run_selenium(employee,month,output_folder_path):
    print("Opening Chrome...")
    chrome_options = webdriver.ChromeOptions()
    prefs = {
        "profile.default_content_setting_values.notifications": 2,
        "download.default_directory": r"D:\Susmith Tasks\TImesheet_API\Employee_timesheet\Output_files"  # Specify your desired download path
    }
    chrome_options.add_experimental_option("prefs", prefs)

    # chrome_path = r"C:\Users\Rahul\Desktop\Onedrive-sharepoint-Login_Time\chromedriver-win64\chromedriver.exe"
    # chrome_path = r"C:\Users\Rahul\Desktop\Fastapi\Employee_timesheet_api\chromedriver.exe"
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), chrome_options=chrome_options)
    wait = WebDriverWait(driver, 10)

    # url = "http://localhost:8051/"
    url = "http://127.0.0.1:8051/"
    
    driver.get(url)
    driver.maximize_window()
    time.sleep(10)
    print_options = PrintOptions()
    # paper sizes in centimeters
    print_options.page_width = 42
    print_options.page_height = 59
    print_options.background = True
    pdf = driver.print_page(print_options=print_options)
    pdf_bytes = base64.b64decode(pdf)
    file_path = fr"{output_folder_path}\{employee}_{month}.pdf"
    with open(file_path, "wb") as fh:
        fh.write(pdf_bytes)
    # Close the browser
    driver.quit()
    
    return file_path
    
def test_run(content,month):
    # df = pd.read_excel(r"C:\Users\Rahul\Desktop\Onedrive-sharepoint-Login_Time\dash_fastapi\ds_team.xlsx")
    # df = pd.read_excel(excel_path)
    df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
    df.drop(['DEPARTMENT'],axis = 1, inplace = True)
    df.rename(columns={'EMPLOYEE OFFICIAL EMAIL ID':'EMPLOYEE MAIL'}, inplace=True) # added remove if needed
    # print("test_run",df.columns)
    
    
    employee_names = filter_emp_name(df)
    
    exclude_emp = ['ASIF AHMED MOHAMMAD','PARDHA SARADHI. KATTA','TATI SASI KRISHNA','KOTI AZMIRA',
                'KHAJA SALEEMUDDIN','M CHENNAKESHAVULU','MOHAMMED SHOUKAT ALI','VENKAYALA V. UDAY KIRAN','SYED IMTIAZ','MOHD AKHEEL','NAZIMA ERSHAD']
    employee_names = employee_names[~employee_names['EMPLOYEE NAME'].isin(exclude_emp)].reset_index(drop = True)
    
    employee_names = employee_names[["EMPLOYEE NAME","EMPLOYEE MAIL"]].drop_duplicates().dropna().reset_index()
    
    data = {
        'EMP_NAME':[],
        'EMP_MAIL':[],
        'EMP_PATH':[]
        }
    
    for index, row in employee_names.iterrows():
        employee = row["EMPLOYEE NAME"]
        employee_mail = row["EMPLOYEE MAIL"]
        data['EMP_NAME'].append(employee)
        data['EMP_PATH'].append(employee+'_'+month)
        data['EMP_MAIL'].append(employee_mail)
        dash_process = multiprocessing.Process(target=individual_employee, args=(content,employee,))
        dash_process.start()
        
        path_1_process = multiprocessing.Process(target=run_selenium, args=(employee,month,output_folder_path,))
        path_1_process.start()
        path_1_process.join()

        time.sleep(5)
        dash_process.terminate()
        path_1_process.terminate()
        
    emp_mail_df = pd.DataFrame(data)
    emp_mail_df.to_excel(fr"{output_folder_path}\employee_output_path.xlsx")
    return emp_mail_df

def mail_sending(content,emp_mail_df,month):
    sent_list = []
    fail_list = []
    pdf_list = os.listdir(fr"{output_folder_path}")
    for index, row in tqdm(emp_mail_df.iterrows()):
        employee = row["EMP_NAME"]
        employee_mail = row["EMP_MAIL"]
        employee_path = row["EMP_PATH"]
        try:
            # df = pd.read_excel(excel_path)
            df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
            df.drop(['DEPARTMENT'],axis = 1, inplace = True)
            df.rename(columns={'EMPLOYEE OFFICIAL EMAIL ID':'EMPLOYEE MAIL'}, inplace=True) # added remove if needed
            
            # print("mail_sending",df.columns)
            
            emp_data_df, emp_total_df= emp_table_data(df)
            # email_sender = "technocrat3128@gmail.com"
            # email_password = "nhbn dtvr rsbe dtgu"
            email_sender = "corporate.affairs@piloggroup.org"
            email_password = "ymcb udvb nury ukcl"
            recipient_mail = employee_mail
            subject = f"Time Sheet for the month of {month}."
            table1 = emp_data_df[emp_data_df["EMPLOYEE NAME"]==employee].reset_index(drop=True)
            table2 = emp_total_df[emp_total_df["EMPLOYEE NAME"]==employee].drop_duplicates().drop(["EMPLOYEE NAME"],axis = 1)
            
            table1['TOTAL']  = table1['TOTAL'].apply(lambda x : decimal_correct(x)) # added if error remove
            
            table2['ACTUAL WORK HOURS'] = len(table1["DATE"].unique())*9.5 # added if error remove
            # print(table2['ACTUAL WORK HOURS'][0])
            table2['COMPENSATION HOURS'] = table2['ACTUAL WORK HOURS'] - table2['TOTAL WORKED HOURS'] # added if error remove
            table2['TOTAL WORKED HOURS']  = table2['TOTAL WORKED HOURS'].apply(lambda x : hrs_to_min(x)) # added if error remove
            table2['ACTUAL WORK HOURS']  = table2['ACTUAL WORK HOURS'].apply(lambda x : hrs_to_min(x)) # added if error remove
            table2['ACTUAL WORK HOURS'] = table2['ACTUAL WORK HOURS'].apply(lambda x : decimal_correct(x)) # added if error remove 
            table2['COMPENSATION HOURS'] = table2['COMPENSATION HOURS'].apply(lambda x : hrs_to_min(x)) # added if error remove
            
            # print(table2)
            
            comp_time = table2['COMPENSATION HOURS'].iloc[0]
            
            table2['TOTAL WORKED HOURS']  = table2['TOTAL WORKED HOURS'].apply(lambda x : float_to_hrs_min(x)) # added if error remove
            table2['ACTUAL WORK HOURS']  = table2['ACTUAL WORK HOURS'].apply(lambda x : float_to_hrs_min(x)) # added if error remove
            table2['COMPENSATION HOURS'] = table2['COMPENSATION HOURS'].apply(lambda x : float_to_hrs_min(abs(x))) # added if error remove

            
            if comp_time < 0:
                table2.rename(columns = {'COMPENSATION HOURS':'Extra Worked Hours'},inplace=True)
            

            body = """
            <p>Kindly review your timesheet login details, and refer to the attached detailed report on your work timings. Requesting you to complete pending compensation hours if present. If you notice any mistakes in the report, please notify us at your earliest convenience. Your cooperation is appreciated.</p>
            <p>Thanks & Regards,  <br>
            Pilog Zone 3,  <br>
            Data Science Team.</p>
            """


            # Create a multipart message
            message = MIMEMultipart()
            message["From"] = email_sender
            message["To"] = recipient_mail
            message["Subject"] = subject

            # Attach HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Email Notification</title>
                <style>
                    body {{
                        font-family: 'Arial', sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f5f5f5;
                    }}

                    .email-container {{
                        max-width: 900px;
                        width: 95%;
                        margin: 30px auto;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        position: relative; /* Add this to make position:relative work */
                    }}

                    /* Add an extra shadow only at the top */
                    .email-container::before {{
                        content: '';
                        position: absolute;
                        top: -8px;
                        left: 0;
                        right: 0;
                        height: 8px;
                        background: rgba(0, 0, 0, 0.1);
                        border-radius: 8px 8px 0 0;
                    }}

                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 20px;
                    }}

                    th, td {{
                        border: 1px solid #ddd;
                        padding: 12px;
                        text-align: left;
                    }}

                    th {{
                        background-color: #007BFF;
                        color: #fff;
                    }}
                </style>
            </head>
            <body>

                <div class="email-container">
                    <h2>Hi {employee},</h2>
                    <p>{body}</p>

                    <!-- Include the formatted tables here -->
                    {build_table(table1, 'blue_light')}

                    {build_table(table2, 'blue_light')}
                </div>

            </body>
            </html>
            """



            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # pdf_file_path = None
            
            print(f"employee_path: {employee_path}")
            print(f"pdf_list: {pdf_list}")
            
            for path in pdf_list:
                if employee_path == path.rstrip('.pdf'):
                    pdf_file_path = fr"{output_folder_path}\{path}"
                    break  
            # Attach a PDF file
            # pdf_file_path = file_path  # Replace with the actual path to your PDF file
            pdf_file_name = f"{employee}.pdf"  # Replace with the desired name for the attached PDF file

            with open(pdf_file_path, "rb") as pdf_attachment:
                pdf_part = MIMEBase("application", "octet-stream")
                pdf_part.set_payload(pdf_attachment.read())

            encoders.encode_base64(pdf_part)
            pdf_part.add_header("Content-Disposition", f"attachment; filename= {pdf_file_name}")
            message.attach(pdf_part)

            # Create a connection and send the email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
                server.ehlo()
                server.login(email_sender, email_password)
                server.sendmail(email_sender, recipient_mail, message.as_string())
            
            sent_list.append(employee)
        except Exception as e:
            print(e)
            fail_list.append(employee)
            pass
    return  sent_list, fail_list


# df = pd.read_excel(r"C:\Users\Rahul\Desktop\Onedrive-sharepoint-Login_Time\dash_fastapi\ds_team.xlsx")

# if __name__ == "__main__":

#     result = test_run(r"C:\Users\Rahul\Desktop\Onedrive-sharepoint-Login_Time\dash_fastapi\ds_team_2.xlsx",'January')
#     mail_sending(r"C:\Users\Rahul\Desktop\Onedrive-sharepoint-Login_Time\dash_fastapi\ds_team_2.xlsx",result,'January')
    
def employee_api_mail(content,month):
    result = test_run(content,month)
    emp_sent_list, emp_fail_list = mail_sending(content,result,month)
    
    return {"Statue":"mails_sent_successfully","Sent_list":emp_sent_list,"Failed_list":emp_fail_list,'sent_count':len(emp_sent_list),'fail_count':len(emp_fail_list)}
    