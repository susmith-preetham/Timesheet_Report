import dash
import ssl
import base64
import math
import time
import psutil
import warnings
import smtplib
import threading
import multiprocessing
import os, signal
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc 
from dash import dcc
from dash import html
from datetime import datetime
from dash.dependencies import Input, Output
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.print_page_options import PrintOptions
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pretty_html_table import build_table
from tqdm import tqdm

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

def twentyfour_hrs_frmt(value):
    value = str(value)
    if value.startswith('24'):
        value = value.replace('24','00')
        
    return(float(value))

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
    if 12.3<=out_time_hlf<=15.0:   ######################### Updated
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
    # Melt the original dataframe to transform it into a long format
    melted_df = pd.melt(dataframe, id_vars=['EMPLOYEE NAME','DEPARTMENT'], var_name='datetime_type', value_name='value')

    # Extract date, time, and type from the 'datetime_type' column
    melted_df[['date', 'type']] = melted_df['datetime_type'].str.split(' ', n=1, expand=True)

    # Pivot the melted dataframe to get the desired format
    final_df = melted_df.pivot_table(index=['EMPLOYEE NAME','DEPARTMENT', 'date'], columns='type', values='value', aggfunc='first').reset_index()

    # Rename columns
    final_df.columns.name = None
    final_df.columns = ['EMPLOYEE NAME','DEPARTMENT', 'DATE', 'IN TIME', 'OUT TIME', 'TOTAL']

    final_df["TOTAL"] = final_df.apply(lambda row: 0 if row['IN TIME'] == 'L' else row['TOTAL'], axis=1)
    final_df["IN TIME"] = final_df["IN TIME"].apply(lambda x: np.nan if x == 'L' else x)
    final_df["IN TIME"] = final_df["IN TIME"].apply(twentyfour_hrs_frmt) # added
    
    final_df["OUT TIME"] = final_df["OUT TIME"].apply(lambda x: np.nan if x == 'L' else x)
    final_df["OUT TIME"] = final_df["OUT TIME"].apply(twentyfour_hrs_frmt) # added
    
    
    final_df["DATE"] = pd.to_datetime(final_df["DATE"])
    final_df["IN TIME"] = final_df["IN TIME"].astype(float)
    final_df["OUT TIME"] = final_df["OUT TIME"].astype(float)
    final_df["IN TIME_AVG"] = final_df["IN TIME"].apply(half_day_in)
    final_df["OUT TIME_AVG"] = final_df["OUT TIME"].apply(half_day_out)
    final_df["TOTAL"] = final_df["TOTAL"].astype(float)  
    final_df["TOTAL"] = final_df["TOTAL"].apply(lambda x : 0 if x<0 else x)
    # final_df["TOTAL"] = final_df["TOTAL"].apply(lambda x : adjust_decimal_values(x))
    final_df = final_df.dropna(thresh=0.85*len(final_df.columns))
    final_df["TOTAL"] = final_df.apply(lambda x : calculate_time_difference(x['IN TIME'],x['OUT TIME']),axis = 1)
    final_df["TOTAL_HLF"] = final_df.apply(lambda x : calculate_time_difference(x['IN TIME_AVG'],x['OUT TIME_AVG']),axis = 1)
    

    # Merge with the original 'Grand Total' columns
    final_df = final_df.merge(dataframe[['EMPLOYEE NAME', 'Grand Total Worked', 'Grand Total Grace Period', 'Grand Total Actual', 'Grand Total Final']], on='EMPLOYEE NAME', how='left')
    
    final_df["Grand Total Worked"] = final_df.groupby(["EMPLOYEE NAME"])["TOTAL_HLF"].transform('sum')
    final_df["Grand Total Actual"] = len(final_df["DATE"].unique())*9.5
    final_df["Grand Total Actual_real"] = final_df.groupby(['EMPLOYEE NAME'])['DATE'].transform('count') * 9.5

    final_df["Grand Total Final"] = final_df["Grand Total Actual"] - final_df["Grand Total Worked"]
    final_df["Grand Total Final_real"] = final_df["Grand Total Actual_real"] - final_df["Grand Total Worked"]
    
    
    return final_df

def leave_create(dataframe):
    dataframe["Grand Total Actual"] = np.nan
    dataframe["Grand Total Final"] = np.nan
    dataframe["Grand Total Grace Period"] = np.nan
    dataframe["Grand Total Worked"] = np.nan    
    # Melt the original dataframe to transform it into a long format
    melted_df = pd.melt(dataframe, id_vars=['EMPLOYEE NAME','DEPARTMENT'], var_name='datetime_type', value_name='value')

    # Extract date, time, and type from the 'datetime_type' column
    melted_df[['date', 'type']] = melted_df['datetime_type'].str.split(' ', n=1, expand=True)

    # Pivot the melted dataframe to get the desired format
    final_df = melted_df.pivot_table(index=['EMPLOYEE NAME','DEPARTMENT', 'date'], columns='type', values='value', aggfunc='first').reset_index()

    # Rename columns
    final_df.columns.name = None
    final_df.columns = ['EMPLOYEE NAME','DEPARTMENT', 'DATE', 'IN TIME', 'OUT TIME', 'TOTAL']

    final_df['LEAVES'] = final_df.apply(lambda x: 'LEAVE' if x['IN TIME'] == 'L' or x['OUT TIME'] == 'L' else 'WORK', axis=1) # ADDED FOR LEAVE COUNT
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
    dataframe["Average In Time "] = average_time(dataframe["IN TIME"])
    dataframe["Average Out Time"] = average_time(dataframe["OUT TIME"])
    dataframe["Average In Time "] = pd.to_datetime(dataframe["Average In Time "], format='%H:%M').dt.time
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
    df_avg_vals = dataframe[["Average In Time ","Average Out Time","Avg Grand Total Worked","Grand Total Actual","Avg Per day Work","Avg Grand Total Compensation"]].drop_duplicates().reset_index(drop = True)
    df_avg_vals["Total Work Hrs"] = dataframe["Grand Total Worked"].sum()
    df_avg_vals["Total Actual Work Hrs"] = dataframe["Grand Total Actual"].sum()
    df_avg_vals["Total Pending Hrs"] = df_avg_vals["Total Actual Work Hrs"] - df_avg_vals["Total Work Hrs"]
    df_avg_vals["Actual In Time"] = "09:00"
    df_avg_vals["Actual In Time"] = pd.to_datetime(df_avg_vals["Actual In Time"], format='%H:%M').dt.time
    df_avg_vals["Actual Out Time"] = "18:30"
    df_avg_vals["Actual Out Time"] = pd.to_datetime(df_avg_vals["Actual Out Time"], format='%H:%M').dt.time
    df_avg_vals["Actual In Time_temp"] = df_avg_vals["Actual In Time"].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))
    df_avg_vals["Average In Time _temp"] = df_avg_vals["Average In Time "].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))
    # Calculate time difference
    df_avg_vals["In Time Difference"] = df_avg_vals["Average In Time _temp"] - df_avg_vals["Actual In Time_temp"]
    # Convert timedelta64 to hours and minutes
    df_avg_vals["In Time Difference"] = df_avg_vals["In Time Difference"].apply(lambda x: f"{int(x.seconds // 3600)} hr {int((x.seconds % 3600) // 60)} min")
    
    df_avg_vals["Actual Out Time_temp"] = df_avg_vals["Actual Out Time"].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))
    df_avg_vals["Average Out Time_temp"] = df_avg_vals["Average Out Time"].apply(lambda x: pd.to_datetime(x.strftime('%H:%M'), format='%H:%M'))

    # Calculate time difference
    df_avg_vals["Out Time Difference"] = df_avg_vals["Average Out Time_temp"] - df_avg_vals["Actual Out Time_temp"]

    # Convert timedelta64 to hours and minutes
    df_avg_vals["Out Time Difference"] = df_avg_vals["Out Time Difference"].apply(lambda x: f"{int(x.seconds // 3600)} hr {int((x.seconds % 3600) // 60)} min")
    
    new_column_order = ['Average In Time ', 'Actual In Time', 'In Time Difference', 
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


def apply_all_functions(dataframe):
    dataframe.drop(['EMPLOYEE OFFICIAL EMAIL ID'],axis = 1,inplace=True)  # added
    dataframe = removing_unnecessary_values(dataframe)
    dataframe = columns_name_change(dataframe)
    Leave_count = (dataframe == 'L').stack().sum()
    leave_df = leave_create(dataframe)
    dataframe = create_dataframe(dataframe)
    dataframe = add_week_no(dataframe)
    dataframe = new_columns(dataframe)
    overall_avg = overall_avg_df(dataframe)
    day_wise = day_wise_df(dataframe)
    date_wise = date_wise_df(dataframe)
    return dataframe, overall_avg, day_wise, date_wise, leave_df, Leave_count
def filter_emp_name(dataframe):
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

def card_values(values):
    values = str(values)
    if '.' in values:
        number = values.split('.')[0]
        decimal = values.split('.')[1]
        if len(decimal) == 1:
            decimal = decimal + '0'
        else:
            decimal = decimal
        final_number = number + '.' + decimal
    else:
        final_number = values
    return final_number

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])

def kill_process_by_port(port):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if f':{port}' in proc.info['cmdline']:
                print(f"Killing process {proc.info['pid']} using port {port}")
                os.kill(proc.info['pid'], 9)  # Send SIGKILL signal
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def dash_app_selenium(df):
    # Applying Functions to Dataframe
    final_df, Avg_df, day_df, date_df, leave_df, leave_count = apply_all_functions(df)
    
    final_df['DEPARTMENT'] = final_df['DEPARTMENT'].apply(lambda x : x.replace('-','_'))
    leave_df['DEPARTMENT'] = leave_df['DEPARTMENT'].apply(lambda x : x.replace('-','_'))
    
    print(leave_df)
    
    # employee_count = len(final_df["EMPLOYEE NAME"].unique())
    day_count = len(final_df['DATE'].unique())
    in_count = len(final_df["IN TIME"])
    latest_date = final_df['DATE'].max().strftime('%Y-%m-%d')
    # Header_df = pd.DataFrame({
    #                         "Days":day_count,
    #                         "Hours":day_count*9.5,
    #                         "Total Employees":employee_count,
    #                         "Till Date":latest_date
    # },index = [0,1,2,3])
    # Header_df['Hours'] = Header_df['Hours'].apply(hrs_to_min)

    exclude_emp = ['ASIF AHMED MOHAMMAD','PARDHA SARADHI. KATTA','TATI SASI KRISHNA','KOTI AZMIRA',
                'KHAJA SALEEMUDDIN','M CHENNAKESHAVULU','MOHAMMED SHOUKAT ALI','VENKAYALA V. UDAY KIRAN','SYED IMTIAZ','MOHD AKHEEL','NAZIMA ERSHAD']
    
    final_df = final_df[~final_df['EMPLOYEE NAME'].isin(exclude_emp)].reset_index(drop = True)

    in_df = Avg_df[["Average In Time ","Actual In Time","In Time Difference"]]
    out_df = Avg_df[["Average Out Time","Actual Out Time","Out Time Difference"]]

    grand_total_df = Avg_df[["Avg Grand Total Worked","Grand Total Actual"]]
    grand_total_df["Avg Grand Total Difference"] = grand_total_df["Grand Total Actual"] - grand_total_df["Avg Grand Total Worked"]
    grand_total_df = grand_total_df.round(2) # added
    
    # grand_total_df = grand_total_df.applymap(hrs_to_min)
    work_hrs = Avg_df[["Avg Per day Work"]]
    work_hrs["Actual Per day Work"] = 9.5
    work_hrs["Time Difference"] = work_hrs["Actual Per day Work"] - work_hrs["Avg Per day Work"]
    work_hrs = work_hrs.round(2) # added
    # work_hrs = work_hrs.applymap(hrs_to_min)

    Total_work = pd.DataFrame({"Total Work Hrs":final_df["TOTAL"].sum(),
                # "Total Actual Work Hrs":day_count*(employee_count)*9.5
                "Total Actual Work Hrs":final_df.shape[0]*9.5
                },index = [0])
    Total_work["Total Pending Hrs"] = Total_work["Total Actual Work Hrs"] - Total_work["Total Work Hrs"]
    Total_work = Total_work.round(2) # added

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

    date_df['DATETIME_IN'] = pd.to_datetime(date_df['DATETIME_IN'])

    # Extract Date and Time columns
    date_df['DATE'] = date_df['DATETIME_IN'].dt.date
    date_df['TIME_IN'] = date_df['DATETIME_IN'].dt.time

    # Sort DataFrame by time
    date_df = date_df.sort_values(by='TIME_IN')

    # Convert time values to datetime format
    date_df['TIME_DATETIME_IN'] = pd.to_datetime(date_df['TIME_IN'].astype(str))

    # Group by Date and aggregate Time (using the first time value of each date)
    grouped_data_date_in = date_df.groupby('DATE')['TIME_DATETIME_IN'].first().reset_index()

    date_df['DATETIME_OUT'] = pd.to_datetime(date_df['DATETIME_OUT'])

    # Extract Date and Time columns
    date_df['DATE'] = date_df['DATETIME_OUT'].dt.date
    date_df['TIME_OUT'] = date_df['DATETIME_OUT'].dt.time

    # Sort DataFrame by time
    date_df = date_df.sort_values(by='TIME_OUT')

    # Convert time values to datetime format
    date_df['TIME_DATETIME_OUT'] = pd.to_datetime(date_df['TIME_OUT'].astype(str))

    # Group by Date and aggregate Time (using the first time value of each date)
    grouped_data_date_out = date_df.groupby('DATE')['TIME_DATETIME_OUT'].first().reset_index()
    
    # # Avg in time and out time
    # AVG_TIME_ADMIN = final_df.groupby(['EMPLOYEE NAME'])['IN TIME'].mean().round(2).reset_index()
    # AVG_TIME_ADMIN['OUT TIME'] = final_df.groupby(['EMPLOYEE NAME'])['OUT TIME'].transform('mean').round(2)
    # AVG_TIME_ADMIN['IN TIME'] = AVG_TIME_ADMIN['IN TIME'].apply(hrs_to_min)
    # AVG_TIME_ADMIN['OUT TIME'] = AVG_TIME_ADMIN['OUT TIME'].apply(hrs_to_min)
    
    
    
    # bin_edges_in = [4,8.30,9, 9.30, 10, 10.30, 19]
    # bin_edges_out = [9, 18.30, 19, 19.30, 20,20.30,23.50]
    # bin_names_in = ['before 8:30','8:30 - 9:00','9:00 - 9:30', '9:30 - 10:00', '10:00 - 10:30', '10:30 and above']
    # bin_names_out = ['before 18:30', '18:30 - 19:00', '19:00 - 19:30', '19:30 - 20:00', '20:00 - 20:30', '20:30 and above']
    # AVG_TIME_ADMIN['bin_in'] = pd.cut(AVG_TIME_ADMIN['IN TIME'], bins=bin_edges_in, labels=bin_names_in)
    # AVG_TIME_ADMIN['bin_out'] = pd.cut(AVG_TIME_ADMIN['OUT TIME'], bins=bin_edges_out, labels=bin_names_out)

    # intime_df = AVG_TIME_ADMIN.groupby('bin_in').size().reset_index(name='count')
    # outtime_df = AVG_TIME_ADMIN.groupby('bin_out').size().reset_index(name='count')
    
    # Avg in time and out time
    AVG_TIME_ADMIN = final_df[['EMPLOYEE NAME','IN TIME_AVG','OUT TIME_AVG']]
    AVG_TIME_ADMIN['hours_in'] = AVG_TIME_ADMIN['IN TIME_AVG'].astype(int)
    AVG_TIME_ADMIN['minutes_in'] = ((AVG_TIME_ADMIN['IN TIME_AVG'] - AVG_TIME_ADMIN['hours_in']) * 100).astype(int)
    
    AVG_TIME_ADMIN['IN TIME_AVG'] = AVG_TIME_ADMIN.apply(lambda row : datetime.strptime(f"{row['hours_in']:02d}:{row['minutes_in']:02d}", "%H:%M"), axis =1)
 
    # AVG_TIME_ADMIN['IN TIME_AVG'] = pd.to_datetime(AVG_TIME_ADMIN[['hours_in', 'minutes_in']].astype(str).agg(':'.join, axis=1), format='%H:%M')
    
    AVG_TIME_ADMIN['hours_out'] = AVG_TIME_ADMIN['OUT TIME_AVG'].astype(int)
    AVG_TIME_ADMIN['minutes_out'] = ((AVG_TIME_ADMIN['OUT TIME_AVG'] - AVG_TIME_ADMIN['hours_out']) * 100).astype(int)
    
    AVG_TIME_ADMIN['OUT TIME_AVG'] = AVG_TIME_ADMIN.apply(lambda row : datetime.strptime(f"{row['hours_out']:02d}:{row['minutes_out']:02d}", "%H:%M"), axis = 1)
    
    # AVG_TIME_ADMIN['OUT TIME_AVG'] = pd.to_datetime(AVG_TIME_ADMIN[['hours_out', 'minutes_out']].astype(str).agg(':'.join, axis=1), format='%H:%M')
    
    AVG_TIME_ADMIN_IN = AVG_TIME_ADMIN.groupby(['EMPLOYEE NAME'])['IN TIME_AVG'].mean().reset_index()
    AVG_TIME_ADMIN_OUT = AVG_TIME_ADMIN.groupby(['EMPLOYEE NAME'])['OUT TIME_AVG'].mean().reset_index()

    bin_edges_in = pd.to_datetime(['00:00','08:30', '09:00', '09:30', '10:00', '10:30', '23:59'],format='%H:%M')
    bin_names_in = ['before 8:30','8:30 - 9:00','9:00 - 9:30', '9:30 - 10:00', '10:00 - 10:30', '10:30 and above']
    bin_edges_out = pd.to_datetime(['09:00', '18:30', '19:00', '19:30', '20:00', '20:30', '23:50'],format='%H:%M')
    bin_names_out = ['before 18:30', '18:30 - 19:00', '19:00 - 19:30', '19:30 - 20:00', '20:00 - 20:30', '20:30 and above']
    AVG_TIME_ADMIN_IN['bin_in'] = pd.cut(AVG_TIME_ADMIN_IN['IN TIME_AVG'], bins=bin_edges_in, labels=bin_names_in)
    AVG_TIME_ADMIN_OUT['bin_out'] = pd.cut(AVG_TIME_ADMIN_OUT['OUT TIME_AVG'], bins=bin_edges_out, labels=bin_names_out)

    intime_df = AVG_TIME_ADMIN_IN.groupby('bin_in').size().reset_index(name='count')
    outtime_df = AVG_TIME_ADMIN_OUT.groupby('bin_out').size().reset_index(name='count')
    
    
    employee_count = len(AVG_TIME_ADMIN["EMPLOYEE NAME"].unique())
    
    # Header_df = pd.DataFrame({
    #                         "Days":day_count,
    #                         "Hours":day_count*9.5,
    #                         "Total Employees":employee_count,
    #                         "Till Date":latest_date
    # },index = [0,1,2,3])
    
    work_hrs_df = final_df[['EMPLOYEE NAME','DEPARTMENT','DATE','Grand Total Worked']].reset_index(drop = True)
    df_present_count = work_hrs_df['EMPLOYEE NAME'].value_counts()*9.5
    df_present_count = df_present_count.reset_index()
    work_hrs_df = pd.merge(work_hrs_df,df_present_count,on="EMPLOYEE NAME",how = 'left')
    work_hrs_df["Time Difference"] = work_hrs_df["Grand Total Worked"] - work_hrs_df["count"] 
    work_hrs_df = work_hrs_df.sort_values(by = 'Time Difference',ascending=False)
    top_5 = work_hrs_df[["EMPLOYEE NAME",'DEPARTMENT',"Time Difference"]].drop_duplicates()
    top_5 = top_5.head(10)
    top_5["Time Difference"] = top_5["Time Difference"].round(2)
    top_5["Time Difference"] = top_5["Time Difference"].apply(hrs_to_min)
    top_5["Time Difference"] = top_5["Time Difference"].apply(card_values)
    top_5.columns = ["Employee Name",'Department',"Extra Worked Hours"]
    bottom_5 = work_hrs_df[["EMPLOYEE NAME",'DEPARTMENT',"Time Difference"]].drop_duplicates()
    bottom_5 = bottom_5.tail(10).sort_values(by = 'Time Difference',ascending=True)
    bottom_5["Time Difference"] = bottom_5["Time Difference"].round(2)
    bottom_5["Time Difference"] = abs(bottom_5["Time Difference"])
    bottom_5["Time Difference"] = bottom_5["Time Difference"].apply(hrs_to_min)
    bottom_5["Time Difference"] = bottom_5["Time Difference"].apply(card_values)
    bottom_5["Time Difference"] = bottom_5["Time Difference"].apply(lambda x : "-" + str(x))
    bottom_5.columns = ["Employee Name",'Department',"Compensation Work Hours"]

    emp_avg_time = final_df[['EMPLOYEE NAME','DEPARTMENT','IN TIME_AVG','OUT TIME_AVG']]
    emp_avg_time["Average In Time"] = emp_avg_time.groupby(["EMPLOYEE NAME"])['IN TIME_AVG'].transform(average_time)
    emp_avg_time["Average Out Time"] = emp_avg_time.groupby(["EMPLOYEE NAME"])['OUT TIME_AVG'].transform(average_time)
    # emp_avg_time = emp_avg_time.drop(['IN TIME_AVG','OUT TIME_AVG','Average Out Time'],axis = 1).drop_duplicates().reset_index(drop=True)
    # emp_avg_time = emp_avg_time.sort_values(by='Average In Time', ascending=True).reset_index(drop=True)
    # emp_avg_time = emp_avg_time[~emp_avg_time['EMPLOYEE NAME'].isin(exclude_emp)]
    # emp_avg_time_head = emp_avg_time.head(10)
    # emp_avg_time_head.columns = ["Employee Name","Average In Time"]
    # emp_avg_time_tail = emp_avg_time.tail(10)
    # emp_avg_time_tail = emp_avg_time_tail.sort_values(by='Average In Time', ascending=False).reset_index(drop=True)
    # emp_avg_time_tail.columns = ["Employee Name","Average In Time"]
    emp_avg_time_in = emp_avg_time.drop(['IN TIME_AVG','OUT TIME_AVG','Average Out Time'],axis = 1).drop_duplicates().reset_index(drop=True)
    emp_avg_time_in = emp_avg_time_in.sort_values(by='Average In Time', ascending=True).reset_index(drop=True)
    emp_avg_time_in = emp_avg_time_in[~emp_avg_time_in['EMPLOYEE NAME'].isin(exclude_emp)]
    emp_avg_time_in_head = emp_avg_time_in.head(10)
    emp_avg_time_in_head.columns = ["Employee Name","Department","Average In Time"]
    emp_avg_time_in_tail = emp_avg_time_in.tail(10)
    emp_avg_time_in_tail = emp_avg_time_in_tail.sort_values(by='Average In Time', ascending=False).reset_index(drop=True)
    emp_avg_time_in_tail.columns = ["Employee Name","Department","Average In Time"]
    
    emp_avg_time_out = emp_avg_time.drop(['IN TIME_AVG','OUT TIME_AVG','Average In Time'],axis = 1).drop_duplicates().reset_index(drop=True)
    emp_avg_time_out = emp_avg_time_out.sort_values(by='Average Out Time', ascending=True).reset_index(drop=True)
    emp_avg_time_out = emp_avg_time_out[~emp_avg_time_out['EMPLOYEE NAME'].isin(exclude_emp)]
    emp_avg_time_out_head = emp_avg_time_out.head(10)
    emp_avg_time_out_head.columns = ["Employee Name","Department","Average Out Time"]
    emp_avg_time_out_tail = emp_avg_time_out.tail(10)
    emp_avg_time_out_tail = emp_avg_time_out_tail.sort_values(by='Average Out Time', ascending=False).reset_index(drop=True)
    emp_avg_time_out_tail.columns = ["Employee Name","Department","Average Out Time"]
    
    
    # LEAVE COUNT
    leave_count = int(leave_count/2)
    leave_df_cnt = leave_df[['EMPLOYEE NAME','LEAVES','DEPARTMENT']]
    leave_df_cnt = leave_df_cnt[leave_df_cnt['LEAVES'] == 'LEAVE'].groupby(['EMPLOYEE NAME','DEPARTMENT'])['LEAVES'].value_counts().reset_index(name = 'Leave Count').sort_values(by='Leave Count', ascending=False)
    leave_df_cnt.drop(['LEAVES'],axis = 1, inplace = True)
    leave_df_cnt.rename(columns = {'EMPLOYEE NAME':'Employee Name','DEPARTMENT':'Department'},inplace = True)
    print(leave_df_cnt['Leave Count'].sum())
    leave_df_cnt = leave_df_cnt.head(10)
    
    leave_statement = f"Total Number of Leaves Taken: {leave_count}"
    
    # leave_df_cnt = leave_df[['EMPLOYEE NAME','LEAVES']]
    # leave_df_cnt = leave_df_cnt[leave_df_cnt['LEAVES'] == 'LEAVE']
    # leave_df_cnt = leave_df_cnt.groupby('EMPLOYEE NAME').size().reset_index(name='Leave Count')
    # leave_df_cnt = leave_df_cnt.sort_values(by='Leave Count', ascending=False)
    # leave_df_cnt.rename(columns={'EMPLOYEE NAME': 'Employee Name'}, inplace=True)
    # leave_df_cnt = leave_df_cnt.head(10)
    # leave_count = leave_df_cnt['Leave Count'].sum()
    # leave_statement = f"Total Number of Leaves Taken: {leave_count}"
    
    half_day_count = (((day_count*9.5*employee_count) - Total_work["Total Actual Work Hrs"]) - (leave_count*9.5))/4.75
    leave_count = leave_count + half_day_count
    
    # Header_df = pd.DataFrame({
    #                         "Working Days":day_count,
    #                         "Work Hrs Per Emp":day_count*9.5,
    #                         "Leaves":leave_count,
    #                         # "Half Leaves": half_day_count,
    #                         "Total Employees":employee_count,
    #                         "Till Date":latest_date
    # },index = [0,1,2,3])

    # Color Palette
    colors = {
        'background': '#c2d9ed',#'#f0f8ff',  # Light blue background
        'dark background':'#fafafa', #a9cceb',
        'text': 'black',  # Black text color
        'plotly_blue': '#1f77b4',  # Plotly default blue color 
        'text_card' : '#000000',
        'banner_color':'white',
    }

    bar_colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    ##### DEPARTMENT
    department_df = final_df[['EMPLOYEE NAME','DEPARTMENT','Grand Total Worked','Grand Total Actual_real','Grand Total Final_real']]
    department_df.drop_duplicates(inplace = True)
    
    total_work_df = department_df
    total_work_df['Total Work Hrs'] = department_df.groupby(['DEPARTMENT'])['Grand Total Worked'].transform('sum')
    total_work_df['Actual Work Hrs'] = department_df.groupby(['DEPARTMENT'])['Grand Total Actual_real'].transform('sum')
    total_work_df.sort_values(by = 'Total Work Hrs',ascending=False,inplace=True)
    total_work_df = total_work_df[['DEPARTMENT','Total Work Hrs','Actual Work Hrs']]
    total_work_df.drop_duplicates(inplace=True)
    total_work_df['Total Work Hrs'] = total_work_df['Total Work Hrs'].astype(int)
    total_work_df['Actual Work Hrs'] = total_work_df['Actual Work Hrs'].astype(int)
    


    dep_avg_df = final_df.groupby(['EMPLOYEE NAME','DEPARTMENT'])['TOTAL_HLF'].mean().reset_index(name = 'Average Work Hours')
    dep_avg_df['Average Work Hours'] = dep_avg_df.groupby('DEPARTMENT')['Average Work Hours'].transform(average_time)
    dep_avg_df['Average Work Hours'] = dep_avg_df['Average Work Hours'].apply(lambda x : float(str(x).replace(':', '.')))
    
    dep_avg_df_count = dep_avg_df['DEPARTMENT'].value_counts().reset_index(name = 'Employee Count')  ####
    
    dep_avg_df = dep_avg_df[['DEPARTMENT','Average Work Hours']].drop_duplicates()
    dep_avg_df.sort_values(by = 'Average Work Hours', ascending = False, inplace = True)
    
    

    leave_df_dep = leave_df[['EMPLOYEE NAME', 'DEPARTMENT', 'LEAVES']]
    print(leave_df_dep)
    leave_df_dep['LEAVES'] = leave_df_dep['LEAVES'].apply(lambda x: 1 if x == 'LEAVE' else 0)
    leaves_count_department = leave_df_dep.groupby(['DEPARTMENT'])['LEAVES'].sum().reset_index(name = 'Leaves Count')
    leaves_count_department.sort_values(by = 'Leaves Count', ascending=False, inplace=True)
    
    leaves_count_department = leaves_count_department[leaves_count_department['Leaves Count'] != 0]    
    
    print(leaves_count_department['Leaves Count'].sum())
    
    
    Header_df = pd.DataFrame({
                        "Working Days":day_count,
                        "Work Hrs Per Emp":day_count*9.5,
                        "Leaves":leaves_count_department['Leaves Count'].sum(),
                        # "Half Leaves": half_day_count,
                        "Total Employees":employee_count,
                        "Till Date":latest_date
    },index = [0,1,2,3])
    # leaves_count_department_sorted = leaves_count_department.sort_values(by='Leaves Count', ascending=False)
    # top_7_departments = leaves_count_department_sorted.head(10)


    Compensation_hrs_department = department_df
    Compensation_hrs_department['Compensation Hrs'] = Compensation_hrs_department.groupby(['DEPARTMENT'])['Grand Total Final_real'].transform('sum')
    Compensation_hrs_department['Average Compensation Hrs'] = Compensation_hrs_department.groupby(['DEPARTMENT'])['Grand Total Final_real'].transform('mean')
    Compensation_hrs_department.sort_values(by='Average Compensation Hrs', ascending=False, inplace=True)
    Compensation_hrs_department = Compensation_hrs_department[['DEPARTMENT','Compensation Hrs','Average Compensation Hrs']]
    Compensation_hrs_department.drop_duplicates(inplace=True)
    Compensation_hrs_department['Compensation Hrs'] = Compensation_hrs_department['Compensation Hrs'].apply(lambda x : str(round(x,2)))
    Compensation_hrs_department['Average Compensation Hrs'] = Compensation_hrs_department['Average Compensation Hrs'].apply(lambda x : str(round(x,2)))
    Compensation_hrs_department = pd.merge(Compensation_hrs_department,dep_avg_df_count, on = "DEPARTMENT", how =  'left')
    Compensation_hrs_department.rename(columns = {'DEPARTMENT':'Department'},inplace = True)
    compensation_df = Compensation_hrs_department.head(10)
    extra_work_df = Compensation_hrs_department.tail(10)
    extra_work_df['Compensation Hrs'] = extra_work_df['Compensation Hrs'].apply(lambda x : str(abs(round(float(x),2))) if float(x)<0 else np.nan)
    extra_work_df['Average Compensation Hrs'] = extra_work_df['Average Compensation Hrs'].apply(lambda x : str(abs(round(float(x),2))) if float(x)<0 else np.nan)
    extra_work_df.sort_values(by = 'Average Compensation Hrs',ascending = False,inplace = True)
    extra_work_df.dropna(inplace = True)
    extra_work_df.rename(columns = {'Compensation Hrs':'Extra Worked Hrs','Average Compensation Hrs':'Average Extra Worked Hrs'},inplace = True)


    final_df['Emp_Avg_IN TIME'] = final_df.groupby(['EMPLOYEE NAME'])['IN TIME_AVG'].transform('mean')
    final_df['Emp_Avg_OUT TIME'] = final_df.groupby(['EMPLOYEE NAME'])['OUT TIME_AVG'].transform('mean')
    
    ##########
    
    # final_df['Dep_Avg_IN TIME'] = final_df.groupby(['DEPARTMENT'])['IN TIME_AVG'].transform('mean')
    # final_df['Dep_Avg_OUT TIME'] = final_df.groupby(['DEPARTMENT'])['OUT TIME_AVG'].transform('mean')
    # dep_avg_time = final_df[['EMPLOYEE NAME','DEPARTMENT','Dep_Avg_IN TIME','Dep_Avg_OUT TIME']]
    # dep_avg_time['Dep_Avg_IN TIME'] = dep_avg_time['Dep_Avg_IN TIME'].apply(float_to_string)
    # dep_avg_time['Dep_Avg_OUT TIME'] = dep_avg_time['Dep_Avg_OUT TIME'].apply(float_to_string)
    # dep_avg_time['Dep_Avg_IN TIME'] = dep_avg_time['Dep_Avg_IN TIME'].apply(lambda x: datetime.strptime(x, "%H:%M"))
    # dep_avg_time['Dep_Avg_OUT TIME'] = dep_avg_time['Dep_Avg_OUT TIME'].apply(lambda x: datetime.strptime(x, "%H:%M"))
    
    
    final_df['Dep_Avg_IN TIME'] = final_df.groupby(['DEPARTMENT'])['IN TIME_AVG'].transform(average_time)
    final_df['Dep_Avg_OUT TIME'] = final_df.groupby(['DEPARTMENT'])['OUT TIME_AVG'].transform(average_time)
    ##########
    
    # final_df['Dep_Avg_IN TIME'] = final_df.groupby(['DEPARTMENT'])['Emp_Avg_IN TIME'].transform(average_time)
    # final_df['Dep_Avg_OUT TIME'] = final_df.groupby(['DEPARTMENT'])['Emp_Avg_OUT TIME'].transform(average_time)
    dep_avg_time = final_df[['EMPLOYEE NAME','DEPARTMENT','Dep_Avg_IN TIME','Dep_Avg_OUT TIME']]
    dep_avg_time['Dep_Avg_IN TIME'] = dep_avg_time['Dep_Avg_IN TIME'].apply(lambda x: datetime.strptime(x, "%H:%M"))
    dep_avg_time['Dep_Avg_OUT TIME'] = dep_avg_time['Dep_Avg_OUT TIME'].apply(lambda x: datetime.strptime(x, "%H:%M"))
    dep_avg_in_df = dep_avg_time.groupby(['DEPARTMENT'])['Dep_Avg_IN TIME'].mean().reset_index()
    dep_avg_in_df.sort_values(by = 'Dep_Avg_IN TIME', ascending = False, inplace = True)
    dep_avg_out_df = dep_avg_time.groupby(['DEPARTMENT'])['Dep_Avg_OUT TIME'].mean().reset_index()
    dep_avg_out_df.sort_values(by = 'Dep_Avg_OUT TIME', ascending = True, inplace = True)
    dep_avg_in_df['Dep_Avg_IN TIME'] = dep_avg_in_df['Dep_Avg_IN TIME'].apply(lambda x: str(x)[11:16])
    dep_avg_out_df['Dep_Avg_OUT TIME'] = dep_avg_out_df['Dep_Avg_OUT TIME'].apply(lambda x: str(x)[11:16])
    early_login_department = dep_avg_in_df.tail(10).iloc[::-1]
    early_login_department.rename(columns={'DEPARTMENT':'Department','Dep_Avg_IN TIME':'Average In-Time'}, inplace=True)

    late_login_department = dep_avg_in_df.head(10)
    late_login_department.rename(columns={'DEPARTMENT':'Department','Dep_Avg_IN TIME':'Average In-Time'}, inplace=True)

    # Extract unique 'Department' values from early_login_department
    early_departments = early_login_department['Department'].unique()
    late_login_department = late_login_department[~late_login_department['Department'].isin(early_departments)]
    
    early_logout_department = dep_avg_out_df.head(10)
    early_logout_department.rename(columns={'DEPARTMENT':'Department','Dep_Avg_OUT TIME':'Average Out-Time'}, inplace=True)

    late_logout_department = dep_avg_out_df.tail(10).iloc[::-1]
    late_logout_department.rename(columns={'DEPARTMENT':'Department','Dep_Avg_OUT TIME':'Average Out-Time'}, inplace=True)
    
    late_departments = early_logout_department['Department'].unique()
    late_logout_department = late_logout_department[~late_logout_department['Department'].isin(late_departments)]


    ####### 
    bin_in_value = intime_df.loc[intime_df['count'].idxmax(), 'bin_in']
    bin_out_value = outtime_df.loc[outtime_df['count'].idxmax(), 'bin_out']
    
    # statement_overall = f"""
    # \n* The average logout time is **{out_df['Average Out Time'][0]}** for the past **{Header_df['Days'][0]}** working days.
    # \n* Most of the employees are loggingin in the time range of **{bin_in_value}**.
    # \n* Most of the employees are loggingout in the time range of **{bin_out_value}**.
    # \n* **{top_5['Employee Name'][0]}** from **{top_5['Department'][0]}** department has worked extra hours of **{top_5['Extra Worked Hours'][0]}** in the past **{Header_df['Days'][0]}** working days.
    # \n* **{bottom_5['Employee Name'][0]}** from **{bottom_5['Department'][0]}** department has the most compensation hours of **{bottom_5['Extra Worked Hours'][0]}** for the past **{Header_df['Days'][0]}** working days.
    # \n* **{emp_avg_time_in_head['Employee Name'][0]}** form **{emp_avg_time_in_head['Department'][0]}** department is logging in early with an average login time of **{emp_avg_time_in_head['Average In Time'][0]}** for the past **{Header_df['Days'][0]}** working days.
    # \n* **{emp_avg_time_in_tail['Employee Name'][0]}** form **{emp_avg_time_in_tail['Department'][0]}** department is logging in late with an average login time of **{emp_avg_time_in_tail['Average In Time'][0]}** for the past **{Header_df['Days'][0]}** working days.
    # \n* **{leave_df_cnt['Employee Name']}** from **{leave_df_cnt['Department']}** department has the highest leave count of **{leave_df_cnt['Leave Count']} days in this month.
    # """
    
    dep_avg_stat_dep_hi = dep_avg_df['DEPARTMENT'].iloc[0]
    dep_avg_statement_hi = dep_avg_df['Average Work Hours'].iloc[0]
    dep_avg_stat_dep_low = dep_avg_df['DEPARTMENT'].iloc[-1]
    dep_avg_statement_low = dep_avg_df['Average Work Hours'].iloc[-1]
    lev_cnt_dep_hi = leaves_count_department['DEPARTMENT'].iloc[0]
    lev_cnt_dep_hi_stat = leaves_count_department['Leaves Count'].iloc[0]
    
    # compensation_dep_hi_df = compensation_df
    # compensation_dep_hi_df['Average Compensation Hrs'] = compensation_dep_hi_df['Average Compensation Hrs'].astype('float')
    # compensation_dep_hi_df = compensation_dep_hi_df[compensation_dep_hi_df['Compensation Hrs']==compensation_dep_hi_df['Compensation Hrs'].max()]
    compensation_dep_hi = compensation_df['Department'].iloc[0]
    avg_compensation_dep_hi = compensation_df['Average Compensation Hrs'].iloc[0]
    # extra_wrk_dep_hi_df = extra_work_df
    # extra_wrk_dep_hi_df['Extra Worked Hrs'] = extra_wrk_dep_hi_df['Extra Worked Hrs'].astype('float')
    # extra_wrk_dep_hi_df = extra_wrk_dep_hi_df[extra_wrk_dep_hi_df['Extra Worked Hrs']==extra_wrk_dep_hi_df['Extra Worked Hrs'].max()]
    extra_wrk_dep_hi = extra_work_df['Department'].iloc[0]
    avg_extra_wrk_dep_hi = extra_work_df['Average Extra Worked Hrs'].iloc[0]
    

    early_in_dep_hi = early_login_department['Department'].iloc[0]
    early_in_dep_hi_stat = early_login_department['Average In-Time'].iloc[0]
    late_in_dep_hi = late_login_department['Department'].iloc[0]
    late_in_dep_stat = late_login_department['Average In-Time'].iloc[0]
    
    early_out_dep_hi = early_logout_department['Department'].iloc[0]
    early_out_dep_hi_stat = early_logout_department['Average Out-Time'].iloc[0]
    late_out_dep_hi = late_logout_department['Department'].iloc[0]
    late_out_dep_stat = late_logout_department['Average Out-Time'].iloc[0]
    
    statement_overall = f"""
    \n* The average login time is **{str(in_df['Average In Time '].iloc[0])[0:5]}** and the average logout time is **{str(out_df['Average Out Time'].iloc[0])[0:5]}** for the past **{Header_df['Working Days'].iloc[0]}** working days.
    \n* Majority of the employees are logging in during the time range of **{bin_in_value}** and logging out during the time range of **{bin_out_value}**.
    \n* **{dep_avg_stat_dep_hi}** department has the highest average work hours of **{dep_avg_statement_hi} hrs** and **{dep_avg_stat_dep_low}** department has the lowest average work hours of **{dep_avg_statement_low} hrs** when compared  to all other departments.
    \n* **{compensation_dep_hi}** department has the highest average compensation hours of **{avg_compensation_dep_hi} hrs** and **{extra_wrk_dep_hi}** department has the highest average Extra work hours of **{avg_extra_wrk_dep_hi} hrs**.
    \n* **{early_in_dep_hi}** department is the first to arrive, with an average login time of **{early_in_dep_hi_stat}** and **{late_in_dep_hi}** department is the late to arrive, with an average login time of **{late_in_dep_stat}**.
    \n* **{early_out_dep_hi}** department is the first to leave, with an average logout time of **{early_out_dep_hi_stat}** and **{late_out_dep_hi}** department is the late to leave, with an average logout time of **{late_out_dep_stat}**.
    """
    
    ## \n* **{lev_cnt_dep_hi}** department has the highest number of leaves count of **{lev_cnt_dep_hi_stat}** days when compared  to all other departments.
    
    employee_statement = f"""
    \n* **{top_5['Employee Name'].iloc[0]}** from **{top_5['Department'].iloc[0]}** department has worked extra hours of **{top_5['Extra Worked Hours'].iloc[0]}** in the past **{Header_df['Working Days'].iloc[0]}** working days.
    \n* **{bottom_5['Employee Name'].iloc[0]}** from **{bottom_5['Department'].iloc[0]}** department has the most compensation hours of **{abs(float(bottom_5['Compensation Work Hours'].iloc[0]))}** for the past **{Header_df['Working Days'].iloc[0]}** working days.
    \n* **{emp_avg_time_in_head['Employee Name'].iloc[0]}** from **{emp_avg_time_in_head['Department'].iloc[0]}** department is logging in early with an average login time of **{emp_avg_time_in_head['Average In Time'].iloc[0]}** for the past **{Header_df['Working Days'].iloc[0]}** working days.
    \n* **{emp_avg_time_in_tail['Employee Name'].iloc[0]}** from **{emp_avg_time_in_tail['Department'].iloc[0]}** department is logging in late with an average login time of **{emp_avg_time_in_tail['Average In Time'].iloc[0]}** for the past **{Header_df['Working Days'].iloc[0]}** working days.
    \n* **{leave_df_cnt['Employee Name'].iloc[0]}** from **{leave_df_cnt['Department'].iloc[0]}** department has the highest leave count of **{leave_df_cnt['Leave Count'].iloc[0]}** days in this month.
    """
    
    
    # Graph 1
    fig1 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in in_df.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[in_df[col] for col in in_df.columns]))
    ])
    fig1.update_layout(
        title='In-Time',
        height=100,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=14  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.9
    )

    # Graph 2
    fig2 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in out_df.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[out_df[col] for col in out_df.columns]))
    ])

    fig2.update_layout(
        title='Out-Time',
        height=100,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=14  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.9
    )

    # Graph 3
    # text_values_1 = grand_total_df.iloc[0, :].round(2).astype(str)
    text_values_1 = grand_total_df.values.tolist()[0]
    text_values_1 = plot_values(text_values_1)
    fig3 = px.bar(x=grand_total_df.iloc[0, :], y=['Average Worked Total','Average Grand Total','Average Difference'],
                labels={'x': 'Time in Hours', 'y': ''}, height=250, width=480,
                text=text_values_1,
                title='Total Employees Average Work',
                color=grand_total_df.columns, color_discrete_sequence=bar_colors)

    fig3.update_layout(bargap=0.3, bargroupgap=0.5, margin=dict(l=0, r=10, b=0, t=30),
                    showlegend=False,
                    paper_bgcolor=colors['dark background'], font_color=colors['text'],
                    font=dict(
                        family='Optima',
                        size=14  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)

    # Update hovertemplate to include desired information
    fig3.update_traces(hovertemplate='<b>Time in Hours</b>: %{x}')

    # Graph 4
    text_values_2 = work_hrs.values.tolist()[0]
    text_values_2 = plot_values(text_values_2)
    # text_values_2 = work_hrs.iloc[0, :].round(2).astype(str)
    # text_values_2 =["9.30","3.54","58.2"]
    fig4 = px.bar(x=work_hrs.iloc[0, :], y=['Average Worked Hrs','Actual Work Hrs','Pending Hrs'],
                labels={'x': 'Time in Hours', 'y': ''}, height=250, width=480,
                text=text_values_2,color=grand_total_df.columns, color_discrete_sequence=bar_colors)
    fig4.update_layout(bargap=0.3, bargroupgap=0.5, title='Per Day Average Work', margin=dict(l=0, r=10, b=0, t=30),
                    showlegend=False,
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['text'],
                    font=dict(
                        family='Optima',
                        size=14  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)
    fig4.update_traces(hovertemplate='<b>Time in Hours</b>: %{x}')
    # Graph 5
    text_values_3 = Total_work.values.tolist()[0]
    text_values_3 = plot_values(text_values_3)

    fig5 = px.bar(x=Total_work.iloc[0, :], y=['Total Worked Hrs','Actual Work Hrs','Pending Hrs'],
                labels={'x': 'Time in Hours', 'y': ''}, height=250, width=480,
                text=text_values_3, color=Total_work.columns, color_discrete_sequence=bar_colors)

    # Update layout
    fig5.update_layout(bargap=0.3, bargroupgap=0.5, title='Total Work',
                    showlegend=False,  # Set showlegend to False
                    margin=dict(l=0, r=10, b=0, t=30),
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['text'],
                    font=dict(
                        family='Optima',
                        size=14  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)
    fig5.update_traces(hovertemplate='<b>Time in Hours</b>: %{x}')
    # Graph 6
    fig6 = go.Figure(data=go.Scatter(x = grouped_data_IN["Day"],
                                    y = grouped_data_IN["IN_TIME_DATETIME"],
                                    mode='lines+markers+text',
                                    text=grouped_data_IN["IN_TIME_DATETIME"].apply(lambda x: str(x)[11:16]),
                                    textposition='bottom center'),
                    
                    layout=go.Layout(height=290, margin=dict(b=20, t=50, l=50, r=50)))
    # fig6 = px.line(grouped_data_IN, x="Day", y='IN_TIME_DATETIME', markers=True, line_shape='linear', height=300)

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
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=14  # Adjust the font size as needed
        ),
    )

    # Graph 7

    fig7 = go.Figure(data=go.Scatter(
        x=grouped_data_OUT["Day"],
        y=grouped_data_OUT["OUT_TIME_DATETIME"],
        mode='lines+markers+text',
        text=grouped_data_OUT["OUT_TIME_DATETIME"].apply(lambda x: str(x)[11:16]),
        textposition='bottom center'
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
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=14  # Adjust the font size as needed
        ),
    )

    # Graph 8
    fig8 = px.line(grouped_data_date_in, x='DATE', y='TIME_DATETIME_IN', markers=True, line_shape='linear', height=350)

    # Customize the layout
    fig8.update_layout(
        title='Date Wise Average In-Time',
        xaxis_title='Date',
        yaxis_title='Time',
        # xaxis=dict(tickangle=-45),
        yaxis=dict(
            tickformat="%H:%M:%S",
        ),
        title_x=0.5, title_y=0.98,
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=14  # Adjust the font size as needed
        ),
    )

    # Graph 9
    fig9 = px.line(grouped_data_date_out, x='DATE', y='TIME_DATETIME_OUT', markers=True, line_shape='linear', height=300)

    # Customize the layout
    fig9.update_layout(
        title='Date Wise Average Out-Time',
        xaxis_title='Date',
        yaxis_title='Time',
        # xaxis=dict(tickangle=-45),
        yaxis=dict(
            tickformat="%H:%M:%S",
            
        ),
        title_x=0.5, title_y=0.98,
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=14  # Adjust the font size as needed
        ),
    )
    # Graph 10 
    fig10 = px.bar(intime_df, x='bin_in', y='count',
                height=300, width=728,
                text='count',
                color_discrete_sequence=['#3498db'])  # Display the count on top of each bar

    fig10.update_layout(bargap=0.3, bargroupgap=0.5, title='Employees In-Time Count',
                    margin=dict(l=0, r=10, b=0, t=30),
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['text'],
                    xaxis_title='Time Range',
                    yaxis_title='Count',
                    font=dict(
                        family='Optima',
                        size=14  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)

    # fig10.update_traces(hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')
    fig10.update_traces(textangle=0, hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')
    

    # Graph 11
    fig11 = px.bar(outtime_df, x='bin_out', y='count',
                height=300, width=728,
                text='count',
                color_discrete_sequence=['#3498db'])  # Display the count on top of each bar

    fig11.update_layout(bargap=0.3, bargroupgap=0.5, title='Employees Out-Time Count',
                    margin=dict(l=0, r=10, b=0, t=30),
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['text'],
                    xaxis_title='Time Range',
                    yaxis_title='Count',
                    font=dict(
                        family='Optima',
                        size=14  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)

    # fig11.update_traces(hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')
    fig11.update_traces(textangle=0, hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')
    

    # Graph 12
    fig12 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in top_5.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[top_5[col] for col in top_5.columns]))
    ])
    fig12.update_layout(
        title='Top 10 Highest Extra Worked Employees',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.98
    )

    # Graph 13
    fig13 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in bottom_5.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[bottom_5[col] for col in bottom_5.columns]))
    ])
    fig13.update_layout(
        title='Top 10 Highest Compensation Hrs Employees',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.98
    )
    # Graph 14
    fig14 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in emp_avg_time_in_head.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[emp_avg_time_in_head[col] for col in emp_avg_time_in_head.columns])
    )])
    fig14.update_layout(
        title='Top 10 Consistent Early Login Employees',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5,
        title_y=0.98
    )
    # Graph 15
    fig15 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in emp_avg_time_in_tail.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[emp_avg_time_in_tail[col] for col in emp_avg_time_in_tail.columns])
    )])
    fig15.update_layout(
        title='Top 10 Consistent Late Login Employees',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5,
        title_y=0.98
    )
    
    # Graph 14_1
    fig14_1 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in emp_avg_time_out_head.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[emp_avg_time_out_head[col] for col in emp_avg_time_out_head.columns])
    )])
    fig14_1.update_layout(
        title='Top 10 Consistent Early Logout Employees',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5,
        title_y=0.98
    )
    # Graph 15_1
    fig15_1 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in emp_avg_time_out_tail.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[emp_avg_time_out_tail[col] for col in emp_avg_time_out_tail.columns])
    )])
    fig15_1.update_layout(
        title='Top 10 Consistent Late Logout Employees',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5,
        title_y=0.98
    )
    
    # Graph 16
    Header_df['Work Hrs Per Emp'] = Header_df['Work Hrs Per Emp'].apply(card_values)
    fig16 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in Header_df.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[Header_df.iloc[0][col] for col in Header_df.columns]))
    ])
    fig16.update_layout(
        # title='In Time',
        height=80,
        margin=dict(l=5, r=5, b=0, t=5),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=14  # Adjust the font size as needed
        ),
    #     title_x=0.5, title_y=0.9
    )
    
    # Graph 17
    fig17 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in leave_df_cnt.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[leave_df_cnt[col] for col in leave_df_cnt.columns])
    )])
    fig17.update_layout(
        title='Top 10 Employees with Highest Leave Counts',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5,
        title_y=0.98
    )
    # Graph 17_1
    fig17_1 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in dep_avg_df_count.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[dep_avg_df_count[col] for col in dep_avg_df_count.columns])
    )])
    fig17_1.update_layout(
        title='Department Wise Employee Count',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5,
        title_y=0.98
    )
    
    # Graph 18
    fig18 = px.bar(dep_avg_df, x='DEPARTMENT', y='Average Work Hours',
                height=300, width=728,  # 728
                text='Average Work Hours',
                color_discrete_sequence=['#3498db'])  # Display the count on top of each bar

    fig18.update_layout(bargap=0.3, bargroupgap=0, title='Department Wise Average Worked Hours',
                    margin=dict(l=0, r=10, b=0, t=30),
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['text'],
                    xaxis_title='Department',
                    yaxis_title='Average Work Hrs',
                    font=dict(
                        family='Optima',
                        size=14  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)
    fig18.update_yaxes(range=[7, dep_avg_df['Average Work Hours'].max()+0.5])  # Set the range of y-axis

    # fig10.update_traces(hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')
    fig18.update_traces(textangle=0, hovertemplate='<b>Department</b>: %{x}<br><b>Average Work Hours</b>: %{y}')




    # # Graph 19
    # fig19 = px.pie(top_7_departments, values='Leaves Count', names='DEPARTMENT',
    #                             title='Department Wise Leaves Count',
    #                             width=728, height=300,
    #                             hole=0.3  # Adjust the hole size if needed
    #                             )

    # # Update layout
    # fig19.update_layout(
    #     margin=dict(l=0, r=10, b=0, t=30),
    #     paper_bgcolor=colors['dark background'],
    #     font_color=colors['text'],
    #     font=dict(
    #         family='Optima',
    #         size=14  # Adjust the font size as needed
    #     ),
    #     title_x=0.5, title_y=0.98,
    #     piecolorway=px.colors.sequential.Viridis  # Set the colorway to Viridis
    # )

    # # Add leaves count inside the pie chart slices without percentage
    # fig19.update_traces(textinfo='value')
    
    # Graph 19
    fig19 = px.bar(leaves_count_department, x='DEPARTMENT', y='Leaves Count',
                height=300, width=728,
                text='Leaves Count',
                color_discrete_sequence=['#3498db'])  # Display the count on top of each bar

    fig19.update_layout(bargap=0.3, bargroupgap=0, title='Department Wise Leaves Count',
                    margin=dict(l=0, r=10, b=0, t=30),
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['text'],
                    xaxis_title='Department',
                    yaxis_title='Leaves Count',
                    font=dict(
                        family='Optima',
                        size=14  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)

    # fig19.update_traces(hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')
    fig19.update_traces(textangle=0, hovertemplate='<b>Department</b>: %{x}<br><b>Leaves Count</b>: %{y}')
    
    # Graph 19_1
    fig19_1 = px.bar(dep_avg_df_count, x='DEPARTMENT', y='Employee Count',
                height=300, width=480,
                text='Employee Count',
                color_discrete_sequence=['#3498db'])  # Display the count on top of each bar

    fig19_1.update_layout(bargap=0.3, bargroupgap=0, title='Department Wise Employee Count',
                    margin=dict(l=0, r=10, b=0, t=30),
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['text'],
                    xaxis_title='Department',
                    yaxis_title='Employee Count',
                    font=dict(
                        family='Optima',
                        size=14  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98)

    # fig19.update_traces(hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')
    fig19_1.update_traces(textangle=0, hovertemplate='<b>Department</b>: %{x}<br><b>Employee Count</b>: %{y}')
    
    
    fig_19_1_line = px.line(dep_avg_df_count,x="DEPARTMENT", y='Employee Count', markers=True, line_shape='linear', height=300,text='Employee Count',color_discrete_sequence=['#3498db'])
    fig_19_1_line.update_traces(textposition='top right',
                                textfont=dict(size=10))
    fig_19_1_line.update_layout(title='Department Wise Employee Count',
                                margin=dict(l=0, r=10, b=0, t=5),
                                paper_bgcolor=colors['dark background'],
                                font_color=colors['text'],
                                xaxis_title='Department',
                                yaxis_title='Employee Count',
                                font=dict(
                                    family='Optima',
                                    size=14  # Adjust the font size as needed
                                ),
                                title_x=0.5, title_y=0.98)
    # Graph 20
    fig20 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in compensation_df.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[compensation_df[col] for col in compensation_df.columns]))
    ])
    fig20.update_layout(
        title='Departments with the Most Compensation Hours',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.98
    )


    # Graph 21
    fig21 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in extra_work_df.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[extra_work_df[col] for col in extra_work_df.columns]))
    ])
    fig21.update_layout(
        title='Departments with Extra Worked Hours',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.98
    )

    # Graph 22
    fig22 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in late_login_department.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[late_login_department[col] for col in late_login_department.columns]))
    ])
    fig22.update_layout(
        title='Top 10 Late Login Departments',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.98
    )

    # Graph 23
    fig23 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in early_logout_department.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[early_logout_department[col] for col in early_logout_department.columns]))
    ])
    fig23.update_layout(
        title='Top 10 Early Logout Departments',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.98
    )
    
    # Graph 24 
    fig24 = px.bar(total_work_df, x='DEPARTMENT', y=['Actual Work Hrs','Total Work Hrs'],
                height=300, width=980,
                color_discrete_map={'Actual Work Hrs': '#3498db', 'Total Work Hrs': 'red'}
                )

    fig24.update_layout(barmode = 'group',
                    bargap=0.2, bargroupgap=0.05, title='Department Wise Total Worked Hours',
                    margin=dict(l=0, r=10, b=0, t=30),
                    paper_bgcolor=colors['dark background'],
                    font_color=colors['text'],
                    xaxis_title='Department',
                    yaxis_title='Total Work in Hrs',
                    font=dict(
                        family='Optima',
                        size=14  # Adjust the font size as needed
                    ),
                    title_x=0.5, title_y=0.98,
                    legend=dict(x=0.95, y=0.95, xanchor='right', yanchor='top', orientation='v', title='')
                    )
    fig24.update_traces(texttemplate='%{y:.0f}', textposition='outside') # Adjust texttemplate as per your need

    # fig24.update_traces(hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')
    fig24.update_traces(textangle=0, hovertemplate='<b>Department</b>: %{x}<br><b>Work Hrs</b>: %{y}')
    
    
    # Graph 25
    fig25 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in early_login_department.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[early_login_department[col] for col in early_login_department.columns]))
    ])
    fig25.update_layout(
        title='Top 10 Early Login Departments',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.98
    )

    # Graph 26
    fig26 = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in late_logout_department.columns],  # Making column names bold
            line_color='#7a7a7a',  # Specify line color for the header
            fill_color='lightblue',  # Specify fill color for the header
            align='center',  # Align the text to the left
        ),
        cells=dict(values=[late_logout_department[col] for col in late_logout_department.columns]))
    ])
    fig26.update_layout(
        title='Top 10 Late Logout Departments',
        height=265,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor=colors['dark background'],
        font_color=colors['text'],
        font=dict(
            family='Optima',
            size=12  # Adjust the font size as needed
        ),
        title_x=0.5, title_y=0.98
    )
    
    # Component 1 
    Header_component = html.H1("Time Sheet Analysis", style={'textAlign': 'center', 'color': 'black', 'marginBottom': '20px',"font-family": "Optima", "font-size": "44px","font-weight": "bold"})
    
    department_component = html.H1("Department Analysis", style={'textAlign': 'center', 'color': 'black', 'marginBottom': '20px',"font-family": "Optima", "font-size": "44px","font-weight": "bold"})
    
    # Design the app Layout

    # Define a slightly darker shade of blue
    colors['dark background'] = '#1b4965'

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
                    dcc.Graph(figure=fig16, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=4,  # Set the width for the graph column
                    className='mb-4'
                )
            ]),
            dbc.Row([
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Overview", className="card-title",style={"font-family": "Optima", "font-size": "22px", "font-weight": "bold","color": colors['text_card']}),
                                    dcc.Markdown(statement_overall),
                                ]
                            ),
                            style={"background-color": colors['banner_color'],"font-family": "Optima", "font-size": "18px","width": "98%","margin": "0 auto","color":colors['text_card']},  #color=colors['dark background'], 
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
                    width=4,
                    className='mb-4'
                ),
                dbc.Col(
                    dcc.Graph(figure=fig5, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=4,
                    className='mb-4'
                ),
                dbc.Col(
                    dcc.Graph(figure=fig3, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=4,
                    className='mb-4'
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
                dbc.Col(
                    dcc.Graph(figure=fig12, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                ),
                dbc.Col(
                    dcc.Graph(figure=fig13, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                )
            ]),  # Add margin at the bottom for the last row
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig14, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                ),
                dbc.Col(
                    dcc.Graph(figure=fig15, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                )
            ]),
            html.Div(style={'height': '100px'}), 
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig14_1, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                ),
                dbc.Col(
                    dcc.Graph(figure=fig15_1, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                )
            ]),
            dbc.Row([
                # dbc.Col(
                #     dcc.Graph(figure=fig_19_1_line, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                #     width=6,
                #     className='mb-4'
                # ),
                dbc.Col(
                    dcc.Graph(figure=fig17, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                ),
                # dbc.Col(
                #     dbc.Card(
                #         dbc.CardBody(
                #             [
                #                 html.H4("Leave Count", className="card-title",style={"font-family": "Optima", "font-size": "20px", "font-weight": "bold"}),
                #                 dcc.Markdown(leave_statement)
                #             ]
                #         ),

                #         style={"background-color": colors['background'],"font-family": "Optima", "font-size": "18px"},
                #         className='mb-4',
                #     ),
                # width=6,
                # )
            ]),
            html.Div(style={'height': '100px'}), 
            dbc.Row([
                dbc.Col(
                    department_component,
                    width=9,  # Set the width for the header column
                    className='mb-4 mx-auto text-center'
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig19_1, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=4,
                    className='mb-4'
                ),
                dbc.Col(
                    dcc.Graph(figure=fig24, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=8,
                    className='mb-4'
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig18, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                ),
                dbc.Col(
                    dcc.Graph(figure=fig19, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                )
            ]),
            
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig20, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                ),
                dbc.Col(
                    dcc.Graph(figure=fig21, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                )
            ]),
            # html.Div(style={'height': '100px'}), 
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig25, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                ),
                dbc.Col(
                    dcc.Graph(figure=fig22, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                )
            ]),
            html.Div(style={'height': '150px'}), 
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig23, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                ),
                dbc.Col(
                    dcc.Graph(figure=fig26, style={'backgroundColor': colors['background'], 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)','border-radius': '15px',"overflow": "hidden","border": "none"}),
                    width=6,
                    className='mb-4'
                )
            ]),
            dbc.Row([
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Employee Report :", className="card-title",style={"font-family": "Optima", "font-size": "22px", "font-weight": "bold","color": colors['text_card']}),
                                dcc.Markdown(employee_statement),
                            ]
                        ),
                        style={"background-color": colors['banner_color'],"font-family": "Optima", "font-size": "18px","width": "98%","margin": "0 auto","color":colors['text_card']},  #color=colors['dark background'], 
                        className='mb-4',
                    )
            ]),
        ],
        style={
            'backgroundColor': 'white',
            'padding': '20px'  # Add padding to the entire layout
        }
    )
    app.run_server(debug=False, port=8050)

def run_selenium(emp_df,month):
    sent_list = []
    fail_list = []
    try:
        print("Opening Chrome...")
        chrome_options = webdriver.ChromeOptions()
        prefs = {
            "profile.default_content_setting_values.notifications": 2,
            "download.default_directory": r"D:\Susmith Tasks\TImesheet_API\Overall_timesheet\Output_files"  # Specify your desired download path
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # chrome_path = r"C:\Users\Rahul\Desktop\Onedrive-sharepoint-Login_Time\chromedriver-win64\chromedriver.exe"
        # chrome_path = r"C:\Users\Rahul\Desktop\Fastapi\chromedriver-win64_02_05_24\chromedriver-win64\chromedriver.exe"
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), chrome_options=chrome_options)
        wait = WebDriverWait(driver, 10)

        url = "http://localhost:8050/"
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
        file_path = fr"C:\Users\Rahul\Desktop\Onedrive-sharepoint-Login_Time\dash_fastapi\Employee Report\Timesheet_report_{month}.pdf"
        with open(file_path, "wb") as fh:
            fh.write(pdf_bytes)
        # Close the browser
        driver.quit()
        for index, row in tqdm(emp_df.iterrows()):
            employee = row['Employee Name']
            # email_sender = "technocrat3128@gmail.com"
            # email_password = "nhbn dtvr rsbe dtgu"
            email_sender = "corporate.affairs@piloggroup.org"
            email_password = "ymcb udvb nury ukcl"
            recipient_mail = row['Employee Mail']
            subject = f"Time Sheet for the month of {month}"

            body = f"""
            <p>Kindly review timesheet report for the month of {month}.</p>
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
                </div>

            </body>
            </html>
            """
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)

            # Attach a PDF file
            pdf_file_path = file_path
            pdf_file_name = f"Time Sheet for {month}.pdf"

            with open(pdf_file_path, "rb") as pdf_attachment:
                pdf_part = MIMEBase("application", "octet-stream")
                pdf_part.set_payload(pdf_attachment.read())

            encoders.encode_base64(pdf_part)
            pdf_part.add_header("Content-Disposition", f"attachment; filename= {pdf_file_name}")
            message.attach(pdf_part)

            # Create a connection and send the emails
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


# def run_app():
#     app.layout = html.Div("Hello Dash!")
#     app.run_server(debug=False, port=8050)

# def test_run(df):
#     dash_thread = threading.Thread(target=run_app)
#     dash_thread.start()
#     time.sleep(5)
#     path_1 = dash_app_selenium(df)
#     dash_thread.join()
#     return path_1

def test_run(df,emp_df,month):
    dash_process = multiprocessing.Process(target=dash_app_selenium, args=(df,))
    dash_process.start()
    
    path_1_process = multiprocessing.Process(target=run_selenium, args=(emp_df,month,))
    path_1_process.start()
    path_1_process.join()

    time.sleep(5)
    path_1_process.terminate()

    path_1_process.join()
    dash_process.terminate()   # added remove if needed
    
    return {"message":"success"}

