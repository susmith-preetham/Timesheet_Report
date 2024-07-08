import warnings
from datetime import datetime
import pandas as pd
import numpy as np
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

def twentyfour_hrs_frmt(value):
    value = str(value)
    if value.startswith('24'):
        value = value.replace('24','00')
        
    return(float(value))


def create_dataframe(dataframe):
    dataframe["Grand Total Actual"] = np.nan
    dataframe["Grand Total Final"] = np.nan
    dataframe["Grand Total Grace Period"] = np.nan
    dataframe["Grand Total Worked"] = np.nan   
    dataframe.rename(columns=lambda x: x.strip() if isinstance(x, str) else x, inplace=True) # added remove if needed
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
    final_df["IN TIME"] = final_df["IN TIME"].apply(twentyfour_hrs_frmt) # added
    
    final_df["OUT TIME"] = final_df["OUT TIME"].apply(lambda x: np.nan if x == 'L' else x)
    final_df["OUT TIME"] = final_df["OUT TIME"].apply(twentyfour_hrs_frmt) # added
    
    final_df["DATE"] = pd.to_datetime(final_df["DATE"])
    final_df["IN TIME"] = final_df["IN TIME"].astype(float)
    final_df["OUT TIME"] = final_df["OUT TIME"].astype(float)
    final_df["IN TIME"] = final_df["IN TIME"].apply(half_day_in)
    final_df["OUT TIME"] = final_df["OUT TIME"].apply(half_day_out)
    final_df["TOTAL"] = final_df["TOTAL"].astype(float)
    final_df["TOTAL"] = final_df["TOTAL"].apply(lambda x : 0 if x<0 else x)
    # final_df["TOTAL"] = final_df["TOTAL"].apply(lambda x : adjust_decimal_values(x))
    final_df = final_df.dropna(thresh=0.85*len(final_df.columns))
    final_df["TOTAL"] = final_df.apply(lambda x : calculate_time_difference(x['IN TIME'],x['OUT TIME']),axis = 1)
    # final_df = final_df.dropna(thresh=0.85*len(final_df.columns))
    

    # Merge with the original 'Grand Total' columns
    final_df = final_df.merge(dataframe[['EMPLOYEE NAME', 'Grand Total Worked', 'Grand Total Grace Period', 'Grand Total Actual', 'Grand Total Final']], on='EMPLOYEE NAME', how='left')
    
    final_df["Grand Total Worked"] = final_df.groupby(["EMPLOYEE NAME"])["TOTAL"].transform('sum')
    
    # final_df["Grand Total Actual"] = len(final_df["DATE"].unique())*9.5  
    final_df["Grand Total Actual"] = 0
    
    # final_df["Grand Total Final"] = final_df["Grand Total Actual"] - final_df["Grand Total Worked"] 
    final_df["Grand Total Final"] = 0
    return final_df

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

def apply_all_functions(dataframe):
    dataframe.rename(columns=lambda x: x.strip() if isinstance(x, str) else x, inplace=True)
    dataframe = removing_unnecessary_values(dataframe)
    dataframe = columns_name_change(dataframe)
    dataframe = create_dataframe(dataframe)
    return dataframe



def emp_table_data(df):
    df = apply_all_functions(df)
    emp_data_df = df[['EMPLOYEE NAME',
    'DATE',
    'IN TIME',
    'OUT TIME',
    'TOTAL']] 
    emp_data_df['IN TIME'] = emp_data_df['IN TIME'].apply(lambda x : float_to_string(x))
    emp_data_df['OUT TIME'] = emp_data_df['OUT TIME'].apply(lambda x : float_to_string(x))
    emp_data_df['TOTAL'] = emp_data_df['TOTAL'].apply(lambda x : hrs_to_min(x))
    emp_total_df = df[['EMPLOYEE NAME',
    'Grand Total Worked',
    'Grand Total Actual',
    'Grand Total Final']]
    # emp_total_df['Grand Total Final'] = emp_total_df['Grand Total Final'].apply(lambda x : hrs_to_min(x))
    # emp_total_df['Grand Total Worked'] = emp_total_df['Grand Total Worked'].apply(lambda x : hrs_to_min(x))
    # emp_total_df['Grand Total Actual'] = emp_total_df['Grand Total Actual'].apply(lambda x : hrs_to_min(x))
    emp_total_df.columns = ['EMPLOYEE NAME','TOTAL WORKED HOURS','ACTUAL WORK HOURS','COMPENSATION HOURS']
    
    return emp_data_df,emp_total_df

