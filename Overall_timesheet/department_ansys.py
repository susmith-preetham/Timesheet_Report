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


dataframe['DEPARTMENT'] = dataframe['DEPARTMENT'].apply(lambda x : x.replace('-','_'))

leave_df['DEPARTMENT'] = leave_df['DEPARTMENT'].apply(lambda x : x.replace('-','_'))

department_df = final_df[['EMPLOYEE NAME','DEPARTMENT','Grand Total Worked','Grand Total Actual_real','Grand Total Final_real']]
department_df.drop_duplicates(inplace = True)

total_work_df = department_df
total_work_df['Total Work Hrs'] = department_df.groupby(['DEPARTMENT'])['Grand Total Worked'].transform('sum')
total_work_df['Actual Work Hrs'] = department_df.groupby(['DEPARTMENT'])['Grand Total Actual_real'].transform('sum')
total_work_df.sort_values(by = 'Total Work Hrs',ascending=False,inplace=True)
total_work_df = total_work_df[['DEPARTMENT','Total Work Hrs','Actual Work Hrs']]
total_work_df.drop_duplicates(inplace=True)


dep_avg_df = final_df.groupby(['EMPLOYEE NAME','DEPARTMENT'])['TOTAL_HLF'].mean().reset_index(name = 'Average Work Hours')
dep_avg_df['Average Work Hours'] = dep_avg_df.groupby('DEPARTMENT')['Average Work Hours'].transform(average_time)
dep_avg_df['Average Work Hours'] = dep_avg_df['Average Work Hours'].apply(lambda x : float(str(x).replace(':', '.')))
dep_avg_df = dep_avg_df[['DEPARTMENT','Average Work Hours']].drop_duplicates()
dep_avg_df.sort_values(by = 'Average Work Hours', ascending = False, inplace = True)


leave_df = leave_df[['EMPLOYEE NAME','DEPARTMENT','LEAVES']]
leave_df['LEAVES'] = leave_df['LEAVES'].apply(lambda x: 1 if x == 'LEAVE' else 0)
leaves_count_department = leave_df.groupby(['DEPARTMENT'])['LEAVES'].sum().reset_index(name = 'Leaves Count')
leaves_count_department.sort_values(by = 'Leaves Count', ascending=False, inplace=True)
leaves_count_department = leaves_count_department[leaves_count_department['Leaves Count'] != 0]
# leaves_count_department_sorted = leaves_count_department.sort_values(by='Leaves Count', ascending=False)
# top_7_departments = leaves_count_department_sorted.head(10)

Compensation_hrs_department = department_df
Compensation_hrs_department['Compensation Hrs'] = Compensation_hrs_department.groupby(['DEPARTMENT'])['Grand Total Final_real'].transform('sum')
Compensation_hrs_department['Average Compensation Hrs'] = Compensation_hrs_department.groupby(['DEPARTMENT'])['Grand Total Final_real'].transform('mean')
Compensation_hrs_department.sort_values(by='Compensation Hrs', ascending=False, inplace=True)
Compensation_hrs_department = Compensation_hrs_department[['DEPARTMENT','Compensation Hrs','Average Compensation Hrs']]
Compensation_hrs_department.drop_duplicates(inplace=True)
Compensation_hrs_department['Compensation Hrs'] = Compensation_hrs_department['Compensation Hrs'].apply(lambda x : str(round(x,2)))
Compensation_hrs_department['Average Compensation Hrs'] = Compensation_hrs_department['Average Compensation Hrs'].apply(lambda x : str(round(x,2)))
compensation_df = Compensation_hrs_department.head(10)
extra_work_df = Compensation_hrs_department.tail(10)
extra_work_df['Compensation Hrs'] = extra_work_df['Compensation Hrs'].apply(lambda x : str(abs(round(float(x),2))) if float(x)<0 else np.nan)
extra_work_df['Average Compensation Hrs'] = extra_work_df['Average Compensation Hrs'].apply(lambda x : str(abs(round(float(x),2))) if float(x)<0 else np.nan)
extra_work_df.sort_values(by = 'Compensation Hrs',ascending = True,inplace = True)
extra_work_df.dropna(inplace = True)
extra_work_df.rename(columns = {'Compensation Hrs':'Extra Worked Hrs'},inplace = True)


final_df['Emp_Avg_IN TIME'] = final_df.groupby(['EMPLOYEE NAME'])['IN TIME_AVG'].transform('mean')
final_df['Emp_Avg_OUT TIME'] = final_df.groupby(['EMPLOYEE NAME'])['OUT TIME_AVG'].transform('mean')
final_df['Dep_Avg_IN TIME'] = final_df.groupby(['DEPARTMENT'])['Emp_Avg_IN TIME'].transform(average_time)
final_df['Dep_Avg_OUT TIME'] = final_df.groupby(['DEPARTMENT'])['Emp_Avg_OUT TIME'].transform(average_time)
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

early_logout_department = dep_avg_out_df.head(10)
early_logout_department.rename(columns={'DEPARTMENT':'Department','Dep_Avg_OUT TIME':'Average Out-Time'}, inplace=True)

late_logout_department = dep_avg_out_df.tail(10).iloc[::-1]
late_logout_department.rename(columns={'DEPARTMENT':'Department','Dep_Avg_OUT TIME':'Average Out-Time'}, inplace=True)


# Graph 24 
fig24 = px.bar(total_work_df, x='DEPARTMENT', y=['Actual Work Hrs','Total Work Hrs'],
            height=400, width=1400,
            # text='DEPARTMENT',
            # color_discrete_sequence=['#3498db'],
            )

fig24.update_layout(barmode = 'group',
                bargap=0.3, bargroupgap=0.1, title='Department Wise Total Worked Hours',
                margin=dict(l=0, r=10, b=0, t=30),
                paper_bgcolor=colors['dark background'],
                font_color=colors['text'],
                xaxis_title='Department',
                yaxis_title='Total Work in Hrs',
                font=dict(
                    family='Optima',
                    size=14  # Adjust the font size as needed
                ),
                title_x=0.5, title_y=0.98)

# fig24.update_traces(hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')
fig24.update_traces(textangle=0, hovertemplate='<b>Department</b>: %{x}<br><b>Work Hrs</b>: %{y}')


# Graph 18
fig18 = px.bar(dep_avg_df, x='DEPARTMENT', y='Average Work Hours',
            height=300, width=728,  # 728
            text='Average Work Hours',
            color_discrete_sequence=['#3498db'])  # Display the count on top of each bar

fig18.update_layout(bargap=0.3, bargroupgap=0.5, title='Department Wise Average Worked Hours',
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

# fig19.update_traces(hovertemplate='<b>Time Range</b>: %{x}<br><b>Count</b>: %{y}')
fig18.update_traces(textangle=0, hovertemplate='<b>Department</b>: %{x}<br><b>Average Work Hours</b>: %{y}')




# # Graph 19
# fig19 = px.pie(top_7_departments, values='Leaves Count', names='DEPARTMENT',
#                                title='Top 10 Departments By Leaves Count',
#                                width=728, height=400,
#                                hole=0.3  # Adjust the hole size if needed
#                                )

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

fig19.update_layout(bargap=0.3, bargroupgap=0.5, title='Department Wise Leaves Count',
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
    height=260,
    margin=dict(l=0, r=0, b=0, t=30),
    paper_bgcolor=colors['dark background'],
    font_color=colors['text'],
    font=dict(
        family='Optima',
        size=13.2  # Adjust the font size as needed
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
    title='Departments with the Most Extra Work Hours',
    height=260,
    margin=dict(l=0, r=0, b=0, t=30),
    paper_bgcolor=colors['dark background'],
    font_color=colors['text'],
    font=dict(
        family='Optima',
        size=13.2  # Adjust the font size as needed
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
    height=260,
    margin=dict(l=0, r=0, b=0, t=30),
    paper_bgcolor=colors['dark background'],
    font_color=colors['text'],
    font=dict(
        family='Optima',
        size=13.2  # Adjust the font size as needed
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
    height=260,
    margin=dict(l=0, r=0, b=0, t=30),
    paper_bgcolor=colors['dark background'],
    font_color=colors['text'],
    font=dict(
        family='Optima',
        size=13.2  # Adjust the font size as needed
    ),
    title_x=0.5, title_y=0.98
)

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
    height=260,
    margin=dict(l=0, r=0, b=0, t=30),
    paper_bgcolor=colors['dark background'],
    font_color=colors['text'],
    font=dict(
        family='Optima',
        size=13.2  # Adjust the font size as needed
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
    height=260,
    margin=dict(l=0, r=0, b=0, t=30),
    paper_bgcolor=colors['dark background'],
    font_color=colors['text'],
    font=dict(
        family='Optima',
        size=13.2  # Adjust the font size as needed
    ),
    title_x=0.5, title_y=0.98
)