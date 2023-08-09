#!/usr/bin/env python
# coding: utf-8

# In[5]:


#1.MAKE YOUR DFS
import pandas as pd

#MAKE A DF FOR RAIN DATA
rain =pd.read_csv(r'C:\Users\jonat\OneDrive\Documents\Diss\Rainfall.csv',sep=',', on_bad_lines='skip')
rain.rename(columns={' mean_FLDAS_NOAH001_G_CA_D_001_Rainf_tavg': 'rain'}, inplace=True)
rain['time'] = pd.to_datetime(rain['time'])
rain['rain'] = pd.to_numeric(rain['rain'], errors='coerce')
rain['year'] = rain['time'].dt.year
rain['month'] = rain['time'].dt.month

# Subset the data for years 2014-2021
rain_19_23 = rain[(rain['year'] >= 2019)]


#MAKE A DF FOR NDWI
df1 = pd.read_csv(r'C:\Users\jonat\OneDrive\Documents\Diss\NDWI_S2_1.csv',sep=',', on_bad_lines='skip')
df2 =pd.read_csv(r'C:\Users\jonat\OneDrive\Documents\Diss\NDWI_S2_2.csv',sep=',', on_bad_lines='skip')
df1.rename(columns={'system:time_start':'Date'}, inplace=True)
df2.rename(columns={'system:time_start':'Date'}, inplace=True)

NDWI = pd.merge(df1, df2, on='Date', how='inner')

#DO THE SAME FOR AMWI
df3 = pd.read_csv(r'C:\Users\jonat\OneDrive\Documents\Diss\AMWI_S2_1.csv',sep=',', on_bad_lines='skip')
df4 =pd.read_csv(r'C:\Users\jonat\OneDrive\Documents\Diss\AMWI_S2_2.csv',sep=',', on_bad_lines='skip')
df3.rename(columns={'system:time_start':'Date'}, inplace=True)
df4.rename(columns={'system:time_start':'Date'}, inplace=True)

AMWI = pd.merge(df3, df4, on='Date', how='inner')
AMWI


# In[6]:


#2 MAKE ALL ROWS UNIQUE 
import numpy as np

# Custom aggregation function to combine values and discard NaNs
def combine_values(series):
    combined = series.dropna().values
    if len(combined) > 0:
        return combined[0]
    return np.nan

# Group by 'Date' and apply the custom aggregation function to each column
AMWI_unique = AMWI.groupby('Date').agg(combine_values).reset_index()

NDWI_unique = NDWI.groupby('Date').agg(combine_values).reset_index()


# In[7]:


# 3. SET ALL NON-WATERY BODIES, AND BODIES WITH NEGATIVE AMWI, TO 0 CONTAMINATION
amwi_columns = AMWI_unique.columns[1:]
AMWI_unique['Date'] = pd.to_datetime(AMWI_unique['Date'])
NDWI_unique['Date'] = pd.to_datetime(NDWI_unique['Date'])

AMWI_unique = AMWI_unique.sort_values('Date')
NDWI_unique = NDWI_unique.sort_values('Date')


# Iterate over each column in AMWI and update contamination values to 0 where water value < 0
for column in amwi_columns:
    AMWI_unique.loc[NDWI_unique[column] < 0, column] = 0
    AMWI_unique.loc[AMWI_unique[column] < 0, column] = 0
AMWI_unique.set_index('Date', inplace=True)

# Print the updated AMWI dataframe
AMWI_unique.head(100)
#NOW ALL NON WATERY BODIES ARE SET TO 0 CONTAMINATION 


# In[10]:


#AVERAGEAMWIPLOT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Select a random sample of 10 columns from df_other
sample_columns = AMWI_unique.sample(n=5, axis=1)
sample_columns = sample_columns.interpolate(method='linear')

# Calculate the average and standard deviation for the sample columns
average = AMWI_unique.mean(axis=1)
std_dev = AMWI_unique.std(axis=1)

# Create the figure and axes
fig, ax1 = plt.subplots()

# Plot the AMWI data on the first y-axis
line1 = ax1.plot(AMWI_unique.index, average, label='Average AMWI - S2', linewidth=2, marker='o', color='black')
# Set the y-axis label for AMWI
ax1.set_ylabel('AMWI - Contamination Levels')

# Create a twin axes sharing the x-axis
ax2 = ax1.twinx()

# Filter the rainfall data for April to August each year starting from 2019
filtered_rain = rain[(rain['time'].dt.year >= 2019)]

# Group the filtered data by 'year' and 'month' and calculate monthly total rainfall
monthly_total = filtered_rain.groupby(['year', 'month'])['rain'].sum().reset_index()

# Create a new DataFrame for the monthly total rainfall with a single 'time' column
monthly_total['time'] = pd.to_datetime(monthly_total[['year', 'month']].assign(day=1))
monthly_total = monthly_total.set_index('time')

# Plot the monthly total rainfall on the second y-axis
#line2 = ax2.plot(monthly_total.index, monthly_total['rain'], label, linewidth=2, color='blue')

# Set the y-axis label for rainfall
ax2.set_ylabel('Monthly Total Rainfall, kg m-2 s-1')

# Plot a non-transparent white overlay for the months of September to March
for year in range(2019, 2023):
    start_date = pd.to_datetime(f'{year}-09-01')
    end_date = pd.to_datetime(f'{year + 1}-03-31')
    ax2.axvspan(start_date, end_date, color='white')

# Set the x-axis label
plt.xlabel('Dates')

# Set the x-axis limits to start from January 2019 and end at July 2023
plt.xlim(pd.to_datetime('2019-01-01'), pd.to_datetime('2023-07-31'))

# Rotate x-axis labels by 90 degrees
plt.setp(ax1.get_xticklabels(), rotation=90)


# Set the x-axis major ticks to show January and July each year
ax1.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1, 7]))
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))

# Combine the line handles and labels for the legend
#lines = line1 + line2
#labels = [l.get_label() for l in lines]

# Move the legend outside the graph area
plt.legend(loc='upper right')
#ax1.grid(True)

# Display the plot
plt.show()



# In[11]:


#RAIN PLOT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Select a random sample of 10 columns from df_other
sample_columns = AMWI_unique.sample(n=5, axis=1)
sample_columns = sample_columns.interpolate(method='linear')

# Calculate the average and standard deviation for the sample columns
average = AMWI_unique.mean(axis=1)
std_dev = AMWI_unique.std(axis=1)

# Create the figure and axes
fig, ax1 = plt.subplots()

# Plot the AMWI data on the first y-axis
line1 = ax1.plot(AMWI_unique.index, average, label='Average AMWI', linewidth=2, color='black')

# Set the y-axis label for AMWI
ax1.set_ylabel('AMWI - Contamination Levels')

# Create a twin axes sharing the x-axis
ax2 = ax1.twinx()

# Filter the rainfall data for April to August each year starting from 2019
filtered_rain = rain[(rain['time'].dt.year >= 2019)]

# Group the filtered data by 'year' and 'month' and calculate monthly total rainfall
monthly_total = filtered_rain.groupby(['year', 'month'])['rain'].sum().reset_index()

# Create a new DataFrame for the monthly total rainfall with a single 'time' column
monthly_total['time'] = pd.to_datetime(monthly_total[['year', 'month']].assign(day=1))
monthly_total = monthly_total.set_index('time')

# Plot the monthly total rainfall on the second y-axis
line2 = ax2.plot(monthly_total.index, monthly_total['rain'], label='Monthly Total Rainfall', linewidth=2, color='blue')

# Set the y-axis label for rainfall
ax2.set_ylabel('Monthly Total Rainfall, kg m-2 s-1')

# Plot a non-transparent white overlay for the months of September to March
for year in range(2019, 2023):
    start_date = pd.to_datetime(f'{year}-09-01')
    end_date = pd.to_datetime(f'{year + 1}-03-31')
    ax2.axvspan(start_date, end_date, color='white')

# Set the x-axis label
plt.xlabel('Dates')

# Set the x-axis limits to start from January 2019 and end at July 2023
plt.xlim(pd.to_datetime('2019-01-01'), pd.to_datetime('2023-07-31'))

# Rotate x-axis labels by 90 degrees
plt.setp(ax1.get_xticklabels(), rotation=90)


# Set the x-axis major ticks to show January and July each year
ax1.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1, 7]))
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))

# Combine the line handles and labels for the legend
lines = line1 + line2
labels = [l.get_label() for l in lines]

# Move the legend outside the graph area
plt.legend(lines, labels, loc='upper right')
#ax1.grid(True)

# Display the plot
plt.show()



# In[ ]:


#INDIVIDUAL BODY LINES
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'AMWI_unique'
# Assuming the 'Zolote' column contains numeric data
Zolote = AMWI_unique['Illinka'].interpolate(method='linear')

# Creating the figure and axes
fig, ax1 = plt.subplots()

# Plotting the line graph
ax1.plot(Zolote, label='Luhan', linewidth=2, marker='o',color='orange')

# Adding labels and title to the plot
ax1.set_xlabel('Date')
ax1.set_ylabel('AMWI')
ax1.set_title('Line Graph for Illinka')
ax2 = ax1.twinx()

# Blocking out data outside of April to August
for year in range(2019, 2023):
    start_date = pd.to_datetime(f'{year}-09-10')
    end_date = pd.to_datetime(f'{year+1}-04-01')
    ax2.axvspan(start_date, end_date, facecolor='white', edgecolor='none')

plt.setp(ax1.get_xticklabels(), rotation=90)

# Displaying the plot
plt.show()





# In[14]:


#6. GET ROLLING 7 DAY AVERAGES FOR RAINFALL
# Convert the date column in the 'rain' DataFrame to a consistent format
rain['time'] = pd.to_datetime(rain['time'])

# Convert the index (row names) in the 'AMWI_unique' DataFrame to a consistent format
AMWI_unique.index = pd.to_datetime(AMWI_unique.index)

# Merge the DataFrames based on the date column using a left join
merged_df = rain.merge(AMWI_unique, left_on='time', right_index=True, how='left')

# Convert 'rain' column to numeric type
merged_df['rain'] = pd.to_numeric(merged_df['rain'], errors='coerce')

merged_df['rainfall_7_day_total'] = merged_df['rain'].rolling(window=7, min_periods=1).sum()
merged_df['rainfall_14_day_total'] = merged_df['rain'].rolling(window=14, min_periods=1).sum()
merged_df['rainfall_30_day_total'] = merged_df['rain'].rolling(window=30, min_periods=1).sum()
merged_df


# In[15]:


from scipy.stats import pearsonr

# Define the columns for different lagged periods
lagged_columns = {
    '1-day': merged_df['rain'].shift(1),
    '7-day': merged_df['rainfall_7_day_total'],
    '14-day': merged_df['rainfall_14_day_total'],
    '30-day': merged_df['rainfall_30_day_total'],
}

# Define the contamination columns
contamination_columns = merged_df.columns[2:-1]  # Assuming the contamination columns start from the 3rd column and end before the last column

# Create a DataFrame to store the correlation coefficients and p-values
correlation_df = pd.DataFrame(index=contamination_columns, columns=lagged_columns.keys())

# Calculate the Pearson's correlation coefficients and p-values for different lagged periods
for lagged_period, column in lagged_columns.items():
    correlation_results = []
    for contamination_column in contamination_columns:
        valid_indices = np.logical_not(np.logical_or(np.isnan(column), np.isnan(merged_df[contamination_column])))
        correlation, p_value = pearsonr(column[valid_indices], merged_df[contamination_column][valid_indices])
        correlation_results.append((round(correlation, 4), round(p_value, 4)))  # Round to 4 decimal places

    correlation_df[lagged_period] = correlation_results

# Print the correlation coefficients and p-values for each contamination column and lagged period
correlation_df.to_csv(r'C:\Users\jonat\OneDrive\Documents\Diss\pearson.csv')
correlation_df

