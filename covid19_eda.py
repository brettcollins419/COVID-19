# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:16:46 2020

@author: u00bec7
"""

import numpy as np
import os
import pandas as pd
import time
import re
import string
from itertools import chain
from matplotlib import pyplot as plt
import seaborn as sns

#%% FUNCTIONS
## ############################################################################


def readTimeSeriesData(path, case):
    '''Read in time series data and add column for which 
        case (confirmed, deaths, recovered)
    '''
    
    
    data = pd.read_csv('{}\\time_series_19-covid-{}.csv'.format(path, case))
    data['status'] = case
    
    return data


#%% ENVIRONMENT
## ############################################################################
    

# US state abbreviations for mapping cities to state level
stateAbrev = {
    'AL' : 'Alabama',
    'AK' : 'Alaska',
    'AZ' : 'Arizona',
    'AR' : 'Arkansas',
    'CA' : 'California',
    'CO' : 'Colorado',
    'CT' : 'Connecticut',
    'DE' : 'Delaware',
    'FL' : 'Florida',
    'GA' : 'Georgia',
    'HI' : 'Hawaii',
    'ID' : 'Idaho',
    'IL' : 'Illinois',
    'IN' : 'Indiana',
    'IA' : 'Iowa',
    'KS' : 'Kansas',
    'KY' : 'Kentucky',
    'LA' : 'Louisiana',
    'ME' : 'Maine',
    'MD' : 'Maryland',
    'MA' : 'Massachusetts',
    'MI' : 'Michigan',
    'MN' : 'Minnesota',
    'MS' : 'Mississippi',
    'MO' : 'Missouri',
    'MT' : 'Montana',
    'NE' : 'Nebraska',
    'NV' : 'Nevada',
    'NH' : 'New Hampshire',
    'NJ' : 'New Jersey',
    'NM' : 'New Mexico',
    'NY' : 'New York',
    'NC' : 'North Carolina',
    'ND' : 'North Dakota',
    'OH' : 'Ohio',
    'OK' : 'Oklahoma',
    'OR' : 'Oregon',
    'PA' : 'Pennsylvania',
    'RI' : 'Rhode Island',
    'SC' : 'south Carolina',
    'SD' : 'South Dakota',
    'TN' : 'Tennessee',
    'TX' : 'Texas',
    'UT' : 'Utah',
    'VT' : 'Vermont',
    'VA' : 'Virginia',
    'WA' : 'Washington',
    'WV' : 'West Virginia',
    'WI' : 'Wisconsin',
    'WY' : 'Wyoming',
    }


#%% DAILY REPORT DATA INGESTION
## ############################################################################

path = 'csse_covid_19_data\\csse_covid_19_daily_reports'

dailyReport = pd.concat([
    pd.read_csv('{}\\{}'.format(path, f)) for f in os.listdir(path)
    if f.endswith('.csv') == True
    ],
    axis = 0,
    sort = True
    )




#%% TIME SERIES DATA INGESTION
## ############################################################################
path = 'csse_covid_19_data\\\csse_covid_19_time_series'



timeSeriesReport = pd.concat([
    readTimeSeriesData(path, case) 
    for case in ('Confirmed', 'Deaths', 'Recovered')
    ],
    axis = 0
    ).fillna({'Province/State' : 'x'})




timeSeriesReport['USstate'] = [
    (location.split(',')[-1]).strip()
    if len(location.split(',')) > 1 else 'x'
    for location in timeSeriesReport['Province/State'].fillna('x').values.tolist()
    ]


timeSeriesReport['Province/State_Agg'] = [
    stateAbrev.get(st, loc)
    for st, loc in 
    timeSeriesReport[['USstate', 'Province/State']].values.tolist()
    ]

timeSeriesReportMelt = timeSeriesReport.melt(
    id_vars = ['Province/State', 'Country/Region', 
               'Lat', 'Long', 
               'status', 'USstate',
               'Province/State_Agg'],
    var_name = 'date',
    value_name = 'cumlCount'
    )
    

# Convert to timestamp
timeSeriesReportMelt['date'] = pd.to_datetime(timeSeriesReportMelt['date'])



#%% TIME SERIES DATA AGGREGATION
## ############################################################################

# Align Confirmed, Recovered, and Death columns per day
timeSeriesReportMeltPivot = timeSeriesReportMelt.pivot_table(
    index = ['Province/State', 'Country/Region', 
             'Lat', 'Long', 
             'date', 'USstate',
             'Province/State_Agg'],
    values = 'cumlCount',
    columns = 'status',
    aggfunc = np.sum,
    fill_value = 0
    ).reset_index()




# Group by Country and Aggregated Province/State
timeSeriesStateProv = (
    timeSeriesReportMeltPivot
        .groupby(['Country/Region', 'Province/State_Agg', 'date'])
        .agg({
            'Lat' : np.mean,
            'Long' : np.mean,
            'Confirmed': np.sum,
            'Deaths' : np.sum,
            'Recovered' : np.sum
             })
        .reset_index()
    )



# Group data by country
timeSeriesCountry = (
    timeSeriesReportMeltPivot
        .groupby(['Country/Region', 'date'])
        .agg({
            'Lat' : np.mean,
            'Long' : np.mean,
            'Confirmed': np.sum,
            'Deaths' : np.sum,
            'Recovered' : np.sum
             })
        .reset_index()
    )


#%% IDENTIFY ONSET DATE
## ############################################################################

confirmedThreshold = 50

# Dates where confirmed cases above threshold
timeSeriesCountry['daysAfterOnset'] = [
    dte if confirmed >= confirmedThreshold
    else None
    for dte, confirmed in 
    timeSeriesCountry[['date', 'Confirmed']].values.tolist()
    ]



# Date of first case and total # of cases for each country
countryCases = (
    timeSeriesCountry[timeSeriesCountry['Confirmed'] >= 1]
        .groupby(['Country/Region'])
        .agg({'date' : np.min,
              'daysAfterOnset' : np.min,
              'Confirmed' : np.max
              })
        .to_dict('index')
    )


# Create date use field in case country doesn't have minimum # of cases
[d.update(dateUse = [
    d['date'] if pd.isna(d['daysAfterOnset']) else d['daysAfterOnset']][0])
    for d in countryCases.values()
]



timeSeriesCountry['daysSinceFirstCase'] = [
    max((dte - countryCases.get(country, 
                                {'dateUse' : dte}
                                ).get('dateUse')
         ).days, 0)
    for country, dte in 
    timeSeriesCountry[['Country/Region', 'date']].values.tolist()
    ]
        



#%% VISUALIZE MOST IMPACTED COUNTRIES
## ###########################################################################

fig, ax = plt.subplots(1)

sns.lineplot(x = 'daysSinceFirstCase',
             y = 'Confirmed',
             hue = 'Country/Region',
             palette= 'tab20',
             data = timeSeriesCountry[
                 [(countryCases.get(country, 
                                    {'Confirmed' : 0}
                                    ).get('Confirmed') > 1000)
                  for country in 
                  timeSeriesCountry['Country/Region'].values.tolist()
                  ]],
             ax = ax)

plt.grid()
plt.tight_layout()



#%% AGGREGATE US CASES BY STATE
## ############################################################################

# Dataset switches from city level to state level, so aggregate historical
# city level into the state level for continuity



