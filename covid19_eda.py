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

#%% FUNCTIONS
## ############################################################################


def readTimeSeriesData(path, case):
    
    data = pd.read_csv('{}\\time_series_19-covid-{}.csv'.format(path, case))
    data['status'] = case
    
    return data



#%% TIME SERIES DATA INGESTION
## ############################################################################
path = 'csse_covid_19_data\\\csse_covid_19_time_series'



timeSeriesReport = pd.concat([
    readTimeSeriesData(path, case) 
    for case in ('Confirmed', 'Deaths', 'Recovered')
    ],
    axis = 0
    )


timeSeriesReport['USstate'] = [
    (location.split(',')[-1]).strip()
    if len(location.split(',')) > 1 else 'x'
    for location in timeSeriesReport['Province/State'].fillna('x').values.tolist()
    ]


timeSeriesReportMelt = timeSeriesReport.melt(
    id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long', 'status', 'USstate'],
    var_name = 'date',
    value_name = 'cumlCount'
    )
    

timeSeriesReportMeltPivot = timeSeriesReportMelt.pivot_table(
    index = ['Province/State', 'Country/Region', 'Lat', 'Long', 'date', 'USstate'],
    values = 'cumlCount',
    columns = 'status',
    aggfunc = np.sum,
    fill_value = 0)



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

