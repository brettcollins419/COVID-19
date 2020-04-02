# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:39:36 2020

@author: U00BEC7
"""


import numpy as np
import os
import pandas as pd
import time
import re
import string
from itertools import chain, combinations, product, repeat
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import date, timedelta
from collections import defaultdict
from win32api import GetSystemMetrics
import sys

# Format plots for 4k screens
if plt.get_backend() == 'Qt5Agg':
    from matplotlib.backends.qt_compat import QtWidgets
    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()

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
    


#%% DAILY REPORT DATA INGESTION
## ############################################################################

path = 'csse_covid_19_data\\csse_covid_19_time_series'

# Read in confirmed and death files for global
dailyReportGlobal = (
    pd. read_csv('{}\\time_series_covid19_confirmed_global.csv'.format(path))
    .melt(id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'],
          var_name = 'reportDate',
          value_name = 'Confirmed')
    .fillna({'Province/State':'x'})
    )
    

dailyReportGlobalDeaths = (
    pd. read_csv('{}\\time_series_covid19_deaths_global.csv'.format(path))
    .melt(id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'],
          var_name = 'reportDate',
          value_name = 'Deaths')
    .fillna({'Province/State':'x'})    
    
    )


# Merge confirmed and death files
dailyReportGlobal = (
    dailyReportGlobalDeaths
        .set_index(['Province/State', 'Country/Region', 'reportDate'])
        .merge(
            pd.DataFrame(
                dailyReportGlobalDeaths
                    .set_index(['Province/State', 'Country/Region', 'reportDate'])
                    ['Deaths']
                    ),
            left_index = True,
            right_index = True,
            how = 'left'
            )
        .reset_index()
        .fillna({'Deaths':0})
        )
              

del(dailyReportGlobalDeaths)


# Read in confirmed and death files for US
dailyReportUS = (
    pd. read_csv('{}\\time_series_covid19_confirmed_US.csv'.format(path))
    .melt(id_vars = [
        'UID',
        'iso2',
        'iso3',
        'code3',
        'FIPS',
        'Admin2',
        'Province_State',
        'Country_Region',
        'Lat',
        'Long_',
        'Combined_Key'
        ],
          var_name = 'reportDate',
          value_name = 'Confirmed')
    .fillna('x')
    )


dailyReportUSDeaths = (
    pd. read_csv('{}\\time_series_covid19_deaths_US.csv'.format(path))
    .melt(id_vars = [
        'UID',
        'iso2',
        'iso3',
        'code3',
        'FIPS',
        'Admin2',
        'Province_State',
        'Country_Region',
        'Lat',
        'Long_',
        'Combined_Key',
        'Population'
        ],
          var_name = 'reportDate',
          value_name = 'Deaths')
    .fillna('x')    
    )


# Merge confirmed and death files
dailyReportUS = (
    dailyReportUS
        .set_index(['Combined_Key', 'reportDate'])
        .merge((
            dailyReportUSDeaths
                .set_index(['Combined_Key', 'reportDate'])
                [['Population', 'Deaths']]
            ),
            left_index = True,
            right_index = True,
            how = 'left'
            )
        .reset_index()
        .rename(columns = {'Long_':'Long'})
        .fillna({'Deaths':0})
        )
              

del(dailyReportUSDeaths)    

