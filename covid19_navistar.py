# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:25:10 2020

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
# if plt.get_backend() == 'Qt5Agg':
#     from matplotlib.backends.qt_compat import QtWidgets
#     qApp = QtWidgets.QApplication(sys.argv)
#     plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()

#%% FUNCTIONS
## ############################################################################


def readTimeSeriesData(path, case):
    '''Read in time series data and add column for which 
        case (confirmed, deaths, recovered)
    '''
    
    
    data = pd.read_csv('{}\\time_series_19-covid-{}.csv'.format(path, case))
    data['status'] = case
    
    return data


def readDailyReportData(path, dataFile, colRenameDict):
    '''Read in daily report data and add column of file date'''
    
    data = pd.read_csv('{}\\{}'.format(path, dataFile)
                       ).rename(columns = colRenameDict)
    
    
    # Error handle for file name changes on 3/23/20
    if dataFile[:-4] >= '03-23-2020':
        data = (
            data.groupby(['Country/Region', 'Province/State'])
                .agg({'Last Update':max,
                      'Longitude':np.mean,
                      'Latitude':np.mean,
                      'Confirmed':sum,
                      'Deaths':sum,
                      'Recovered':sum})
                .reset_index()
                )
    
    data['reportDate'] = dataFile.replace('.csv','')
    
    return data


def readDailyReportDataAllFields(path, dataFile, colRenameDict):
    '''Read in daily report data and add column of file date'''
    
    data = pd.read_csv('{}\\{}'.format(path, dataFile)
                       ).rename(columns = colRenameDict)
    
    

    
    data['reportDate'] = dataFile.replace('.csv','')
    
    return data




def extractUSState(location, stateAbrevDict):
    '''Split city, state location and look up full name for state.
        
        Return full state name or original location name if no match
        '''
    
    return stateAbrevDict.get((location.split(',')[-1]).strip()[:2], location)
    


def daysAfterOnset(dte, levelOfDetail, onsetDict):
    '''Calculate days after onset of outbreak give the country, current date
        and outbreak date
        
    Return day delta
    '''
    
    firstDate = onsetDict.get(levelOfDetail, 
                                {'reportDate' : dte}
                                ).get('reportDate')
    
    daysAfterOnset = (pd.to_datetime(dte) - pd.to_datetime(firstDate)).days
  
    
    return daysAfterOnset




def generateOnsetDict(dailyData, levelOfDetail, 
                      thresholdMetric, thresholdValue, 
                      aggDict = {
                          'reportDate' : min,
                          'Confirmed' : max,
                          'Deaths' : max,
                          'Latitude' : np.mean,
                          'Longitude' : np.mean
                      }
                      ):
    
    '''Return dictionary by levelOfDetail with the first date where
        the threshold Value is exceeded
    '''
    
    
    onsetDict = (
        dailyData[dailyData[thresholdMetric] >= thresholdValue]
            .groupby(levelOfDetail)
            .agg(aggDict)
            .to_dict('index')
        )
    
    return onsetDict



def calculateDaysAferOnset(dailyData, levelOfDetail, 
                            thresholdMetric, thresholdValue, 
                            aggDict = {
                                'reportDate' : min,
                                'Confirmed' : max,
                                'Deaths' : max,
                                'Latitude' : np.mean,
                                'Longitude' : np.mean
                                }
                            ):
        
    '''Add a column to dailyData with the # of days after onset of outbreak'''
                         
    # Identify first date of outbreak                                    
    onsetDict = generateOnsetDict(dailyData, levelOfDetail, 
                      thresholdMetric, thresholdValue, aggDict)
    
    
    # Populate days after column
    
    # Special handling for multi-index levelOfDetail
    if (((type(levelOfDetail) == tuple) | (type(levelOfDetail) == list))
        & (len(levelOfDetail) > 1)):
        dailyData['daysAfterOnset'] = [
            max(daysAfterOnset(cols[0], tuple(cols[1:]), onsetDict), 0)
            for cols in 
            dailyData[['reportDate', *levelOfDetail]].values.tolist()
            ]   
        
        
    # Single level of detail        
    else:
        dailyData['daysAfterOnset'] = [
            max(daysAfterOnset(cols[0], cols[1], onsetDict), 0)
            for cols in 
            dailyData[['reportDate', levelOfDetail]].values.tolist()
            ]       
        
  
    return dailyData
    

def aggregateDailyReport(dailyData, levelOfDetail,
                         dailyAggDict = {
                             'Confirmed': sum,
                             'Deaths': sum,
                             'Recovered': sum,
                             'Open': sum,
                             'dataIsCurrent': max,
                             'Longitude': np.mean,
                             'Latitude': np.mean
                             },
                         thresholdMetric = 'Confirmed',
                         thresholdValue = 100,
                         calculateOutbreak = True,
                         onsetAggDict = {
                             'reportDate' : min,
                             'Confirmed' : max,
                             'Deaths' : max,
                             'Latitude' : np.mean,
                             'Longitude' : np.mean
                             }
                         ):

    
    '''Aggregate daily daily to the desired level of detail and calculate
        days after outbreak if desired.
        Return aggregated dataframe at daily level by levelOfDetail 
    '''
    
    # Append reportDate for Aggregation
    if (type(levelOfDetail) == list) | (type(levelOfDetail) == tuple):
        dailyAgg = levelOfDetail + ['reportDate']
        
    else:
        dailyAgg = [levelOfDetail, 'reportDate']
        
        
    # Aggregate daily data
    dailyDataAgg = (
        dailyData
        # .fillna(0)
        .groupby(dailyAgg)
        .agg(dailyAggDict)
        .reset_index()
        )   
   
    
    # Calculate Death Rate
    dailyDataAgg['deathRate'] = (
        dailyDataAgg['Deaths'] 
        / dailyDataAgg['Confirmed']
        ).fillna(0)
    
    
    # Calculate days after outbreak
    if calculateOutbreak == True:
        dailyDataAgg = calculateDaysAferOnset(
            dailyDataAgg, levelOfDetail, 
            thresholdMetric, thresholdValue, 
            aggDict = onsetAggDict 
            )


    return dailyDataAgg


def rollingGrowthRate(df, windowSize = 4, case = 'Confirmed', outbreakThreshold = 3):
    

    df['rollingPctIncrease'] = (
        (df[case].rolling(window = windowSize)
                 .apply(lambda x: (x.iloc[-1] - x.iloc[0])/x.iloc[0])
        ) /
        (df['daysAfterOnset'].rolling(window = windowSize)
             .apply(lambda x: x.iloc[-1] - x.iloc[0])
             )
        )
    
    df['rollingPctIncrease'] = [
        increase if daysAfterOutbreak >= outbreakThreshold else np.nan
        for increase, daysAfterOutbreak in 
        df[['rollingPctIncrease', 'daysAfterOnset']].values.tolist()
        ]
    
    
    df['rollingGrowthRate'] = (df[case].rolling(window = windowSize)
                 .apply(lambda x: np.log(x.iloc[-1] / x.iloc[0]))
        /
        (df['daysAfterOnset'].rolling(window = windowSize)
             .apply(lambda x: x.iloc[-1] - x.iloc[0])
             )
        )


    df['rollingGrowthRate'] = [
        growthRate if daysAfterOutbreak >= outbreakThreshold else np.nan
        for growthRate, daysAfterOutbreak in 
        df[['rollingGrowthRate', 'daysAfterOnset']].values.tolist()
        ]
    
    
    df['rollingDoubleRate'] = [
        np.log(2)/growthRate if daysAfterOutbreak >= outbreakThreshold
        else np.nan
        for growthRate, daysAfterOutbreak in 
        df[['rollingGrowthRate', 'daysAfterOnset']].values.tolist()
        ]
    
    return df



def estimateSIRb(di, dt, k, i0, s0):
    '''Estimate b coefficient for SIR model
    
        retrun b
    '''
    
    
    b = ((di/dt) + k*i0) / (s0*i0)
    
    return b


def rollingSIRb(df, 
                windowSize = 4, 
                case = 'infectedPct', 
                k = 1/14,
                outbreakThreshold = 3):
    
    # Change in infection %
    diList = (df[case].rolling(window = windowSize)
                 .apply(lambda x: x.iloc[-1] - x.iloc[0])
        ).values.tolist()
    
    # time delta
    dtList = (
        df['daysAfterOnset']
            .rolling(window = windowSize)
            .apply(lambda x: x.iloc[-1] - x.iloc[0])
            .values
            .tolist()
            )
    
    # starting infection %
    i0List = (
        df[case]
            .rolling(window = windowSize)
            .min()
            .values
            .tolist()
            )
    
    # starting susceptible %
    s0List = (
        df['susceptiblePct']
            .rolling(window = windowSize)
            .min()
            .values
            .tolist()
            )

    
    
    
    df['rollingSIRb'] = [
        estimateSIRb(di, dt, k, i0, s0) if ((i0 > 0) & (dt > 0))
        else np.nan
        for di, dt, k, i0, s0
        in zip(diList, dtList, repeat(k, len(diList)), i0List, s0List)
        ]
    
    
    
    
    df['rollingSIRb'] = [
        increase if daysAfterOutbreak >= outbreakThreshold else np.nan
        for increase, daysAfterOutbreak in 
        df[['rollingSIRb', 'daysAfterOnset']].values.tolist()
        ]
       
    
    
    return df



def sirModelSim(N, I0, b, k, dayRange, R0 = 0, F0 = 0):
    
    '''Perform SIR model simulation.
    
        Return dataframe of susceptable, infected, and recovered by day'''
    
    # Empty lists for populating
    s,i,r = [],[],[]
    
    # % susceptible start
    s0 = (N - I0 - F0) / N
    
    # % infected
    i0 = (I0 / N)
    
    # % recovered start
    r0 = ((R0 + F0) / N)
    
    # s[0] = np.float16(s0) 
    # i[0] = np.float16(i0)
    # r[0] = np.float16(r0)
    
    
    s.append(s0)
    i.append(i0)
    r.append(r0)
    
    for d in range(dayRange):
        
        s.append(s[d-1] - b*s[d-1]*i[d-1])
        
        r.append(r[d-1] + k*i[d-1])

        i.append(i[d-1] + (b*s[d-1]*i[d-1] - k*i[d-1]))
        
        
    sir = pd.DataFrame(
        np.vstack((range(dayRange+1), s, i, r)).T,
        columns = ['days', 'susceptiblePct', 'infectedPct', 'recoveredPct'])
        

    for col in ['susceptible', 'infected', 'recovered']:
        sir[col] = N * sir['{}Pct'.format(col)]
        
        
    return sir

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
    'SC' : 'South Carolina',
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


# Traslate fields for alignment
# Column, From, To
translations = [
    ('Country/Region', 'Mainland China', 'China'),
    ('Country/Region', 'UK', 'United Kingdom'),
    ('Country/Region', 'Iran.*', 'Iran'),
    ('Country/Region', 'Hong Kong SAR', 'Hong Kong'),
    ('Country/Region', 'Korea, South', 'South Korea'),
    ('Country/Region', 'Viet Nam', 'Vietnam'),
    ('Country/Region', 'Taiwan\*', 'Taiwan'),
    ('Country/Region', '.*Congo.*', 'Congo'),
    ('Province/State', 'French Polynesia', 'France'), # added 3/24/20
    ('Province/State', 'Fench Guiana', 'French Guiana'), # added 3/24/20
    ('Province/State', ' County', ''), # remove county from name added 3/25/20
    ('Province/State', '.*Virgin Islands.*', 'Virgin Islands'), # added 3/25/20
    ('Province/State', 'Calgary, Alberta', 'Alberta'), # Canada cleanup (added 3/25/20)
    ('Province/State', 'Edmonton, Alberta', 'Alberta'), # Canada cleanup (added 3/25/20)
    ('Province/State', 'London, ON', 'Ontario'), # Canada cleanup (added 3/25/20)
    ('Province/State', 'Montreal, QC', 'Quebec'), # Canada cleanup (added 3/25/20)
    ('Province/State', 'Toronto, ON', 'Ontario'), # Canada cleanup (added 3/25/20)
    ('Province/State', 'New York, NY', 'New York City'), # Canada cleanup (added 3/25/20)
        ]


# Convert tranlations to dictionary format for pd.replace
translationDict = {
    'to_replace': {},
    'value':{}
    }

# Populate translationDict
for translation in translations:
    col, transFrom, transTo = translation

    translationDict['to_replace'].setdefault(col, []).append(transFrom)
    translationDict['value'].setdefault(col, []).append(transTo)




#%% DAILY REPORT DATA INGESTION
## ############################################################################

path = 'csse_covid_19_data\\csse_covid_19_daily_reports'

fileDates = [f.replace('.csv', '') 
             for f in os.listdir(path)
             if f.endswith('.csv') == True
             ]


# Rename columns for concatenation alignment
colRenameDict = {
    'Country_Region':'Country/Region',
    'Province_State':'Province/State',
    'Last_Update':'Last Update',
    'Lat':'Latitude',
    'Long_':'Longitude',
    }


# Load files and add report date
dailyReport = (
    pd.concat(
        [readDailyReportDataAllFields(path, '{}.csv'.format(f), colRenameDict) 
        for f in fileDates],
        axis = 0,
        ignore_index = True,
        sort = True
        )
    .fillna({'Province/State' : '', 'Admin2': ''})
    .replace(to_replace = translationDict['to_replace'],
             value = translationDict['value'],
             regex = True)
    .replace(to_replace = {'^ | $': ''}, regex = True)
    )


# Change to datetimestamp format
for col in ['Last Update', 'reportDate']:
    dailyReport[col] = [
        str(dte.date()) 
        for dte in pd.to_datetime(dailyReport[col])
        ]


# Sort data
dailyReport.sort_values(
    ['Country/Region', 'Province/State', 'reportDate']
    , inplace = True
    )

# Flag for dates where the data is current and not carryover
dailyReport['dataIsCurrent'] = [
    reportDate == lastUpdate
    for reportDate, lastUpdate in 
    dailyReport[['reportDate', 'Last Update']].values.tolist()
    ]


# Save File
dailyReport.to_csv('output_data\\dailyReport_navistar.csv', index = False)


#%% ERROR HANDING
## ###########################################################################

# France
# Canada cities to provinces
# US state level detail only 3/10-3/21

# Flag diamond princess
dailyReport['from_diamond_princess'] = [
    len(re.findall('.*DIAMOND PRINCESS.*', state.upper())) >= 1
    for state in dailyReport['Province/State'].values.tolist()
    ]

# Clean up diamond princess names
#   First clean up name in US information
#   Then clean up diamond princess names
dailyReport['Province/State'] = [
    re.sub('.*Diamond Princess.*', 'Diamond Princess', 
            re.sub(' \(From Diamond Princess\)', '', state) 
            )
    for state in dailyReport['Province/State'].values.tolist()]


# (dailyReport
#      .replace(' \(From Diamond Princess\)', '', 
#               regex = True)
#      .replace('.*Diamond Princess.*', 'Diamond Princess', 
#               regex = True, inplace = True)
#      )

    


# Split city & state for all US data prior to 3/10
dailyReport['Admin2'] = [
    state.split(', ')[0] if ((admin2 == '') & (country == 'US'))
    else admin2
    for admin2, country, state in 
    dailyReport[['Admin2', 'Country/Region', 'Province/State']].values.tolist()
    ]



# Look up and replace State name  for US
dailyReport['Province/State'] = [
    stateAbrev.get(state.split(', ')[-1], state) 
    for state in 
    dailyReport['Province/State'].values.tolist()
    ]


# Save File
dailyReport.to_csv('output_data\\dailyReport_navistar_clean.csv', 
                   index = False)


#%% LOCATIONS GPS
## ###########################################################################

gpsDict = (
    dailyReport
        .groupby(['Country/Region', 'Province/State', 'Admin2'])
        .agg({
            'Latitude': np.mean,
            'Longitude': np.mean
            })
        .to_dict(orient = 'index')
    )


for gps in ['Latitude', 'Longitude']:
    dailyReport[gps] = [
        gpsDict.get((country, state, admin2), {gps:coords}).get(gps, coords)
        for country, state, admin2, coords in 
        dailyReport[['Country/Region', 'Province/State', 
                         'Admin2', gps]
                        ].values.tolist()
        ]


#%% FEATURE ENGINEERING
## ###########################################################################
# Calculate open cases    
dailyReport['Open'] = (
    dailyReport['Confirmed'].fillna(0) - 
    dailyReport['Deaths'].fillna(0) -
    dailyReport['Recovered'].fillna(0)
    )




#%% AGGREGATE DAILY DATA
## ###########################################################################


dailyReportCountry = aggregateDailyReport(
    dailyData = dailyReport,
    levelOfDetail = 'Country/Region',
    thresholdMetric = 'Confirmed',
    thresholdValue = 100
    )


dailyReportState = aggregateDailyReport(
    dailyData = dailyReport,
    levelOfDetail = ['Country/Region', 'Province/State'],
    thresholdMetric = 'Confirmed',
    thresholdValue = 50
    )



countryCases =generateOnsetDict(
    dailyData = dailyReportCountry, 
    levelOfDetail = 'Country/Region', 
    thresholdMetric = 'Confirmed', 
    thresholdValue = 100)


stateCases =generateOnsetDict(
    dailyData = dailyReportState, 
    levelOfDetail = ['Country/Region', 'Province/State'], 
    thresholdMetric = 'Confirmed', 
    thresholdValue = 50)


dailyReportCountry.sort_values(
    ['Country/Region', 'reportDate'],
    inplace = True
    )

dailyReportState.sort_values(
    ['Country/Region', 'Province/State', 'reportDate'],
    inplace = True
    )


#%% CALCULATE GROWTH RATE
## ############################################################################


for df, threshold in ((dailyReportState, 50), 
                      (dailyReportCountry, 100)):
    
    df['growthRate'] = [
        (np.log(casesToday/threshold) / days) if days > 0 else 0
        for days, casesToday in 
        df[['daysAfterOnset', 'Confirmed']].values.tolist()
        ]
    
    
    df['doublingRate'] = [
        np.log(2)/growthRate if daysAfterOutbreak >= 5
        else np.nan
        for growthRate, daysAfterOutbreak in 
        df[['growthRate', 'daysAfterOnset']].values.tolist()
        ]
    
    


dailyReportCountry = (
    dailyReportCountry
        .groupby('Country/Region')
        .apply(lambda c: rollingGrowthRate(c))
        )

dailyReportState = (
    dailyReportState
        .groupby(['Country/Region', 'Province/State'])
        .apply(lambda c: rollingGrowthRate(c))
        )



#%% MOST RECENT DATA
## ############################################################################

currentStats = (
    dailyReportCountry[
        (dailyReportCountry['reportDate'] 
        == dailyReportCountry['reportDate'].max())
        & (dailyReportCountry['Confirmed'] > 1000)]
    )



currentStatsUS = (
    dailyReportState[
        (dailyReportState['reportDate'] 
        == dailyReportState['reportDate'].max())
        & (dailyReportState['Confirmed'] > 100)
        & (dailyReportState['Country/Region']=='US')]
    )




#%% VISUALIZE MOST IMPACTED COUNTRIES
## ###########################################################################

sns.set_context('paper')

fig, axArr = plt.subplots(nrows = 2, ncols = 2,
                          figsize = (0.9*GetSystemMetrics(0)//96, 
                                    0.8*GetSystemMetrics(1)//96)
                          )

confirmedThreshold = 10000


for ax, case in enumerate(
        ('Confirmed', 'Deaths', 'deathRate', 'rollingDoubleRate')
        ):

    plotDict = {
        'x' : 'daysAfterOnset',
        'y' : case,
        'hue' : 'Country/Region',
        'palette' : 'tab20',
        'data' : dailyReportCountry[
            [(countryCases.get(country, 
                               {'Confirmed' : 0}
                               ).get('Confirmed') > confirmedThreshold)
             for country in 
             dailyReportCountry['Country/Region'].values.tolist()
             ]],
        'ax' : axArr.flatten()[ax]
        }



    sns.lineplot(**plotDict)
    
    plotDict.update(legend = False)
    
    sns.scatterplot(**plotDict)

    axArr.flatten()[ax].grid(True)
    # plt.tight_layout()
    
    if case == 'deathRate':
        axArr.flatten()[ax].set_ylim((0,0.12))

        axArr.flatten()[ax].set_yticklabels(map(lambda v: '{:.0%}'.format(v), 
                           axArr.flatten()[ax].get_yticks()
                           )
                       )


    if case == 'rollingDoubleRate':
        axArr.flatten()[ax].set_ylim(
            (0, min(
                max(axArr.flatten()[ax].get_ylim()), 20)
                )
            )

for i, ax in enumerate(axArr.flatten()):
        # Put legend in 2nd figure
    if i == 1:
        ax.legend(bbox_to_anchor = (1.04,1), borderaxespad=0)
    else:
        ax.legend().remove()


fig.suptitle('Countries with > {:,} Confirmed Cases'.format(confirmedThreshold), 
             fontsize = 24)



#%% VISUALIZE US STATES
## ############################################################################

# Plot Confirmed Cases, Deaths, and Death Rate
        
fig, axArr = plt.subplots(nrows = 2, ncols = 2,
                          figsize = (0.9*GetSystemMetrics(0)//96, 
                                    0.8*GetSystemMetrics(1)//96)
                          )

confirmedThreshold = 5000

plotData = dailyReportState[
    (dailyReportState['Country/Region'] == 'US')
    & (dailyReportState['Confirmed'] > 20) # At least 20 cases reports
    & [stateCases.get((country, state), 
                      {'Confirmed':0}
                      ).get('Confirmed') > confirmedThreshold # State has over 200 cases
    for country, state in 
    dailyReportState[['Country/Region', 'Province/State']].values.tolist()]
    ]



for ax, case in enumerate(
        ('Confirmed', 'Deaths', 'deathRate', 'rollingDoubleRate')
        ):

    plotDict = {
        'x' : 'reportDate',
        'y' : case,
        'hue' : 'Province/State',
        'palette' : 'tab20',
        'data' : plotData.sort_values('reportDate'),
        'ax' : axArr.flatten()[ax],
        }



    sns.lineplot(**plotDict)
    
    plotDict.update(legend = False)
    
    sns.scatterplot(**plotDict)

    axArr.flatten()[ax].tick_params(axis = 'x', labelrotation = 90)
    # axArr.flatten()[ax].xticks(rotation = 90)

    axArr.flatten()[ax].grid(True)


    if case == 'deathRate':
        axArr.flatten()[ax].set_ylim((0,0.1))

        axArr.flatten()[ax].set_yticklabels(map(lambda v: '{:.0%}'.format(v), 
                           axArr.flatten()[ax].get_yticks()
                           )
                       )
    if case == 'rollingDoubleRate':
        # 20 or the current axis limit, whichever is lower
        axArr.flatten()[ax].set_ylim(
            (0, min(
                max(axArr.flatten()[ax].get_ylim()), 20)
                )
            )
        
        
for i, ax in enumerate(axArr.flatten()):
        # Put legend in 2nd figure
    if i == 1:
        ax.legend(bbox_to_anchor = (1.04,1), borderaxespad=0)
    else:
        ax.legend().remove()


fig.suptitle('US States with > {:,} Confirmed Cases'.format(confirmedThreshold), 
             fontsize = 24)



#%% SAVE FILES
## ############################################################################
        
dailyReport.to_csv(
    'output_data\\dailyReport.csv', 
    index = False
    )

dailyReportState.to_csv(
    'output_data\\dailyReportState.csv', 
    index = False
    )

   
dailyReportCountry.to_csv(
    'output_data\\dailyReportCountry.csv', 
    index = False
    )



#%% US CENSUS DATA
## ############################################################################

# source: https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-total.html#par_textimage

usPop = pd.read_csv('external_data\\us_census_data\\nst-est2019-alldata.csv')

usPopDict = {
    state : population for state, population in 
    usPop[['NAME', 'POPESTIMATE2019']].values.tolist()
    }



# Add population data
dailyReportState['population'] = [
    usPopDict.get(state, 0) if country == 'US'
    else 0
    for country, state in 
    dailyReportState[['Country/Region', 'Province/State']].values.tolist()
    ]

dailyReportState['susceptiblePct'] = [
    (pop - confirmed) / pop if pop > 0 
    else 1
    for pop, confirmed in 
    dailyReportState[['population', 'Confirmed']].values.tolist()
    ]

# Infected %
dailyReportState['infectedPct'] = [
    confirmed / pop if pop > 0 
    else 0
    for pop, confirmed in 
    dailyReportState[['population', 'Confirmed']].values.tolist()
    ]


# Estimate b coefficients for SIR model
dailyReportState = (
    dailyReportState
        .groupby(['Country/Region', 'Province/State'])
        .apply(lambda c: rollingSIRb(c))
        )



currentStatsUS = (
    dailyReportState[
        (dailyReportState['reportDate'] 
        == dailyReportState['reportDate'].max())
        & (dailyReportState['Confirmed'] > 1000)
        & (dailyReportState['Country/Region']=='US')]
    )


x = dailyReportState[dailyReportState['Province/State']=='Illinois']


y = rollingSIRb(x, case = 'infectedPct')

df = x

#%% SIR MODEL
## ###########################################################################

# source: https://www.maa.org/press/periodicals/loci/joma/the-sir-model-for-spread-of-disease-the-differential-equation-model

population = 10000
infectedStart = 1
recoveredStart = 0
fatalStart = 0

# recovery rate k
recoverRate = (1/14)



x = dailyReportState[dailyReportState['Province/State']=='New York']


y = rollingSIRb(dailyReportState[dailyReportState['Province/State']=='New York'], case = 'infectedPct')


nySim = sirModelSim(N = x['population'], 
                    I0 = x['Confirmed'], 
                    b = x['rollingSIRb'], 
                    k = 1/14, 
                    dayRange = 100
                    )

x = y.iloc[-1,:]

ilSim = sirModelSim(N = x['population'], 
                    I0 = x['Confirmed'], 
                    b = x['rollingSIRb'], 
                    k = 1/14, 
                    dayRange = 100
                    )


fig, ax = plt.subplots(1)

plt.stackplot(nySim['days'], ilSim[[
 'infectedPct',
 'susceptiblePct',
 'recoveredPct']].T.values.tolist(), labels=[ 'infectedPct',
 'susceptiblePct',
 'recoveredPct'], colors = ['r', 'b', 'g'])
plt.legend(loc='upper left')
plt.show()


#%% DAILY REPORT DATA FILL
## ###########################################################################

# Populate all days for all locations

# All unique locations
uniqueLocations = list(set(
    [tuple(x) for x in 
      dailyReport[['Country/Region', 'Province/State', 'Admin2']].values]
    ))

# Create shell for filling data
dailyReportFull = pd.DataFrame([
    (*l, d) for l, d in 
    product(uniqueLocations, 
            [str(dte.date()) 
              for dte in pd.to_datetime(fileDates)])
    ],
    columns = ['Country/Region', 'Province/State', 'Admin2', 'reportDate']
    )


# Populate with reported data
dailyReportFull = (
    dailyReportFull.set_index(
        ['Country/Region', 'Province/State', 'Admin2', 'reportDate']
        )
    .merge(dailyReport.set_index(
        ['Country/Region', 'Province/State', 'Admin2', 'reportDate']
        ),
        left_index = True,
        right_index = True,
        how = 'left'
        )
    .reset_index()
    )



#%% DEV
## ###########################################################################






# for gps in ['Latitude', 'Longitude']:
#     dailyReportFull[gps] = [
#         gpsDict.get((country, state, admin2), {gps:coords}).get(gps, coords)
#         for country, state, admin2, coords in 
#         dailyReportFull[['Country/Region', 'Province/State', 
#                          'Admin2', gps]
#                         ].values.tolist()
#         ]



# # Fill all dates prior to first report date with 0
# firstReportDateDict = {}
# for case in ('Confirmed', 'Deaths', 'Recovered'):
    
#     # Get first report date for each case type
#     firstReportDateDict[case] = (
#         dailyReport[dailyReport[case] > 0]
#         .groupby(['Country/Region', 'Province/State'])
#         .agg({
#             'reportDate' : np.min
#             })
#         .to_dict(orient = 'index')
#     )
    
#     # Fill dates prior to first report date with 0
#     dailyReportFull[case] = [
#         0 if dte < firstReportDateDict[case].get((country, state), 
#                                            {'reportDate': dte}
#                                            ).get('reportDate')
#         else caseCount
#         for country, state, dte, caseCount in
#         dailyReportFull[['Country/Region', 'Province/State', 
#                          'reportDate', case]].values.tolist()   
#         ]
    
 

# dailyReportFull['Province/State_Agg'] = [
#     extractUSState(location, stateAbrev) if country == 'US'
#     else location for country, location in 
#     dailyReportFull[['Country/Region', 'Province/State']].values.tolist()
#     ]
    

# # Flag missing data
# dailyReportFull['missingData'] = (
#     dailyReportFull[['Confirmed', 'Deaths', 'Recovered']]
#     .isna()
#     .any(axis = 1)
#     )


# # Fill Missing Data
# for case in ('Confirmed', 'Deaths', 'Recovered'):
#     dailyReportFull[case] = (
#         dailyReportFull.groupby(['Country/Region', 'Province/State'])[case]
#             .apply(lambda df: df.fillna(method = 'ffill')).fillna(0)
#         )

# # Calculate open cases    
# dailyReportFull['Open'] = (
#     dailyReportFull['Confirmed'].fillna(0) - 
#     dailyReportFull['Deaths'].fillna(0) -
#     dailyReportFull['Recovered'].fillna(0)
#     )

# # Fill empty isCurrent fields with False
# dailyReportFull.fillna({'dataIsCurrent':False}, inplace = True)


# dailyReportFull.to_csv('dailyReportFull_test.csv', index = False)



# #%% TIME SERIES DATA INGESTION
# ## ############################################################################
# path = 'csse_covid_19_data\\\csse_covid_19_time_series'


# timeSeriesReport = pd.concat([
#     readTimeSeriesData(path, case) 
#     for case in ('Confirmed', 'Deaths', 'Recovered')
#     ],
#     axis = 0
#     ).fillna({'Province/State' : 'x'})




# timeSeriesReport['USstate'] = [
#     (location.split(',')[-1]).strip()
#     if len(location.split(',')) > 1 else 'x'
#     for location in timeSeriesReport['Province/State'].fillna('x').values.tolist()
#     ]


# timeSeriesReport['Province/State_Agg'] = [
#     stateAbrev.get(st, loc)
#     for st, loc in 
#     timeSeriesReport[['USstate', 'Province/State']].values.tolist()
#     ]

# timeSeriesReportMelt = timeSeriesReport.melt(
#     id_vars = ['Province/State', 'Country/Region', 
#                'Lat', 'Long', 
#                'status', 'USstate',
#                'Province/State_Agg'],
#     var_name = 'date',
#     value_name = 'cumlCount'
#     )
    

# # Convert to timestamp
# timeSeriesReportMelt['date'] = pd.to_datetime(timeSeriesReportMelt['date'])


# # Create string of timestamp
# timeSeriesReportMelt['dateString'] = [
#     str(dte.date()) for dte in timeSeriesReportMelt['date']
#     ]


# # Add dateIsCurrent Boolean
# timeSeriesReportMelt = (timeSeriesReportMelt.merge(
#     pd.DataFrame(
#         dailyReport.set_index(['Country/Region', 'Province/State', 'reportDate']
#                               )['dataIsCurrent']
#         ), 
#     left_on = ['Country/Region', 'Province/State', 'dateString'],
#     right_index = True,
#     how = 'left'
#     )
#     .fillna({'dataIsCurrent' : False})
#     )


# # Create column for interpolating data
# # timeSeriesReportMelt['cumlCountInterpolated'] = [
# #     cumlCount if ((dataIsCurrent == True) 
# #      | (cumlCount == 0 & np.isnan(dataIsCurrent))
# #      ) else np.nan
# #     for cumlCount, dataIsCurrent in 
# #     timeSeriesReportMelt[['cumlCount', 'dataIsCurrent']].values.tolist()
# #     ]


# #%% TIME SERIES DATA AGGREGATION
# ## ############################################################################

# # Align Confirmed, Recovered, and Death columns per day
# timeSeriesReportMeltPivot = timeSeriesReportMelt.pivot_table(
#     index = ['Province/State', 'Country/Region', 
#              'Lat', 'Long', 
#              'date', 'USstate',
#              'Province/State_Agg',
#              'dataIsCurrent'],
#     values = 'cumlCount',
#     columns = 'status',
#     aggfunc = np.sum,
#     fill_value = 0
#     ).reset_index()


# timeSeriesReportMeltPivot.to_csv('timeSeriesReportMeltPivot_teset.csv', index = False)

# # Group by Country and Aggregated Province/State
# timeSeriesStateProv = (
#     timeSeriesReportMeltPivot
#         .groupby(['Country/Region', 'Province/State_Agg', 'date'])
#         .agg({
#             'Lat' : np.mean,
#             'Long' : np.mean,
#             'Confirmed': np.sum,
#             'Deaths' : np.sum,
#             'Recovered' : np.sum,
#             'dataIsCurrent' : np.max
#              })
#         .reset_index()
#     )



# # Group data by country
# timeSeriesCountry = (
#     timeSeriesReportMeltPivot
#         .groupby(['Country/Region', 'date'])
#         .agg({
#             'Lat' : np.mean,
#             'Long' : np.mean,
#             'Confirmed': np.sum,
#             'Deaths' : np.sum,
#             'Recovered' : np.sum,
#             'dataIsCurrent' : np.max
#              })
#         .reset_index()
#     )


# #%% IDENTIFY ONSET DATE
# ## ############################################################################

# confirmedThreshold = 50

# # Dates where confirmed cases above threshold
# timeSeriesCountry['daysAfterOnset'] = [
#     dte if confirmed >= confirmedThreshold
#     else None
#     for dte, confirmed in 
#     timeSeriesCountry[['date', 'Confirmed']].values.tolist()
#     ]



# # Date of first case and total # of cases for each country
# countryCases = (
#     timeSeriesCountry[timeSeriesCountry['Confirmed'] >= 1]
#         .groupby(['Country/Region'])
#         .agg({'date' : np.min,
#               'daysAfterOnset' : np.min,
#               'Confirmed' : np.max
#               })
#         .to_dict('index')
#     )


# # Create date use field in case country doesn't have minimum # of cases
# [d.update(dateUse = [
#     d['date'] if pd.isna(d['daysAfterOnset']) else d['daysAfterOnset']][0])
#     for d in countryCases.values()
# ]



# timeSeriesCountry['daysSinceFirstCase'] = [
#     max((dte - countryCases.get(country, 
#                                 {'dateUse' : dte}
#                                 ).get('dateUse')
#          ).days, 0)
#     for country, dte in 
#     timeSeriesCountry[['Country/Region', 'date']].values.tolist()
#     ]
        



# #%% VISUALIZE MOST IMPACTED COUNTRIES
# ## ###########################################################################

# fig, ax = plt.subplots(1)

# sns.lineplot(x = 'daysSinceFirstCase',
#              y = 'Confirmed',
#              hue = 'Country/Region',
#              palette= 'tab20',
#              data = timeSeriesCountry[
#                  [(countryCases.get(country, 
#                                     {'Confirmed' : 0}
#                                     ).get('Confirmed') > 1000)
#                   for country in 
#                   timeSeriesCountry['Country/Region'].values.tolist()
#                   ]],
#              ax = ax)

# plt.grid()
# plt.tight_layout()



# #%% STORE TIME SERIES DATA
# ## ############################################################################


# timeSeriesCountry.to_csv()




