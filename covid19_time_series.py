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
import matplotlib.path as mpltPath
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib import colors
import seaborn as sns
from datetime import date, timedelta
from collections import defaultdict
from win32api import GetSystemMetrics
import sys
import shapefile
from shapely import wkb
from shapely.geometry import shape, asShape
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint, LineString

import webbrowser
import folium
from folium.plugins import HeatMap

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
    
    df.rename(columns = {
        'rollingGrowthRate':'rollingGrowthRate{}'.format(case),
        'rollingDoubleRate':'rollingDoubleRate{}'.format(case)
        },
        inplace = True)
    
    return df





def shapeFileDetails(shapeFilePath):
    '''Extract shapefile fields and details from shapefile records in the provided path
        and returns the following elements:
        
        shapeFile = shapefile.Reader instance,
        sfDetails = dataframe of contained attributes
    '''
    
    sfReader = shapefile.Reader(shapeFilePath, encoding = 'latin-1')

    #sfShapes = map(shapeFilePolygons, shapeFile.shapes())
    
    colNames = [field[0] for field in sfReader.fields[1:]]   
    
    sfDetails = pd.DataFrame.from_records(sfReader.records(),
                                          columns = colNames)
    
    #return sfReader, sfDetails
    return sfDetails



def shapeFileExtractShapes(shapeFilePath, 
                           labelIndex = None, 
                           addLabels = False, 
                           chainShapes = False):
    '''Extract shapes and label from shapefile instance.
    
        If addLabels is true, provide a label index for the shapeRecord and
        the function will return a tuple (label, shapeList), otherwise
        fuction returns a list of shapes generated from the shapefile.
        
        If chainShapes is true, function will unpack shapes from nested lists
        into a single list, or if addLabels is also true it will return a single
        list of (label, shape) where the label is mapped to each individual shape
        in the shapeList.
        '''    
    
    sfReader = shapefile.Reader(shapeFilePath, encoding = 'latin-1')
    
    if addLabels:    
    
        #shapeList = [(s.record[labelIndex], shapeFilePolygons(s.shape)) 
        #            for s in sfReader.iterShapeRecords()]

        shapeList = list(
            map(lambda s: (s.record[labelIndex], shapeFilePolygons(s.shape)), 
                sfReader.iterShapeRecords())
            )
        

        if chainShapes:
            shapeList = list(
                map(lambda t: map(lambda tt: (t[0], tt), t[1]), 
                    shapeList)
                )
            
            shapeList = list(chain(*shapeList))

    else:

        shapeList = list(
            map(lambda s: shapeFilePolygons(s.shape), 
                sfReader.iterShapeRecords())
            )
        
        if chainShapes:        
            shapeList = list(chain(*shapeList))
      
    return shapeList
        

def getShapeCoords(sShape):
    '''Get coordinates for either a polygon or multipolygon by passing polygon
        objects into the getPolygonCoords function.
        
        Return a list of tuples where the first element in the tuple is
        the exterior boundary of the polygon and the secon element
        is a list of the interior holes (if any) of the polygon'''
        
    if sShape.type == 'MultiPolygon':
        return list(
            map(lambda poly: getPolygonCoords(poly),  
                list(sShape.geoms))        
            )
        #return [getPolygonCoords(poly) for poly in list(shape.geoms)]    
            
    elif sShape.type == 'Polygon':
        return [getPolygonCoords(sShape)]
        

def shapeFilePolygons(shapeFile):
    '''Convert shapefile into individual polygons and return list of Shapely Polygon objects.'''

    # Starting point for each shape
    partIndexStart = list(shapeFile.parts)
    
    # Ending point for each shape (should = starting point)    
    partIndexEnd = np.roll(partIndexStart, -1).tolist()
    
    # Last shape correction for ending point    
    partIndexEnd[-1] = len(shapeFile.points)
    
    # Zip starting and ending points
    partIndexRange = zip(partIndexStart, partIndexEnd)    

    # Get points for each polygon and create shapely polygon object
    polys = list(
        map(lambda r: Polygon(shapeFile.points[r[0]:r[1]]), 
            partIndexRange)
        )
    

    return polys


def plotPath(path, ax, color = 'k', lw = 1, alpha = 0.25, zorder = 2, label = None, **kwargs):
    '''Plot polygon patch using MPL Path Object'''    
    
    patch = patches.PathPatch(path,
                              facecolor = color,
                              alpha = alpha,
                              lw = lw,
                              zorder = zorder,
                              label = label)        
    
    ax.add_patch(patch)        
    return
      
      
      
def plotPolygonCoords(coordList, ax, **kwargs):
    
    # Each coordList is a tuple:
    #   [0] = exterior coordinates
    #   [1] = list of interior coordinates (if any)
    
    # Plot Exterior
    plotPath(mpltPath.Path(coordList[0]), ax, **kwargs)
    
    # Plot Interior
    if len(coordList) > 1:
        map(lambda p: plotPath(mpltPath.Path(p), ax, color = 'w'), coordList[1])

    return



def getShapeListLimits(shapeList):
    '''Get xMin, xMax, yMin, yMax from a list of polygons using
        the envelope or bounding box of each shape
        
        return xMin, yMin, xMax, yMax 
    '''
    if type(shapeList) == list:
        
        # Get bounds for each shape
        boundsList = list(map(lambda s: s.bounds, shapeList))
        
        # Transose list
        boundsList = list(zip(*boundsList))
        
        # Get Limits
        xMin, yMin = list(map(min, boundsList[:2]))
        xMax, yMax = list(map(max, boundsList[2:]))
        
    else: xMin, yMin, xMax, yMax = shapeList.bounds
    
    return (xMin, yMin, xMax, yMax)
     
     

def getShapeDictLimits(shapeDict):
    '''Get overall limits for all shapes contained in a dictionary of 
        shapes organized in the lists by defined keys:
        
        dict input structure: {key1 : [shapeList1],
                               key2 : [shapeList2]}
        
        return overall min and max by calling getShapeListLimits for
        each kv pair and then reducing to find global mins and maxs
    '''
    
    

    shapeBounds = list(
        map(lambda shapeList: getShapeListLimits(shapeList), 
            shapeDict.values())
        )
                  
    xMin, yMin = list(map(min, list(zip(*shapeBounds))[:2]))
    xMax, yMax = list(map(max, list(zip(*shapeBounds))[2:]))
    
    return (xMin, yMin, xMax, yMax)



def plotShapeList(shapeList, ax, colorMap = 'rainbow', title = None, **kwargs):
    '''Plot a list of shapely polygon objects on the provided axes.
    
        Call plotPolygonCoords to plot each shape and call
        getShapeCoords to get shape coordinates for plotting.  Note,
        getShapeCoords returns a nested list of coordinates that needs
        to be unpacked before plotting'''
    
    
    colorList = generateColorList(colorMap, len(shapeList)) 
    
    # Plot each shape
    [plotPolygonCoords(list(chain(*getShapeCoords(s[0]))), 
                       ax,
                       color = s[1], 
                       **kwargs)
     for shape, color in list(zip(shapeList, colorList))
    ]
    
    # Get shape limits and adjust plot boundaries
    plotBoundaries = getShapeListLimits(shapeList)    
    ax.set_xlim((plotBoundaries[0], plotBoundaries[2]))
    ax.set_ylim((plotBoundaries[1], plotBoundaries[3]))
    
    if title != None: 
        ax.set_title(str(title), fontsize = 14)
        
    return



def generateColorList(colorMap, numColors):
    '''Generate an evenly spaced distribution of colors across the full
        range of the input color map.
        
        Return a list of 3 element tuples (R, Y, B) for plotting.'''
        
    colorMap = cm.get_cmap(colorMap)
    colorArray = np.linspace(0,1, numColors)
    colorList = list(map(lambda c: colorMap(c)[:3], colorArray))

    return colorList
    
        

def plotShapeDict(shapeDict, individualPlots = True, colorMap = 'rainbow', **kwargs):
    '''Plot a dictionary containing lists of polygon objects 
    (most likely generated from shape files).  Dictionary must be in the 
    following format:
    
        {key1 : [shapeList1], key2 : [shapeList2], ...}'''

    keys = list(shapeDict.keys())
    keys.sort()
    
    # Create array of colors for plotting
    colorList = generateColorList(colorMap, len(shapeDict.keys()))

    shapeColorDict = {c[0] : c[1] for c in zip(keys, colorList)}


    # Get plot boundaries
    plotBoundaries = getShapeDictLimits(shapeDict)


    if individualPlots:
        # Determine # of plots and plot order
        # Add plot to combine everything
        nRows = min([(len(keys) + 1) // 3 + 1, 3])
        nCols = ((len(keys) + 1) // nRows) + (((len(keys) + 1) % nRows) > 0)*1
        plotOrder = list(itertools.product(range(nRows), range(nCols)))
        plotOrder.sort()
        
        plotOrderDict = {o[0]:o[1] for o in zip(keys, plotOrder[:len(keys)])}        
        
        # Generate plot instance
        fig, axArr = plt.subplots(
            nRows, nCols, sharex = 'all', sharey = 'all',
            figsize = kwargs.get('figsize', 
                                 (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96)
                                 )
            )
    
        # axes with all images
        axT = axArr[(nRows-1, nCols-1)]
    
        for k, shapeList in shapeDict.items():
            ax = axArr[plotOrderDict[k]]
    
            # Plot all shapes on individual axes
            [plotPolygonCoords(list(chain(*getShapeCoords(s))), 
                               ax,
                               color = shapeColorDict[k], 
                               **kwargs)
             for s in shapeList
             ]
            
            
            # map(lambda s: plotPolygonCoords(list(chain(*getShapeCoords(s))), 
            #                                 ax,
            #                                 color = shapeColorDict[k], 
            #                                 **kwargs),
            #     shapeList)
    
            # Plot all shapes on combined axes
            [plotPolygonCoords(list(chain(*getShapeCoords(s))), 
                               axT,
                               color = shapeColorDict[k], 
                               **kwargs)
             for s in shapeList
             ]
            
            # map(lambda s: plotPolygonCoords(list(chain(*getShapeCoords(s))), 
            #                                 axT,
            #                                 color = shapeColorDict[k], 
            #                                 **kwargs),
            #     shapeList)
    
    
    
            # Adjust plot boundaries
            ax.set_xlim((plotBoundaries[0], plotBoundaries[2]))
            ax.set_ylim((plotBoundaries[1], plotBoundaries[3]))
            ax.set_title(k, fontsize = 14)
    
        axT.set_xlim((plotBoundaries[0], plotBoundaries[2]))
        axT.set_ylim((plotBoundaries[1], plotBoundaries[3]))
        axT.set_title('Combined Shapes Plot', fontsize = 14)

    else:
        
        # Generate plot instance
        fig, axT = plt.subplots(1, 
                                figsize = kwargs.get('figsize', 
                                 (0.9*GetSystemMetrics(0)//96, 
                                  0.8*GetSystemMetrics(1)//96)
                                 ))
    
        for k, shapeList in shapeDict.items():
            # Plot all shapes on combined axes
            [plotPolygonCoords(list(chain(*getShapeCoords(s))), 
                               axT,
                               color = shapeColorDict[k], 
                               **kwargs)
             for s in shapeList
             ]
    
        axT.set_xlim((plotBoundaries[0], plotBoundaries[2]))
        axT.set_ylim((plotBoundaries[1], plotBoundaries[3]))
        axT.set_title('Combined Shapes Plot', fontsize = 14)

    return



#%% ENVIRONMENT
## ############################################################################
   



#%% DEV


import branca

m = folium.Map(location=[35, -100], zoom_start=5)
# time.sleep(1)

steps=20
colormap = branca.colormap.linear.YlOrRd_09.scale(0, 1).to_step(steps)
# colormap.add_to(m) #add color bar at the top of the map


steps = 20
gradientMap = {
    step/steps : colormap.rgb_hex_str(step/steps)
    for step in range(steps)
    }

HeatMap(
        data=currentStatsUSDetail[
            ['Latitude', 'Longitude', 'Deaths']
            ].values.tolist(), 
        radius=8, 
        blur = 15,
        max_zoom=13,
        gradient = {0.2: 'blue', 0.5: 'lime', .9: 'red'}
        ).add_to(m)


colorMapPlot = branca.colormap.linear.YlOrRd_09.scale(0, currentStatsUSDetail['Deaths'].max()).to_step(steps)
# colorMapPlot.add_to(m)


# gradient_map=defaultdict(dict)
# for i in range(steps):
#     gradient_map[1/steps*i] = colormap.rgb_hex_str(1/steps*i)
# colormap.add_to(m) #add color bar at the top of the map


m.save('figures\\foliumTest.html')

webbrowser.open('figures\\foliumTest.html')

#%% LOAD & VISUALIZE SHAPEFILES
## ############################################################################

'''
### SHAPEFILE DATA SOURCES 

usFIPS (US FIPS SHAPES):
   http://www.nws.noaa.gov/geodata/
   http://www.nws.noaa.gov/geodata/catalog/county/html/county.htm

'''

shapeFileReadList = [
    ('world', 'shape_files\\TM_WORLD_BORDERS_SIMPL-0.3.shp', 4),
    # ('usZips', 'shape_files\\us_zip_codes\\cb_2016_us_zcta510_500k.shp', 0),
    ('usStates', 'shape_files\\us_state\\cb_2016_us_state_20m.shp', 0)
    ]


shapeFileReadDict = {t[0]:t[1] for t in shapeFileReadList}


# Shapefile details
sfDetailsDict = {
    label: {'shapeFields' : shapeFileDetails(path),
            'labelIndex' : labelIndex}
    for label, path, labelIndex in shapeFileReadList
    }


del(shapeFileReadList)

# Load shapefiles
sfShapesDict = {k: shapeFileExtractShapes(v, chainShapes=True) 
                    for k,v in shapeFileReadDict.items()
                        if k not in ('usZips', 'canZips', 'canFSAc')}


# Add # of shapes to sfDetailsDict
[sfDetailsDict[label].setdefault('numShapes', len(shapeList))
 for label, shapeList in sfShapesDict.items()]


# Plot all shapes
plotShapeDict(sfShapesDict)




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
    dailyReportGlobal
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
        .rename(columns = {'Lat':'Latitude', 'Long':'Longitude'})
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
        .rename(columns = {'Lat':'Latitude', 'Long_':'Longitude'})
        .fillna({'Deaths':0})
        )
              

del(dailyReportUSDeaths)    



#%% DATA TRANSFORMATIONS
## ############################################################################

# Format reportDate as YYYY-MM-DD
dailyReportGlobal['reportDate'] = [
    pd.to_datetime(dte) # .strftime('%Y-%m-%d')
    for dte in dailyReportGlobal['reportDate'].values.tolist()
    ]

# Format reportDate as YYYY-MM-DD
dailyReportUS['reportDate'] = [
    pd.to_datetime(dte) # .strftime('%Y-%m-%d')
    for dte in dailyReportUS['reportDate'].values.tolist()
    ]


#%% AGGREGATE DAILY DATA
## ###########################################################################


dailyReportCountry = aggregateDailyReport(
    dailyData = dailyReportGlobal,
    levelOfDetail = 'Country/Region',
    thresholdMetric = 'Confirmed',
    thresholdValue = 100
    )


dailyReportState = aggregateDailyReport(
    dailyData = dailyReportUS,
    levelOfDetail = ['Country_Region', 'Province_State'],
    thresholdMetric = 'Confirmed',
    thresholdValue = 50,
    dailyAggDict = {
        'Confirmed': sum,
        'Deaths': sum,
        'Longitude': np.mean,
        'Latitude': np.mean,
        'Population':sum
        }
    )


dailyReportUSDetail = aggregateDailyReport(
    dailyData = dailyReportUS,
    levelOfDetail = [
        'Country_Region', 
        'Province_State', 
        'Admin2', 
        'Combined_Key'
        ],
    thresholdMetric = 'Confirmed',
    thresholdValue = 20,
    dailyAggDict = {
        'Confirmed': sum,
        'Deaths': sum,
        'Longitude': np.mean,
        'Latitude': np.mean,
        'Population':sum
        }
    )




countryCases = generateOnsetDict(
    dailyData = dailyReportCountry, 
    levelOfDetail = 'Country/Region', 
    thresholdMetric = 'Confirmed', 
    thresholdValue = 100)


stateCases = generateOnsetDict(
    dailyData = dailyReportState, 
    levelOfDetail = ['Country_Region', 'Province_State'], 
    thresholdMetric = 'Confirmed', 
    thresholdValue = 50)


dailyReportCountry.sort_values(
    ['Country/Region', 'reportDate'],
    inplace = True
    )


dailyReportState.sort_values(
    ['Country_Region', 'Province_State', 'reportDate'],
    inplace = True
    )



#%% CALCULATE GROWTH RATE
## ############################################################################


for df, thresholdConfirmed, thresholdDeath in (
        (dailyReportState, 50, 25), 
        (dailyReportCountry, 100, 50),
                      # (dailyReportUSDetail, 20)
                      ):
    
    df['growthRateConfirmed'] = [
        (np.log(casesToday/thresholdConfirmed) / days) if days > 0 else 0
        for days, casesToday in 
        df[['daysAfterOnset', 'Confirmed']].values.tolist()
        ]
    
    
    df['doublingRateConfirmed'] = [
        np.log(2)/growthRate if daysAfterOutbreak >= 5
        else np.nan
        for growthRate, daysAfterOutbreak in 
        df[['growthRateConfirmed', 'daysAfterOnset']].values.tolist()
        ]
    
 
    df['growthRateDeaths'] = [
        (np.log(casesToday/thresholdConfirmed) / days) if days > 0 else 0
        for days, casesToday in 
        df[['daysAfterOnset', 'Deaths']].values.tolist()
        ]
    
    
    df['doublingRateDeaths'] = [
        np.log(2)/growthRate if daysAfterOutbreak >= 5
        else np.nan
        for growthRate, daysAfterOutbreak in 
        df[['growthRateDeaths', 'daysAfterOnset']].values.tolist()
        ]
    
    

dailyReportCountry = (
    dailyReportCountry
        .groupby('Country/Region')
        .apply(lambda c: rollingGrowthRate(c, case = 'Confirmed'))
        .groupby('Country/Region')
        .apply(lambda c: rollingGrowthRate(c, case = 'Deaths'))
        )


dailyReportState = (
    dailyReportState
        .groupby(['Country_Region', 'Province_State'])
        .apply(lambda c: rollingGrowthRate(c, case = 'Confirmed'))
        .groupby(['Country_Region', 'Province_State'])
        .apply(lambda c: rollingGrowthRate(c, case = 'Deaths'))
        )


#%% MOST RECENT DATA
## ############################################################################

currentStats = (
    dailyReportCountry[
        (dailyReportCountry['reportDate'] 
        == dailyReportCountry['reportDate'].max())]
    )



currentStatsUS = (
    dailyReportState[
        (dailyReportState['reportDate'] 
        == dailyReportState['reportDate'].max())
        & (dailyReportState['Country_Region']=='US')]
    )

currentStatsUSDetail = (
    dailyReportUSDetail[
        dailyReportUSDetail['reportDate'] 
        == dailyReportUSDetail['reportDate'].max()]
    )



#%% VISUALIZE MOST IMPACTED COUNTRIES
## ###########################################################################

sns.set_context('paper')

fig, axArr = plt.subplots(nrows = 2, ncols = 3,
                          figsize = (0.9*GetSystemMetrics(0)//96, 
                                    0.8*GetSystemMetrics(1)//96)
                          )

confirmedThreshold = 50000


for ax, case in enumerate(
        ('Confirmed', 'Deaths', 'deathRate', 
         'rollingDoubleRateConfirmed', 'rollingDoubleRateDeaths')
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


    if case.find('rollingDoubleRate') >= 0:
        axArr.flatten()[ax].set_ylim(
            (0, min(
                max(axArr.flatten()[ax].get_ylim()), 20)
                )
            )

for i, ax in enumerate(axArr.flatten()):
        # Put legend in 2nd figure
    if i == 2:
        ax.legend(bbox_to_anchor = (1.04,1), borderaxespad=0)
    else:
        ax.legend().remove()


fig.suptitle('Countries with > {:,} Confirmed Cases'.format(confirmedThreshold), 
             fontsize = 24)



#%% VISUALIZE US STATES
## ############################################################################

# Plot Confirmed Cases, Deaths, and Death Rate
        
fig, axArr = plt.subplots(nrows = 2, ncols = 3,
                          figsize = (0.9*GetSystemMetrics(0)//96, 
                                    0.8*GetSystemMetrics(1)//96)
                          )

confirmedThreshold = 5000

plotData = (dailyReportState[
    (dailyReportState['Confirmed'] > 20) # At least 20 cases reports
    & [stateCases.get((country, state), 
                      {'Confirmed':0}
                      ).get('Confirmed') > confirmedThreshold # State has over 200 cases
    for country, state in 
    dailyReportState[['Country_Region', 'Province_State']].values.tolist()]
    ]
    )



for ax, case in enumerate(
        ('Confirmed', 'Deaths', 'deathRate', 
         'rollingDoubleRateConfirmed', 'rollingDoubleRateDeaths')
        ):


    plotDict = {
        'x' : 'reportDate',
        'y' : case,
        'hue' : 'Province_State',
        'palette' : 'tab20',
        'data' : plotData,
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
    
    
    if case.find('rollingDoubleRate') >= 0:
        # 20 or the current axis limit, whichever is lower
        axArr.flatten()[ax].set_ylim(
            (0, min(
                max(axArr.flatten()[ax].get_ylim()), 20)
                )
            )
        
        
for i, ax in enumerate(axArr.flatten()):
        # Put legend in 2nd figure
    if i == 2:
        ax.legend(bbox_to_anchor = (1.04,1), borderaxespad=0)
    else:
        ax.legend().remove()


fig.suptitle('US States with > {:,} Confirmed Cases'.format(confirmedThreshold), 
             fontsize = 24)

