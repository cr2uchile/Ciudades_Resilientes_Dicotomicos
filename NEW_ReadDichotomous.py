# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:41:38 2020

Reads PM data from dichotomus samplers and provides simple graph of time series
@author: laura Gallardo +Camilo Menares

"""
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html


#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import info_ReadDichotomous  #Import key settings for the following graphs and data
import numpy as np     
#Reading excel files with data.  


df = pd.DataFrame() 

for i in [1988,1989]:
    print(i)
    var=info_ReadDichotomous.leer_dico0(str(i))
    df=pd.concat([df,var])
df1=df

#The second file contains data for years from 1988 through 2016 (leer_dico1)
# Data structure is as follows:
# Col1=Date (excel format):thereafter stations EMF, EMN, EMM, i.e Independencia, Parque O'Higgins and Las Condes, 
#For each station, the first columns is PM2.5 in ug/m3 and the second PM10  
df2 = pd.DataFrame() 
for i in range(1990,2017):
    print(str(i))
    var=info_ReadDichotomous.leer_dico1(str(i)) 
    df2 = pd.concat([df2,var])

df=pd.concat([df,df2])

#The first file contains data for years from 2017 through 2020 (leer_dico1)
# Data structure is as follows:
# Col1=Date (excel format):thereafter stations EMF, EMN, EMM, i.e Independencia, Parque O'Higgins and Las Condes, 
#For each station, the first columns is PM2.5 in ug/m3 and the second PM10


data_2017 = info_ReadDichotomous.leer_dico2('2017')
data_2018 = info_ReadDichotomous.leer_dico2('2018')
data_2019 = info_ReadDichotomous.leer_dico2('2019')
data_2020 = info_ReadDichotomous.leer_dico2('2020')
df3=pd.concat([data_2017,data_2018,data_2019,data_2020],axis=0)

df=pd.concat([df,df3],axis=0)


# Simple graph

for i in ['Independencia', 'Parque'+' O'+'\'' +'Higgins','Las Condes']:
    info_ReadDichotomous.graph_ts(df,i)
