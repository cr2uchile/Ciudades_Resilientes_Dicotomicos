# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:19:24 2020

@author: laura Gallardo inspired/heled by Camilo Menares

"""

#Importing libraries
import pandas as pd
import numpy as np     

#######

#Reading 1988 and 1989 (only PM10)
def leer_dico0(anho):     
    '''
    

    Parameters
    ----------
    anho : TYPE string of year 'YYYY'
        DESCRIPTION.

    Returns data frame with six columns
    "PM25_EMF","PM10_EMF","PM25_EMN","PM10_EMN","PM25_EMM","PM10_EMM
    with PM25 filled with nan
    -------
    None.

    '''
#Creating time index

    tini=anho+'0101'
    tfin=anho+'1231'
    tid = pd.date_range(tini,tfin, freq= 'D')

    dfa=pd.DataFrame(np.nan, index=tid[0:len(tid)], columns=[0,1,2,3,4,5])

    new_name = ["PM25_EMF","PM10_EMF","PM25_EMN","PM10_EMN","PM25_EMM","PM10_EMM"]
    old_name = dfa.keys()
    
    dfa = dfa.rename(columns={old_name[0]: new_name[0], old_name[1]: new_name[1],
                                    old_name[2]: new_name[2], old_name[3]: new_name[3],
                                    old_name[4]: new_name[4], old_name[5]: new_name[5]})

    
    df = pd.read_excel('DATOS-DICOTOMOS-2-5-10_-1988-2016.xlsx',anho,usecols=[1,2,3],
                       header=2,na_values=["--","S/I",'---'],skipfooter=1)
    old_name = df.keys()
    new_name = ["PM10_EMF","PM10_EMN","PM10_EMM"]
    
    df = df.rename(columns={old_name[0]: new_name[0], old_name[1]: new_name[1],
                                    old_name[2]: new_name[2]})

    PM25_EMF = pd.Series(dfa["PM25_EMF"].values, index=tid[0:len(tid)])
    PM25_EMN = pd.Series(dfa["PM25_EMN"].values, index=tid[0:len(tid)])    
    PM25_EMM = pd.Series(dfa["PM25_EMM"].values, index=tid[0:len(tid)])

    PM10_EMF = pd.Series(df["PM10_EMF"].values, index=tid[0:len(tid)-1])
    PM10_EMN = pd.Series(df["PM10_EMN"].values, index=tid[0:len(tid)-1])
    PM10_EMM = pd.Series(df["PM10_EMM"].values, index=tid[0:len(tid)-1])

    df = pd.DataFrame({ "PM25_EMF" : PM25_EMF, "PM10_EMF" : PM10_EMF, 
                       "PM25_EMN" : PM25_EMN, "PM10_EMN" : PM10_EMN,
                       "PM25_EMM" : PM25_EMM, "PM10_EMM" : PM10_EMM})
    return   df

#######
#Reading 1990-2016 PM 2.5 and 10
    
def leer_dico1(anho):
    '''
    This function reads in an excel file containing 24 h averages of PM2.5 and PM110 in ug/m3N at three sites in Santiago
    collected using a dichotomus sampler. These data were supplied by the Ministry for the Environment
    


    Parameters
    ----------
    anho : TYPE string indicating year 
        DESCRIPTION.

    Returns Data frame containing daily PM2.5 and PM10 for stations F (Independencia), N(Parque O'Higgins') and M(Las Condes)
    -------
    None.

    '''
    
    
 #   df = pd.read_excel('DATOS-DICOTOMOS-2-5-10_-1988-2016.xlsx',anho,header=2,na_values=["--","S/I"],skipfooter=0, index_col=0)
    df = pd.read_excel('DATOS-DICOTOMOS-2-5-10_-1988-2016.xlsx',anho,usecols=[0,1,2,6,7,11,12],
                       header=2,na_values=["--","S/I",'---','S/D'],skipfooter=1,index_col=0)
    new_name = ["PM25_EMF","PM10_EMF","PM25_EMN","PM10_EMN","PM25_EMM","PM10_EMM"]
    old_name = df.keys()
    
    df = df.rename(columns={old_name[0]: new_name[0], old_name[1]: new_name[1],
                                    old_name[2]: new_name[2], old_name[3]: new_name[3],
                                    old_name[4]: new_name[4], old_name[5]: new_name[5]})

    return   df


########
    
#The first file contains data for years from 2017 through 2020
# Data structure is as follows:
# Col1=Date (excel format):thereafter stations EMF, EMN, EMM, i.e Independencia, Parque O'Higgins and Las Condes, 
#For each station, the first columns is PM2.5 in ug/m3 and the second PM10
# Years 2017-2020
def leer_dico2(anho):
    '''
    This function reads in an excel file containing 24 h averages of PM2.5 and PM110 in ug/m3N at three sites in Santiago
    collected using a dichotomus sampler. These data were supplied by the Ministry for the Environment
    


    Parameters
    ----------
    anho : TYPE string indicating year 
        DESCRIPTION.

    Returns Data frame containing daily PM2.5 and PM10 for stations F (Independencia), N(Parque O'Higgins') and M(Las Condes)
    -------
    None.

    '''
    
    
    df = pd.read_excel('Dicotomicos-2017-2020.xlsx',anho,
                       header=2,na_values=["--","S/I",'---'],skipfooter=1, index_col=0)
    
    new_name = ["PM25_EMF","PM10_EMF","PM25_EMN","PM10_EMN","PM25_EMM","PM10_EMM"]
    old_name = df.keys()
    
    df = df.rename(columns={old_name[0]: new_name[0], old_name[1]: new_name[1],
                                    old_name[2]: new_name[2], old_name[3]: new_name[3],
                                    old_name[4]: new_name[4], old_name[5]: new_name[5]})
    
 
    return   df
