# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:27:22 2022

@author: clevine1
"""

import pandas as pd
import matplotlib.pyplot as plt
import lifelines

df = pd.read_csv(r"C:\Users\clevine1\Box\Spring 2022 Work\investigator_nacc57.csv")
#%%
#Model 1
data_all = df.loc[:, ['NACCID', 'VISITYR', 'NACCAGE', 'SEX', 'RACE', 'HISPANIC', 'MARISTAT', 'EDUC', 'JUDGMENT', 'HOMEHOBB', 'NACCMMSE', 'HANDED', 'CDRGLOB']]
#use only data up to 2016
data = data_all[data_all['VISITYR'] <= 2016]
#consider only individuals older than 50 at time of first visit
data = data.loc[data.NACCAGE >= 50]
#Conversion to AD is considered to have occurred if CDR global value is 0.5 or higher (ie nonzero); binary variable
#to be used as censoring column, or 'event_col'
data.loc[data.CDRGLOB > 0, 'CDRGLOB'] = 1

#use only data from participants coded as either black or white due to few data points for other racial backgrounds
data = data[data.RACE <= 2]
#being black is the risk factor
data.loc[data.RACE == 1, 'RACE'] = 0
data.loc[data.RACE == 2, 'RACE'] = 1

#being female is the risk factor
data.loc[data.SEX == 1, 'SEX'] = 0
data.loc[data.SEX == 2, 'SEX'] = 1

data.loc[data.HANDED == 1, 'HANDED'] = 0
data.loc[data.HANDED == 2, 'HANDED'] = 1

#being single would be risk factor
#living as partners & married categories considered lower risk indicator
#drop data where marital status other or unknown
data = data[data.MARISTAT <=6]
data.loc[data.MARISTAT == 1, 'MARISTAT'] = 0
data.loc[data.MARISTAT == 6, 'MARISTAT'] = 0
data.loc[data.MARISTAT >=2, 'MARISTAT'] = 1

#drop invalid data rows
data_c = data.drop(data[(data.HISPANIC == 9) | (data.EDUC == 99)].index)
data_c = data_c.drop(data_c[(data_c.NACCMMSE == -4) | (data_c.NACCMMSE>30)].index)
#consider only right or left-handed people
data_c = data_c.drop(data_c[data_c.HANDED>2].index)
#%%
from lifelines import CoxPHFitter

cph=CoxPHFitter()

datacleaned = pd.DataFrame()

#extract one data point for each participant. For those who convert to MCI, point is age at conversion. For right-censored subjects, take their last data point available.
for particip in data_c['NACCID'].value_counts().index:
    print(data_c.loc[data_c['NACCID']==particip])
    participdata = data_c.loc[data_c['NACCID']==particip]
    #print(participdata.shape[0])
    for i in range(participdata.shape[0]):
        #print(participdata.iloc[i])
        #print(i)
        entry = participdata.iloc[i]
        if(entry['CDRGLOB']==1):
            #print(entry)
            datacleaned = datacleaned.append(entry)
            break
    if((participdata['CDRGLOB']==0).all()):
        #print(participdata.iloc[participdata.shape[0]-1])
        datacleaned = datacleaned.append(participdata.iloc[participdata.shape[0]-1])
    
#save to working directory
datacleaned.to_csv('datacleaned.csv')
#%%
#fit demographic, mmse, cdr information (Model 1)
cph.fit(datacleaned.drop(columns=['NACCID','VISITYR']), duration_col='NACCAGE', event_col='CDRGLOB', formula=('SEX + RACE + HISPANIC + MARISTAT + EDUC + JUDGMENT + HOMEHOBB + HANDED + NACCMMSE'))
#fit the interactions between each factor and the MMSE score along with all the factors alone (Model 2)
cph_inter = CoxPHFitter()
cph_inter.fit(datacleaned.drop(columns=['NACCID','VISITYR']), duration_col='NACCAGE', event_col='CDRGLOB', formula=('NACCMMSE*SEX + NACCMMSE*RACE + NACCMMSE*HISPANIC + NACCMMSE*MARISTAT + NACCMMSE*EDUC + NACCMMSE*JUDGMENT + NACCMMSE*HOMEHOBB + NACCMMSE*HANDED'))
#%%
#Model 3
data_all2 = df.loc[:, ['NACCID', 'VISITYR', 'NACCAGE', 'NACCFAM', 'SMOKYRS', 'HYPERTEN', 'HYPERCHO', 'DEP2YRS', 'CDRGLOB']]
#use only data up to 2016
data_hist = data_all2[data_all2['VISITYR'] <= 2016]
#consider only individuals older than 50 at time of first visit
data_hist = data_hist.loc[data_hist.NACCAGE >= 50]
#Conversion to AD is considered to have occurred if CDR global value is 0.5 or higher (ie nonzero); binary variable
#to be used as censoring column, or 'event_col'
data_hist.loc[data_hist.CDRGLOB > 0, 'CDRGLOB'] = 1

#inactive hypertension and hypercholesterolemia are coded as 2 and active as 1 respectively in original data
#these are switched to reflect higher risk of active disease.
data_hist.loc[data_hist.HYPERTEN == 2, 'HYPERTEN'] = 3
data_hist.loc[data_hist.HYPERTEN == 1, 'HYPERTEN'] = 2
data_hist.loc[data_hist.HYPERTEN == 3, 'HYPERTEN'] = 1
data_hist.loc[data_hist.HYPERCHO == 2, 'HYPERCHO'] = 3
data_hist.loc[data_hist.HYPERCHO == 1, 'HYPERCHO'] = 2
data_hist.loc[data_hist.HYPERCHO == 3, 'HYPERCHO'] = 1

#drop any participant rows with invalid values
data_hist_c = data_hist.drop(data_hist[(data_hist.NACCFAM == -4) | (data_hist.NACCFAM == 9)].index)
data_hist_c = data_hist_c.drop(data_hist_c[(data_hist_c.SMOKYRS == -4) | (data_hist_c.SMOKYRS>=88)].index)
data_hist_c = data_hist_c.drop(data_hist_c[(data_hist_c.HYPERTEN == -4) | (data_hist_c.HYPERTEN == 9)].index)
data_hist_c = data_hist_c.drop(data_hist_c[(data_hist_c.HYPERCHO == -4) | (data_hist_c.HYPERCHO == 9)].index)
data_hist_c = data_hist_c.drop(data_hist_c[(data_hist_c.DEP2YRS == -4) | (data_hist_c.DEP2YRS == 9)].index)
#%%
cphh=CoxPHFitter()

data_hist_cleaned = pd.DataFrame()

for particip in data_hist_c['NACCID'].value_counts().index:
    print(data_hist_c.loc[data_hist_c['NACCID']==particip])
    participdata_c = data_hist_c.loc[data_hist_c['NACCID']==particip]
    #print(participdata_c.shape[0])
    for i in range(participdata_c.shape[0]):
        #print(participdata_c.iloc[i])
        #print(i)
        entry = participdata_c.iloc[i]
        if(entry['CDRGLOB']==1):
            #print(entry)
            data_hist_cleaned = data_hist_cleaned.append(entry)
            break
    if((participdata_c['CDRGLOB']==0).all()):
        #print(participdata_c.iloc[participdata_c.shape[0]-1])
        data_hist_cleaned = data_hist_cleaned.append(participdata_c.iloc[participdata_c.shape[0]-1])
        
data_hist_cleaned.to_csv('data_hist_cleaned.csv')
#%%
cphh.fit(data_hist_cleaned.drop(columns=['NACCID','VISITYR']), duration_col='NACCAGE', event_col='CDRGLOB')

cphh.plot_partial_effects_on_outcome(covariates='HYPERTEN', values=[0, 0.5, 1, 1.5, 2], cmap='coolwarm')

#%%
#Run a combined model that controls for all factors, demographic, medical history, and cognitive exams
data_combined = df.loc[:, ['NACCID', 'VISITYR', 'NACCAGE', 'SEX', 'RACE', 'HISPANIC', 'MARISTAT', 'EDUC', 'JUDGMENT', 'HOMEHOBB', 'NACCMMSE', 'NACCFAM', 'SMOKYRS', 'HYPERTEN', 'HYPERCHO', 'DEP2YRS', 'CDRGLOB']]

#use only data up to 2016
data_comb = data_combined[data_combined['VISITYR'] <= 2016]
#consider only individuals older than 50 at time of first visit
data_comb = data_comb.loc[data_comb.NACCAGE >= 50]
#Conversion to AD is considered to have occurred if CDR global value is 0.5 or higher (ie nonzero); binary variable
#to be used as censoring column, or 'event_col'
data_comb.loc[data_comb.CDRGLOB > 0, 'CDRGLOB'] = 1

#inactive hypertension and hypercholesterolemia are coded as 2 and active as 1; these values are switched to reflect higher risk.
data_comb.loc[data_comb.HYPERTEN == 2, 'HYPERTEN'] = 3
data_comb.loc[data_comb.HYPERTEN == 1, 'HYPERTEN'] = 2
data_comb.loc[data_comb.HYPERTEN == 3, 'HYPERTEN'] = 1
data_comb.loc[data_comb.HYPERCHO == 2, 'HYPERCHO'] = 3
data_comb.loc[data_comb.HYPERCHO == 1, 'HYPERCHO'] = 2
data_comb.loc[data_comb.HYPERCHO == 3, 'HYPERCHO'] = 1

#use only data from participants coded as either black or white due to few data points for other racial backgrounds
data_comb = data_comb[data_comb.RACE <= 2]
#being black is the risk factor
data_comb.loc[data_comb.RACE == 1, 'RACE'] = 0
data_comb.loc[data_comb.RACE == 2, 'RACE'] = 1

#being female is the risk factor
data_comb.loc[data_comb.SEX == 1, 'SEX'] = 0
data_comb.loc[data_comb.SEX == 2, 'SEX'] = 1

#being single would be risk factor
#living as partners & married categories considered lower risk indicator
#drop data where marital status other or unknown
data_comb = data_comb[data_comb.MARISTAT <=6]
data_comb.loc[data_comb.MARISTAT == 1, 'MARISTAT'] = 0
data_comb.loc[data_comb.MARISTAT == 6, 'MARISTAT'] = 0
data_comb.loc[data_comb.MARISTAT >=2, 'MARISTAT'] = 1

#drop any participant rows with invalid values
data_comb_c = data_comb.drop(data_comb[(data_comb.NACCFAM == -4) | (data_comb.NACCFAM == 9)].index)
data_comb_c = data_comb_c.drop(data_comb_c[(data_comb_c.SMOKYRS == -4) | (data_comb_c.SMOKYRS>=88)].index)
data_comb_c = data_comb_c.drop(data_comb_c[(data_comb_c.HYPERTEN == -4) | (data_comb_c.HYPERTEN == 9)].index)
data_comb_c = data_comb_c.drop(data_comb_c[(data_comb_c.HYPERCHO == -4) | (data_comb_c.HYPERCHO == 9)].index)
data_comb_c = data_comb_c.drop(data_comb_c[(data_comb_c.DEP2YRS == -4) | (data_comb_c.DEP2YRS == 9)].index)

data_comb_c = data_comb_c.drop(data_comb_c[(data_comb_c.HISPANIC == 9) | (data_comb_c.EDUC>=99)].index)
data_comb_c = data_comb_c.drop(data_comb_c[(data_comb_c.NACCMMSE == -4) | (data_comb_c.NACCMMSE>30)].index)

data_comb_cleaned = pd.DataFrame()

for particip in data_comb_c['NACCID'].value_counts().index:
    #print(data_comb_c.loc[data_comb_c['NACCID']==particip])
    participdata_cc = data_comb_c.loc[data_comb_c['NACCID']==particip]
    #print(participdata_cc.shape[0])
    for i in range(participdata_cc.shape[0]):
        #print(participdata_cc.iloc[i])
        #print(i)
        entry = participdata_cc.iloc[i]
        if(entry['CDRGLOB']==1):
            #print(entry)
            data_comb_cleaned = data_comb_cleaned.append(entry)
            break
    if((participdata_cc['CDRGLOB']==0).all()):
        #print(participdata_cc.iloc[participdata_cc.shape[0]-1])
        data_comb_cleaned = data_comb_cleaned.append(participdata_cc.iloc[participdata_cc.shape[0]-1])
    
data_comb_cleaned.to_csv('data_comb_cleaned.csv')
#%%
cphhh = CoxPHFitter()
cphhh.fit(data_comb_cleaned.drop(columns=['NACCID','VISITYR']), duration_col='NACCAGE', event_col='CDRGLOB')
#%%
plt.plot(cph.baseline_hazard_)
plt.title("Baseline Hazard, Model 1")

plt.plot(cphh.baseline_hazard_)
plt.title("Baseline Hazard, Model 3")