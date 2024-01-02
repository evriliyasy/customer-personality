import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

df = pd.read_csv('marketing_campaign.csv')

df.rename(index=str, columns={
    'MntMeatProducts' : 'MeatProducts',
    'MntFishProducts' : 'FishProducts',
    'MntSweetProducts' : 'SweetProducts',
    'MntGoldProds' : 'GoldProducts',
    'Z_CostContact' : 'Cost',
    'Z_Revenue' : 'Revenue'
     
}, inplace=True)

df = df.drop(['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Dt_Customer', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response'], axis=0)


st.header("isi dataset")
st.write(df) 