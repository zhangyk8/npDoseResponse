#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: May 6, 2024

Data preprocessing on the PM2.5 CMR data.
"""

import numpy as np
import pandas as pd

def TransVarByYear(x):
    if x["Year"] <= 1995:
        return x[1]
    elif x["Year"] >= 2005:
        return x[3]
    else:
        return x[2]
    
if __name__ == "__main__":
    PM25_CMR = pd.read_csv("./SES_PM25_CMR_data/County_annual_PM25_CMR.csv")
    PM25_CMR = PM25_CMR.iloc[:,1:]

    county_RAW = pd.read_csv("./SES_PM25_CMR_data/County_RAW_variables.csv")
    county_RAW = county_RAW.iloc[:,1:]

    county_loc = pd.read_csv("./SES_PM25_CMR_data/us_county_latlng.csv")

    # Merging data frames
    merge_dat = PM25_CMR.merge(county_RAW, how='left', left_on='FIPS', right_on='FIPS')
    merge_dat = merge_dat.merge(county_loc, how='left', left_on='FIPS', right_on='fips_code')

    var_lst = ["civil_unemploy", "median_HH_inc", "femaleHH_ns_pct", "vacant_HHunit", "owner_occ_pct",
               "eduattain_HS", "pctfam_pover"]
    for var in var_lst:
        org_var = ['Year']
        org_var.append(var+'_1990')
        org_var.append(var+'_2000')
        org_var.append(var+'_2010')
        merge_dat[var] = merge_dat[org_var].apply(lambda x: TransVarByYear(x), axis=1)
        
    final_dat = merge_dat[['FIPS', 'Year', 'name', 'lng', 'lat', 'PM2.5', 'CMR', "population_2000", 
                           "civil_unemploy", "median_HH_inc", "femaleHH_ns_pct", "vacant_HHunit", 
                           "owner_occ_pct", "eduattain_HS", "pctfam_pover"]]
    
    dat = final_dat.groupby(['FIPS', 'name']).mean().reset_index()
    final_dat.to_csv('PM25_county_CMR_final.csv', index=False)
    dat.to_csv('PM25_county_CMR_avg.csv', index=False)