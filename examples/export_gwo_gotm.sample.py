# Export GWO data to GOTM format input files
# Author: Jun Sasaki  Coded on 2024-12-20  Updated on 2024-12-20
# Copy this as export_gwo_gotm.py, edit it, and run it as follows:
# Usage: python export_gwo_gotm.py
import sys
from metdata import gwo

def read_gwo(ymd_ini, ymd_end, stn, dirpath="/mnt/d/dat/met/JMA_DataBase/GWO/Hourly/"):
    '''
    Read GWO data

    Parameters
    ----------
    ymd_ini : str
        Initial date in the format 'YYYY-MM-DD'
    ymd_end : str
        Final date in the format 'YYYY-MM-DD'
    stn : str
        Station name
    dirpath : str
        Directory path where the GWO data are stored
    '''
    datetime_ini = f"{ymd_ini} 00:00:00"
    datetime_end = f"{ymd_end} 00:00:00"
    return gwo.Hourly(datetime_ini=datetime_ini, datetime_end=datetime_end, stn=stn,
                          dirpath=dirpath)

def export_gotm(ymd_ini, ymd_end, stns, dirpath="/mnt/d/dat/met/JMA_DataBase/GWO/Hourly/"):
    '''
    Export GWO data to GOTM format input files

    Parameters
    ----------
    ymd_ini : str
        Initial date in the format 'YYYY-MM-DD'
    ymd_end : str
        Final date in the format 'YYYY-MM-DD'
    stns : list
        List of station names
    dirpath : str
        Directory path where the GWO data are stored

    Returns
    -------
    None
    '''
    
    for stn in stns:
        gwo_data = read_gwo(ymd_ini, ymd_end, stn=stn, dirpath=dirpath)
        if stn == 'Tokyo':
            file = f"{stn}_light_{ymd_ini}_{ymd_end}.dat"
            gwo.export_gotm(file, gwo_data.df, 'slht', fmt='10.5f')
        file = f"{stn}_precip_{ymd_ini}_{ymd_end}.dat"
        gwo.export_gotm(file, gwo_data.df, 'kous')
        file = f"{stn}_meteo_{ymd_ini}_{ymd_end}.dat"
        gwo.export_gotm_meteo(file, gwo_data.df)

if __name__ == "__main__":
    dirpath = "/mnt/d/dat/met/JMA_DataBase/GWO/Hourly/"
    ymd_ini = "2020-01-01"
    ymd_end = "2021-01-01"
    stns = ['Tokyo', 'Chiba', 'Yokohama']

    export_gotm(ymd_ini, ymd_end, stns, dirpath=dirpath)

