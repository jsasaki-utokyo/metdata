# Processing whpview data
# Author: Jun Sasaki@UTokyo  Coded on 2022-02-13  Updated on 2024-07-06
# http://www.roy.hi-ho.ne.jp/ssai/mito_gis/whpview/index.html
# http://www.roy.hi-ho.ne.jp/ssai/mito_gis/
# 気象庁互換形式の値欄の記号について
# https://www.data.jma.go.jp/obd/stats/data/mdrr/man/remark.html

from metdata import gwo
import numpy as np
import pandas as pd
import subprocess
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location

class StationDict:
    '''
    Class to store a dictionary of Whp instances with the keys of station and year.
    Create an instance by whp_data = StationDict(), and access each instance by whp_data[stn, year].
    '''
    def __init__(self):
        self.data = {}

    def __getitem__(self, key):
        station, year = key
        year_key = self._get_year_key(year)
        return self.data[station][year_key]

    def __setitem__(self, key, value):
        station, year = key
        year_key = self._get_year_key(year)
        if station not in self.data:
            self.data[station] = {}
        self.data[station][year_key] = value

    def _get_year_key(self, year):
        if isinstance(year, str) and year.isdigit():
            return int(year)
        return year


class Whp(gwo.Hourly):
    '''
    Adjust the format of whpView hourly dataset to GWO hourly dataset used by gwo.py.
    There are inconsistencies in some columns, e.g., "現在天気" in GWO is missing.
    In such cases, their values are designated as NaN with Remark=0.
    Some columns that do not exist in GWO are deleted.
    気象庁互換形式からGWO形式への変換
    気象庁互換形式の値欄の記号の説明：
    https://www.data.jma.go.jp/obd/stats/data/mdrr/man/remark.html
    '''
    cols_10_times = ["現地気圧(hPa)","海面気圧(hPa)","降水量(mm)","気温(℃)","露点温度(℃)",
                     "蒸気圧(hPa)","風速(m/s)","日照時間(h)"]
    cols_100_times = ["全天日射量(MJ/㎡)"]
    
    rename_cols = {"名称":"観測所名","観測所ID":"ID1","現地気圧(hPa)":"現地気圧(0.1hPa)",
                   "海面気圧(hPa)":"海面気圧(0.1hPa)","降水量(mm)":"降水量(0.1mm/h)","気温(℃)":"気温(0.1degC)",
                   "露点温度(℃)":"露天温度(0.1degC)","蒸気圧(hPa)":"蒸気圧(0.1hPa)","湿度(%)":"相対湿度(%)",
                   "風速(m/s)":"風速(0.1m/s)","風向":"風向(1(NNE)～16(N))","日照時間(h)":"日照時間(0.1時間)",
                   "全天日射量(MJ/㎡)":"全天日射量(0.01MJ/m2/h)","雲量":"雲量(10分比)"}

    def __init__(self, year, stn, data_dir_base='/mnt/d/dat/met/JMA_DataBase/whpView/wwwRoot/data/'):
        '''
        Constructor of class Whp, reading CSV file of f"{dirpath}{year}/{stn}_{year}.csv"
        and creating pandas.DataFrame compatible to GWO data.
        The data cleaning is made based on:
        https://www.data.jma.go.jp/obd/stats/data/mdrr/man/remark.html
        
        Parameters
        ----------
        data_dir_base(str): whpView data directory path, e.g., "/mnt/d/dat/met/JMA_DataBase/whpView/wwwRoot/data/"
        year(str or int): Target year
        stn(str): Target station name
        '''
        self._year = year
        self._stn = stn
        fi_path = f"{data_dir_base}{self._year}/{self._stn}_{self._year}.csv"
        print(f"Reading {fi_path}")
        self._df = pd.read_csv(fi_path, na_values=["×", "///", "#", ""])
        self._df.drop(["降雪(cm)","積雪(cm)","--","視程"], axis=1, inplace=True)
        #self._df.loc[:,"雲量"].replace({"0+":"0"}, inplace=True)
        self._df.replace({"0+":"0"}, inplace=True)
        self._df.replace({"10-":"10"}, inplace=True)
        self._df.replace(to_replace=r'\s*--\s*', value='0', regex=True, inplace=True)
        # Replace the last ")" and "]" in the string with ""
        # self.df.iloc[1:] = self.df.iloc[1:].applymap(lambda x: str(x).replace(")", "") if isinstance(x, str) else x)
        # ")" and "]" may be included in column names, so the above code is not applicable.
        self._df.iloc[1:] = self._df.iloc[1:].replace({r'\)|\]': ''}, regex=True)  

        column_name = "全天日射量(MJ/㎡)"
        #print(self._df.columns)
        if column_name in self._df.columns:
            print(f"Substituting NaN for 0 in {column_name}")
            zero_rows_list_radiation = self._df[self._df[column_name].isna()].index.tolist()
            self._df[column_name] = self._df[column_name].apply(lambda x: 0 if pd.isna(x) and self._df[column_name].notna().any() else x)

        column_name = "日照時間(h)"
        if column_name in self._df.columns:
            print(f"Substituting NaN for 0 in {column_name}")
            zero_rows_list_sunlight = self._df[self._df[column_name].isna()].index.tolist()
            self._df[column_name] = self._df[column_name].apply(lambda x: 0 if pd.isna(x) and self._df[column_name].notna().any() else x)
        
        self._df_org = self._df.copy()
        
        self._df.replace(Whp.wind_direction_jp2num, inplace=True)

        self._df[["観測所ID"]] -= 47000
        self._df[Whp.cols_10_times] = self._df[Whp.cols_10_times].astype('float64') * 10
        self._df[Whp.cols_100_times] = self._df[Whp.cols_100_times].astype('float64') * 100
        self._df[Whp.cols_100_times] = self._df[Whp.cols_100_times].round()

        ymd_cols = np.array([[ymd.split("/")[0], ymd.split("/")[1], ymd.split("/")[2]]
                             for ymd in self._df["日付"]], dtype='i2')
        self._df.insert(2, "年", ymd_cols.T[0])
        self._df.insert(3, "月", ymd_cols.T[1])
        self._df.insert(4, "日", ymd_cols.T[2])
        self._df.drop("日付", axis=1, inplace=True)
        self._df.rename(columns=Whp.rename_cols, inplace=True)
        self._df.insert(0, "観測所ID", self._df["ID1"])
        
        # Reindex columns with adding remark columns with fill_value=8
        self._df=self._df.reindex(columns=Whp.col_names_jp, fill_value=8)
        
        # Remark=2 for unobserved values (nighttime radiation and sunlight)
        num_rows = self._df.shape[0]
        if num_rows == len(zero_rows_list_radiation):
            self._df.loc[:, "ID13"] = 0
        else:
            self._df.loc[zero_rows_list_radiation,"ID13"] = 2
        if num_rows == len(zero_rows_list_sunlight):
            self._df.loc[:, "ID12"] = 0
        else:
            self._df.loc[zero_rows_list_sunlight,"ID12"] = 2
        self._df["現在天気"] = np.NaN
        self._df["ID10"] = 0
        
        #self._df = self._df.astype('int', errors='ignore')
        self._df = self._df.astype('Int32', errors='ignore')

    @property
    def df(self):
        '''
        Property returning pandas.DataFrame
        '''
        return self._df

    @property
    def df_org(self):
        '''
        Property returning pandas.DataFrame
        '''
        return self._df_org

    def to_csv(self, data_dir_base="/mnt/d/dat/met/JMA_DataBase/GWO/Hourly/", isfpath=False, **kwargs):
        '''
        Save the pandas.DataFrame as CSV file.
        
        Parameters
        ----------
        data_dir_base(str): CSV output directory base path when isfpath=False (default)
            or CSV output file path when isfpath=True
        isfpath(bool): True when data_dir_base is CSV file path
        **kwargs: kwargs transferred to pandas.DataFrame.to_csv()
        '''
        #fo_path = f"{data_dir_base}{self._stn}/{self._stn}{self._year}.csv"
        if isfpath:
            fo_path = f"{data_dir_base}"
        else:
            fo_path = f"{data_dir_base}{self._stn}/{self._stn}{self._year}.csv"
        self._df.to_csv(fo_path, columns=None, header=None, index=False, encoding='utf-8', **kwargs)
        cmd = ['nkf', '-w', '-Lu', '--overwrite', fo_path]  # Enforcing LF line feed
        subprocess.call(cmd)
        print(f"Saving {fo_path} in GWO format.")

if __name__ == '__main__':
    from metdata.whp import Whp, StationDict
    whp_dir_base = "/mnt/d/dat/met/JMA_DataBase/whpView/wwwRoot/data/"
    gwo_dir_base = "/mnt/d/dat/met/JMA_DataBase/GWO/Hourly/"
    stns=["Tokyo"]
    years=[2022]
    whp_data = StationDict()
    for stn in stns:
        for year in years:
            whp_data[stn, year] = Whp(year=year, stn=stn, data_dir_base=whp_dir_base)
            whp_data[stn, year].to_csv(data_dir_base=gwo_dir_base, isfpath=False)
