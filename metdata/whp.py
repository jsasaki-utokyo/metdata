# Processing whpview data
# Author: Jun Sasaki@UTokyo
# Coded on February 13, 2022  Updated on Februrary 14, 2022
# http://www.roy.hi-ho.ne.jp/ssai/mito_gis/whpview/index.html
# http://www.roy.hi-ho.ne.jp/ssai/mito_gis/

from metdata import gwo
import numpy as np
import pandas as pd
import subprocess
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location


class Whp(gwo.Hourly):
    '''
    Adjust the format of whpView hourly dataset to GWO hourly dataset used by gwo.py.
    There are inconsistencies in some columns, e.g., "現在天気" in GWO is missing.
    In such cases, their values are designated as NaN with Remark=0.
    Some columns that do not exist in GWO are deleted. 
    '''
    cols_10_times = ["現地気圧(hPa)","海面気圧(hPa)","降水量(mm)","気温(℃)","露点温度(℃)",
                     "蒸気圧(hPa)","風速(m/s)","日照時間(h)","全天日射量(MJ/㎡)"]
    
    rename_cols = {"名称":"観測所名","観測所ID":"ID1","現地気圧(hPa)":"現地気圧(0.1hPa)",
                   "海面気圧(hPa)":"海面気圧(0.1hPa)","降水量(mm)":"降水量(0.1mm/h)","気温(℃)":"気温(0.1degC)",
                   "露点温度(℃)":"露天温度(0.1degC)","蒸気圧(hPa)":"蒸気圧(0.1hPa)","湿度(%)":"相対湿度(%)",
                   "風速(m/s)":"風速(0.1m/s)","風向":"風向(1(NNE)～16(N))","日照時間(h)":"日照時間(0.1時間)",
                   "全天日射量(MJ/㎡)":"全天日射量(0.01MJ/m2/h)","雲量":"雲量(10分比)"}

    def __init__(self, dir_path="/mnt/d/dat/met/JMA_DataBase/whpView/wwwRoot/data/",
                 year="2021", stn="Tokyo"):
        '''
        Constructor of class Whp, reading CSV file of f"{dirpath}{year}/{stn}_{year}.csv"
        and creating pandas.DataFrame compatible to GWO data.
        The data cleaning is made based on:
        https://www.data.jma.go.jp/obd/stats/data/mdrr/man/remark.html
        
        Parameters
        ----------
        dir_path(str): whpView data directory path
        year(str): Year of data
        stn(str): Station of data
        '''
        self._year = year
        self._stn = stn
        print(self._year)
        print(self._stn)
        fi_path = f"{dir_path}{self._year}/{self._stn}_{self._year}.csv"
        print(fi_path)
        self._df = pd.read_csv(fi_path, na_values=["×", "///", "#", ""])
        self._df.drop(["降雪(cm)","積雪(cm)","--","視程"], axis=1, inplace=True)
        #self._df.loc[:,"雲量"].replace({"0+":"0"}, inplace=True)
        self._df.replace({"0+":"0"}, inplace=True)
        self._df.replace({"10-":"10"}, inplace=True)
        self._df.replace({"--":"0"}, inplace=True)

        #self._df = pd.read_csv(fi_path, na_values=["--"])
        self._df_org = self._df.copy()
        #self._df.fillna(0, inplace=True)
        #self._df.replace(gwo.Met.wind_direction_jp2num, inplace=True)
        self._df.replace(Whp.wind_direction_jp2num, inplace=True)

        self._df[["観測所ID"]] -= 47000
        self._df[Whp.cols_10_times] *= 10
        ymd_cols = np.array([[ymd.split("/")[0], ymd.split("/")[1], ymd.split("/")[2]]
                             for ymd in self._df["日付"]], dtype='i2')
        self._df.insert(2, "年", ymd_cols.T[0])
        self._df.insert(3, "月", ymd_cols.T[1])
        self._df.insert(4, "日", ymd_cols.T[2])
        self._df.drop("日付", axis=1, inplace=True)
        self._df.rename(columns=Whp.rename_cols, inplace=True)
        self._df.insert(0, "観測所ID", self._df["ID1"])
        self._df=self._df.reindex(columns=Whp.col_names_jp, fill_value=8)
        self._df["現在天気"] = np.NaN
        self._df["ID10"] = 0
        
        ## ここに全天日射量のmaskを作る
        ## self._df.loc[:,'全天日射量(0.01MJ/m2/h)'][mask].replace(NaN, 0, inplace=True)
        self._df = self._df.astype('int', errors='ignore')

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

    def to_csv(self, dir_path="/mnt/d/dat/met/JMA_DataBase/GWO/Hourly/",
               isfpath=False, **kwargs):
        '''
        Save the pandas.DataFrame as CSV file.

        Parameters
        ----------
        dir_path(str): CSV output directory path (default and isfpath=False)
                       or CSV output file path when isfpath=True
        isfpath(bool): True when dir_path is CSV file path
        **kwargs: kwargs transferred to pandas.DataFrame.to_csv()
        '''
        fo_path = f"{dir_path}{self._stn}/{self._stn}{self._year}.csv"
        if isfpath:
            fo_path = f"{dir_path}"
        self._df.to_csv(fo_path, columns=None, header=None, index=False,
                        encoding='utf-8', **kwargs)
        cmd = ['nkf', '-w', '-Lu', '--overwrite', fo_path]  # Enforcing LF line feed
        subprocess.call(cmd)
        print(f"Saving to {fo_path}")

if __name__ == '__main__':
    from metdata import whp
    dir_path = "/mnt/d/dat/met/JMA_DataBase/whpView/wwwRoot/data/"
    tokyo_2021 = whp.Whp(dir_path=dir_path, year="2021", stn="Tokyo")
    print(tokyo_2021.df.head())
    tokyo_2021.to_csv()
