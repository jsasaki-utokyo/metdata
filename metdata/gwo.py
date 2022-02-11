# Handling for Ground Weather Observations (GWO) data provided
# by the Japan Meteorological Business Support Center
# (JMBSC http://www.jmbsc.or.jp/en/index-e.html).
# Author: Jun Sasaki@UTokyo
# Coded on September 9, 2017  Updated on February 10, 2022
# The dataset was a commercial product in Japanese
# (see http://www.roy.hi-ho.ne.jp/ssai/mito_gis/ for details).
# This is a script to cut out the data, extract missing values and lines,
# edit the data, and plot the data.
# Since 1991, hourly data, including the global horizontal irradiance at some stations, 
# is available.
# Before 1991, only 3-hour data is available; the global horizontal irradiance,
# sunshine duration, and precipitation are not available.
# （財）気象業務支援センター「気象データベース」「地上気象」の切り出し，
# 欠損値・欠損行の抽出，編集，図化
# 1991年以後は毎時データ．一部観測所では全天日射量が存在．
# 1990年以前は3時間データ．全天日射量，日照時間，および降水量は存在しない．

import sys
#import os
import glob
import numpy as np
import math
import pandas as pd
import datetime
import subprocess
from pandas.tseries.offsets import Hour
from dateutil.parser import parse
#import json  # json cannot manipulate datetime
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.dates import date2num, YearLocator, MonthLocator, DayLocator, DateFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

get_ipython().run_line_magic('matplotlib', 'inline')


class Met:
    '''
    気象データベース・地上観測DVD，アメダスDVDの時系列データ抽出
    +--------+------------------------------------------------------------+
    |リマーク|            解                                  説          |
    +--------+------------------------------------------------------------+
    |   ０   |観測値が未作成の場合                                        |
    +--------+------------------------------------------------------------+
    |   １   |欠測                                                        |
    +--------+------------------------------------------------------------+
    |   ２   |観測していない場合                                          |
    +--------+------------------------------------------------------------+
    |   ３   |日の極値が真の値以下の場合，該当現象がない推定値の場合      |
    +--------+------------------------------------------------------------+
    |   ４   |日の極値が真の値以上の場合，該当現象がない地域気象観測データ|
    |        |を使用する場合                                              |
    +--------+------------------------------------------------------------+
    |   ５   |推定値が含まれる場合，または２４回平均値で欠測を含む場合    |
    +--------+------------------------------------------------------------+
    |   ６   |該当する現象がない場合（降水量，日照時間，降雪，積雪，最低海|
    |        |面気圧）                                                    |
    +--------+------------------------------------------------------------+
    |   ７   |日の極値の起時が前日の場合                                  |
    +--------+------------------------------------------------------------+
    |   ８   |正常な観測値                                                |
    +--------+------------------------------------------------------------+
    |   ９   |日の極値の起時が翌日の場合，または１９９０年までの８０型地上|
    |        |気象観測装置からの自動取得値                                |
    +--------+------------------------------------------------------------+

    Extraction of time series data from meteorological database:
        ground observation DVD, and AMeDAS DVD.
    +--------+------------------------------------------------------------+
    |Remarks |            Explanation                                     |
    +--------+------------------------------------------------------------+
    |   0    |Observation data has not been created.                      |
    +--------+------------------------------------------------------------+
    |   1    |Missing                                                     |
    +--------+------------------------------------------------------------+
    |   2    |No observation is made                                      |
    +--------+------------------------------------------------------------+
    |   3    |The extreme value of the day <= true value                  |
    |        |Estimated value with no relevant phenomenon                 |
    +--------+------------------------------------------------------------+
    |   4    |the extreme value of the day >= true value                  |
    |        |Meteorological data from regions with no relevant phenomena |
    +--------+------------------------------------------------------------+
    |   5    |Estimated values included, or                               |
    |        |Averaged value of 24 times including missing values         |
    +--------+------------------------------------------------------------+
    |   6    |No corresponding phenomenon (Precipitation, dailight hours, |
    |        |snowfall, snow accumulation, lowest sealevel pressure       |
    +--------+------------------------------------------------------------+
    |   7    |The starting time of the day's extreme value                |
    |        |is the previous day.                                        |
    +--------+------------------------------------------------------------+
    |   8    |Normal observation value                                    |
    +--------+------------------------------------------------------------+
    |   9    |The starting time of the day's extreme value is the next    |
    |        |day or automatic acquisition from Type 80 ground            |
    |        |meteorological instruments until 1990.                      |
    +--------+------------------------------------------------------------+
    '''

    rmk = {"0":"観測値が未作成の場合",
           "1": "欠測",
           "2":"観測していない場合",
           "3":"日の極値が真の値以下の場合，該当現象がない推定値の場合",
           "4":"日の極値が真の値以上の場合，該当現象がない地域気象観測データを使用する場合",
           "5":"推定値が含まれる場合，または２４回平均値で欠測を含む場合",
           "6":"該当する現象がない場合（降水量，日照時間，降雪，積雪，最低海面気圧）",
           "7":"日の極値の起時が前日の場合", "8":"正常な観測値",
           "9":"日の極値の起時が翌日の場合，または１９９０年までの８０型地上気象観測装置からの自動取得値"}
    # These elements are specified as missing values. 欠損値と判断するRMK値
    nan_0_1_RMK = [0, 1, 2]

    def __init__(self, datetime_ini, datetime_end, stn, dirpath):
        '''
        Constructor for setting datetime period and dataset directory path
        
        Parameters
        ----------
        datetime_ini (str) : Initial datetime
        datetime_end (str) : End datetime
        stn (str) : Station name
        dirpath (str) : Directory path for JMA Databse
        '''
        self.datetime_ini = parse(datetime_ini)
        self.datetime_end = parse(datetime_end)
        # print("Initial datetime = ", self.datetime_ini)
        # print("End datetime = ", self.datetime_end)
        print(f"Start datetime = {self.datetime_ini}")
        print(f"End datetime = {self.datetime_end}")
        self.stn = stn
        self.dirpath = dirpath

    def set_missing_values(self, df, rmk_cols, rmk_nans):
        '''
        DataFrame dfを入力し，RMK列が欠損値条件を満たす行の一つ前の列の値をnp.nanに置換する
        
        Parameters
        ----------
        df (pandas.DataFrame) : Extracted dataset
        rmk_cols (str list) : List of names for remark columns
        rmk_nans (str list) : List of names for remark columns containing missing values
        
        Returns
        -------
        df (pandas.DataFrame) : Updated df
        '''

        ### rmk_cols = [col for col in df.columns if 'RMK' in col]  ## RMK列名のリスト
        for rmk_col in rmk_cols:
            for rmk_nan in rmk_nans:
                idx = df.columns.get_loc(rmk_col) - 1  ## RMKに対応する値の列インデックス
                df.iloc[:, idx].mask(df[rmk_col] == rmk_nan, np.nan, inplace=True)
        return df


class Hourly(Met):
    '''
    Extraction of hourly data from meteorological database ground observation DVDs
        (3-hour intervals before 1990)
    気象データベース・地上観測DVDの時別値（1990年以前は3時間間隔）データ抽出
    The directory name is suppoed to be stn + "/"
        while file name supposed to be stn + year + ".csv"
    '''
    col_names_jp = ["観測所ID","観測所名","ID1","年","月","日","時",
                    "現地気圧(0.1hPa)","ID2","海面気圧(0.1hPa)","ID3",
                    "気温(0.1degC)","ID4","蒸気圧(0.1hPa)","ID5",
                    "相対湿度(%)","ID6","風向(1(NNE)～16(N))","ID7","風速(0.1m/s)","ID8",
                    "雲量(10分比)","ID9","現在天気","ID10","露天温度(0.1degC)","ID11",
                    "日照時間(0.1時間)","ID12","全天日射量(0.01MJ/m2/h)","ID13",
                    "降水量(0.1mm/h)","ID14"]
    col_names_en = ["Station ID","Station name","ID1","Year","Month","Day","Hour",
                    "Local pressure (0.1hPa)","ID2","Sea surface pressure (0.1hPa)","ID3",
                    "Temperature (0.1degC)","ID4","Vapor pressure (0.1hPa)","ID5",
                    "Relative humidity (%)","ID6","Wind direction (1(NNE)～16(N))","ID7",
                    "Wind speed (0.1m/s)","ID8",
                    "Cloud cover (0-10)","ID9","Current weather","ID10",
                    "Dewpoint temperature (0.1degC)","ID11","Daylight time (0.1h)",
                    "ID12","Global horizontal irradiance (0.01MJ/m2/h)","ID13",
                    "Precipitation (0.1mm/h)","ID14"]
    col_names = ["KanID","Kname","KanID_1","YYYY","MM","DD","HH","lhpa","lhpaRMK",
                 "shpa","shpaRMK","kion","kionRMK","stem","stemRMK","rhum","rhumRMK",
                 "muki","mukiRMK","sped","spedRMK","clod","clodRMK","tnki","tnkiRMK",
                 "humd","humdRMK","lght","lghtRMK","slht","slhtRMK","kous","kousRMK"]
    col_items_jp = ["現地気圧(hPa)","海面気圧(hPa)",
                    "気温(degC)","蒸気圧(hPa)",
                    "相対湿度(0-1)","風向(0-360)","風速(m/s)",
                    "雲量(0-1)","現在天気","露天温度(degC)",
                    "日照時間(時間)","全天日射量(W/m2)",
                    "降水量(mm/h)","風速u(m/s)","風速v(m/s)"]
    col_items_en = ["Local pressure (hPa)","Sea surface pressure (hPa)",
                    "Temperature (degC)","Vapor pressure (hPa)",
                    "Relative humidity (0-1)","Wind direction (degree)","Wind speed (m/s)",
                    "Cloud cover (0-1)","Current weather","Dewpoint temperature (degC)",
                    "Daylight hours (h)","Global horizontal irradiance (W/m2)",
                    "Precipitation (mm/h)","x-component of wind vector(m/s)",
                    "y-component of wind vector(m/s)"]
    col_items =  ["lhpa",
                  "shpa","kion","stem","rhum",
                  "muki","sped","clod","tnki",
                  "humd","lght","slht","kous","u","v"]
    col_rmks = ["lhpaRMK","shpaRMK","kionRMK","stemRMK","rhumRMK","mukiRMK","spedRMK",
                "clodRMK","tnkiRMK","humdRMK","lghtRMK","slhtRMK","kousRMK"] 

    def __init__(self, datetime_ini = "2014-1-10 15:00:00", datetime_end = "2014-6-1 00:00:00",
                 stn = "Tokyo", dirpath = "D:/dat/met/JMA_DataBase/GWO/"):
        super().__init__(datetime_ini, datetime_end, stn, dirpath)
        '''
        Constructor for setting parameters for extracting dataset
        
        Parameters
        ----------
        datetime_ini (str) : Initial datetime
        datetime_end (str) : End datetime
        stn (str) : Station name
        dirpath (str) : directory path for dataset
        '''
        self.names_jp = Hourly.col_names_jp
        self.names = Hourly.col_names
        self.items_jp = Hourly.col_items_jp
        self.items = Hourly.col_items

        ## the values of rmk to be set as NaN (RMK=0, 1, 2)
        ## In the case of lght and slht, RMK is 2 during nighttime;
        ## thus, RMK=2 should not be designated as missing value.
        ## clod and tnki are provided with 3-hour interval even after 1990;
        ## thus, their RMK=2 should be designated as missing value.
        ## lghtとslhtの夜間はRMK=2なので，RMK=2を欠損値としない
        ## clodとtnkiは1991年以降も毎3時間データのため，RMK=2を欠損値に指定する
        self.rmk_nan01 = ["0", "1"]
        self.rmk_nan = ["0", "1", "2"]
        self.__df, self.__df_interp, self.__df_interp_1H = self._create_df()

    ## The name of the property is changed from the name in create_df to make it easy
    ## to call the normally used self.__df_interp_1H as self.df.
    ## propertyの名称をcreate_dfでの名称から変更している．
    ## 通常使うself.__df_interp_1Hをself.dfと簡単に呼べる様にするため
    @property
    def df_org(self):
        '''
        pandas.DataFrame at the original time intervals
         (3-hour before 1991 and 1-hour after 1990)
        without interpolateing missing values
        When there are missing rows, the rows are inserted with missing values.
        Note: The original data contains missing rows corresponding to unobserved.
        元の時間間隔で欠損値を補間していないpandas.DataFrame
        存在しないdatetime行は欠損値として挿入済
        元のデータでは，未観測の年月日時は欠損行となっていることに注意
        '''
        return self.__df

    @property
    def df_interp(self):
        '''
        pandas.DataFrame at the original time intervals with interpolating
        missing rows and missing values as much as possible
        元の時間間隔で欠損値を可能な限り補間したpandas.DataFrame
        '''
        return self.__df_interp

    @property
    def df(self):
        '''
        pandas.DataFrame at 1-hour intervals with interpolating
        missing rows and missing values as much as possible
        1時間間隔で欠損値を可能な限り補間したpandas.DataFrame
        '''
        return self.__df_interp_1H

    @property
    def df_missing_rows(self):
        '''
        pandas.DataFrame extracting datetime rows that do not exist in df_interp
        This property is useful to check how the missing rows were interpolated.
        元の時間間隔で欠損値を可能な限り補間したDataFrameにおいて，
        df_interpに存在しないdatetime行を抽出したpandas.DataFrame
        Missing rowsがどのように補間されたか確認するために活用
        '''
        masked = np.isin(self.__df_interp.index, self.__index_masked)

        return self.__df_interp[masked]

    def to_csv(self, df, fo_path='./df.csv'):
        '''
        Export df to a CSV file.
                   
        Parameters
        ----------
        df (pandas.DataFrame) : 
        fo_path (str) : CSV file path
        '''
        ## Shift-JIS is used on Windows, thus, UTF-8 should be specified.
        df.to_csv(fo_path, encoding='utf-8', )
        ## Force conversion of line feed code to Linux LF.
        ## 改行コードをLinuxのLFに強制変換（Linuxとの互換性維持）
        cmd = 'nkf -w -Lu --overwrite ' + fo_path
        subprocess.call(cmd)

    def read_csv(self, fi_path, **kwargs):
        '''
        Read a CSV file into pandas.DataFrame.
        Useful for reading a CSV file exported by to_csv().
                
        Parameters
        ----------
        fi_path (str) : CSV file path
        
        Returns
        -------
        pandas.DataFrame
        '''
        try:
            return pd.read_csv(fi_path, index_col=0, parse_dates=True, **kwargs)
        except:
            print(f"Error in reading csv of {fi_path}")


    ############################################################
    ## The following are private methods.
    ############################################################

    def _create_df(self, interp=True):
        '''
        Create pandas.DataFrame of GWO.
        
        Parameters
        ----------
        interp (bool) : True when interpoting missing values
        '''
        start = self.datetime_ini
        end   = self.datetime_end
        if start >= end:
            print("ERROR: start >= end")
            sys.exit()
        Ys, Ye = int(start.strftime("%Y")), int(end.strftime("%Y"))
        fdir = self.dirpath + self.stn + "/"  # data dirpath
        ## Check whether Ys-1 exists.
        fstart = glob.glob(fdir + self.stn + str(Ys-1) + ".csv")
        if len(fstart) == 1:
            Ys += -1
        else:
            fstart = glob.glob(fdir + self.stn + str(Ys) + ".csv")
        if len(fstart) != 1:
            print(f'ERROR: {fstart} does not exist or has more than one file.')
            sys.exit()
        ## Check whether Ye+1 exists.
        fend = glob.glob(fdir + self.stn + str(Ye+1) + ".csv")
        if len(fend) == 1:
            Ye += 1
        else:
            fend = glob.glob(fdir + self.stn + str(Ye) + ".csv")
        if len(fend) != 1:
            print(f'ERROR: {fend} does not exist or has more than one file.')
            sys.exit()

        ## List of a DataFrame for each year
        tsa = []
        ## List of a datetime index for each year for missing rows
        tsi_masked = []
        
        fyears = list(range(Ys, Ye+1))  # from Ys to Ye

        ## Reading GWO csv files and creating DataFrame considering missing values
        ## Specifying a missing value for each column
        ## Also creating DataFrame without considering missing values
        ## カラム毎に欠損値を指定する
        ## 欠損値を考慮しないデータフレームも併せて作成する
        self.na_values = {"lhpaRMK":self.rmk_nan, "shpaRMK":self.rmk_nan,
                          "kionRMK":self.rmk_nan, "stemRMK":self.rmk_nan,
                          "rhumRMK":self.rmk_nan, "mukiRMK":self.rmk_nan,
                          "spedRMK":self.rmk_nan, "clodRMK":self.rmk_nan,
                          "tnkiRMK":self.rmk_nan, "humdRMK":self.rmk_nan,
                          "lghtRMK":self.rmk_nan01, "slhtRMK":self.rmk_nan01,
                          "kousRMK":self.rmk_nan}

        for year in fyears:
            file = "{}{}{}.csv".format(fdir, self.stn, str(year))
            print(f"Reading from {file}")
            
            df_tmp = pd.read_csv(file, header = None, names = self.names, parse_dates=[[3,4,5]])
            ## YYYY-MM-DD and HH (hour) columns are combined into a datetime index.
            df_tmp.index = [x + y * Hour() for x,y in zip(df_tmp['YYYY_MM_DD'],df_tmp['HH'])]
            df_tmp.drop("YYYY_MM_DD", axis=1, inplace=True)
            df_tmp.drop("HH", axis=1, inplace=True)

            ## Check and fill missing rows with missing_value=9999
            ## to avoid RMKs dtype being float (should be integer).
            ## If replaced with NaN, the dtype of integer is forced to float.
            if interp:
                df_tmp, index_masked_tmp = self._check_fill_missing_rows(df_tmp, year)
                df_tmp = self._check_fill_missing_values(df_tmp)
                tsi_masked.append(index_masked_tmp)
            tsa.append(df_tmp)
          
        ## Concatinate DataFrame in each year and slice with [start:end].
        df = pd.concat(tsa)[start:end]
        
        if interp:
            try:
                self.__index_masked = pd.concat(tsi_masked)[start:end]
            except:
                self.__index_masked = None
            df = self._unit_conversion(df)
            df_interp, df_interp_1H = self._df_interp(df)
            return df, df_interp, df_interp_1H
        else:
            print('Original data with no editing')
            return df

    def _check_fill_missing_rows(self, df, year):
        '''
        There may be missing rows (datetime index) in GWO when unobserved.
        Insert such rows with missing values of 9999.
        NaN was not specified to avoid the dtype of RMK's integer being forced to float.
            
        Parameters
        ----------
        df (pandas.DataFrame) : One year dataset at one station
        year (int) : Year of the dataset
            
        Returns
        -------
        df (pandas.DataFrame) : Updated df with complete datetime rows
        index_masked (ndarray)
            
        Note
        ----
        - df.resample('H').asfreq() should not be applied
          as the initial or the end row may be missing.
        - Hours in pandas must be between 00 and 23
          while those in GWO are between 01 and 24.

        '''       
        ## Get the columns and their values to be set in missing rows
        replaced_vals = df[['KanID', 'Kname', 'KanID_1']].drop_duplicates().values[0]
        dict_for_fill = dict(zip(['KanID', 'Kname', 'KanID_1'], replaced_vals))
        ## Create a complete datetime index for the year dependent on the datetime interval.
        ## The intervals are 3-hour (-1990) and 1-hour (1991-).
        if year < 1991:  # 3-hour intervals from 03 am to 00 am, Not 03 to 24
            datetime_ini = f"{year}-01-01 03:00:00"
            datetime_end = f"{year+1}-01-01 00:00:00"
            complete_index = pd.date_range(datetime_ini, datetime_end, freq='3H')    
        else:            # One-hour intervals from 01 am to 00 am, Not 01 to 24
            #datetime_ini = "{}-01-01 01:00:00".format(year)
            datetime_ini = f"{year}-01-01 01:00:00"
            datetime_end = f"{year+1}-01-01 00:00:00"
            complete_index = pd.date_range(datetime_ini, datetime_end, freq='H')

        ## https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html 
        masked = np.logical_not(np.isin(complete_index, df.index))
                
        ## If missing rows exist:
        if np.count_nonzero(masked) > 0:
            df = df.reindex(complete_index, fill_value = 9999)
                
            rmk_cols = [col for col in df.columns if 'RMK' in col]
            for rmk_col in rmk_cols:
                ## Find a value_col index corresponding to rmk_col
                idx = df.columns.get_loc(rmk_col) - 1
                df.iloc[:,idx].mask(df[rmk_col] == 9999, np.nan, inplace=True)
                df.loc[:, rmk_col].mask(df[rmk_col] == 9999, 0, inplace=True)

            df.loc[:,'KanID'].replace(9999, dict_for_fill['KanID'], inplace=True)
            df.loc[:,'Kname'].replace(9999, dict_for_fill['Kname'], inplace=True)
            df.loc[:,'KanID_1'].replace(9999, dict_for_fill['KanID_1'], inplace=True)
            ## Do not use the following, which does not work. Use slicing above.
            ## df[['KanID']].replace(9999, dict_for_fill['KanID'], inplace=True)
            ## df[['Kname']].replace(9999, dict_for_fill['Kname'], inplace=True)
            ## df[['KanID_1']].replace(9999, dict_for_fill['KanID_1'], inplace=True)
            print("Missing rows are filled with data-val=np.nan and rmk=0")
            print(f"df[masked]={df[masked]}")
            index_masked = df.index.to_series()[masked]
        else:
            print("Successful with no missing row.")
            index_masked = None

        return df, index_masked

    def _check_fill_missing_values(self, df):
        '''
        Fill missing values corresponding to RMKs in pandas.DataFrame.
        
        Parameters
        ----------
        df (pandas.DataFrame)
        
        Returns
        -------
        df (pandas.DataFrame) : Updated df
        '''
        ## rmk_cols = [col for col in df.columns if 'RMK' in col]
        for key in self.na_values:
            ## Find a value_col index corresponding to rmk_col
            idx = df.columns.get_loc(key) - 1  
            ## In a column, set the row that matches the list element to True.
            ## カラムにおいて，リストの要素にマッチした行をTrueにする
            df.iloc[:,idx].mask(df[key].isin(self.na_values[key]), inplace=True)
        return df

    def _unit_conversion(self, df):
        '''
        Convert units. Should be edited appropriately.
        The following for TEEM (Lab's model).          
    
        Parameters
        ----------
        df (pandas.DataFrame)
    
        Returns
        df (pandas.DataFrame) : Updated df
        '''
        df['lhpa']=df['lhpa']/1.0e1  # [0.1hPa] -> [hPa]  現地気圧  local pressure
        df['shpa']=df['shpa']/1.0e1  # [0.1hPa] -> [hPa]  海面気圧  sealevel pressure
        df['kion']=df['kion']/1.0e1  # [0.1degC] -> [degC]  気温  air temperature
        df['stem']=df['stem']/1.0e1  # [0.1hPa] -> [hPa]  蒸気圧  vapor pressure
        df['rhum']=df['rhum']/1.0e2  # [%] -> [0-1]  相対湿度  relative humidity
        ## Wind direction [0-16] -> [deg]  0=N/A, 1=NNE, .., 8=S, .., 16=N
        df['muki']=-90.0 - df['muki'] * 22.5
        ## Wind direction [0-360]: Anticolockwise angle with respect to x (W-E) axis.
        df['muki']=df['muki'] % 360.0
        df['sped']=df['sped']/1.0e1  # [0.1m/s] -> [m/s]  風速  wind speed
        df['clod']=df['clod']/1.0e1  # [0-10] -> [0-1]  雲量  cloud cover
        df['humd']=df['humd']/1.0e1  # [0.1degC] -> [degC]  露点温度  dew-point temperature
        df['lght']=df['lght']/1.0e1  # [0.1h] -> [h]  日照時間  daylight hours
        ## Global horizontal irradiance [0.01MJ/m2/h] -> [J/m2/s] = [W/m2]  短波放射
        df['slht']=df['slht']*1.0e4/3.6e3
        ## wind vector (u,v)
        rad = df["muki"].values * 2 * np.pi / 360.0
        (u, v) = df["sped"].values * (np.cos(rad), np.sin(rad))
        df["u"] = u
        df["v"] = v
        return df

    def _df_interp(self, df):
        '''
        Check the RMK to find the missing values, keep the original integers in the RMK,
        set the corresponding variable values to the missing values,
        and interpolate them appropriately.
        Create a DataFrame df_interp that interpolates them appropriately.
        Further, reindex all df_interp to 1-hour intervals and create df_interp_1H
        with the missing values filled.
        RMKをチェックして欠損値を見つけ，RMKは元の整数を保持したまま，
        対応する変数値を欠損値にし，それを適切に補間したDataFrame df_interpを作る．
        さらに，df_interpをすべて1時間間隔にreindexし，欠損値を埋めたdf_interp_1Hを作る
        '''
        ## Create df_interp with interpolated missing values
        ## 欠損値を内挿したdf_interpを作る
        df_interp = df.interpolate(method='time').copy()

        ## Resample 3-hour intervals before 1990 to 1-hour intervals.
        ## Create a 1-hour interval index.
        ## 1990年以前の3時間間隔を1時間間隔にする
        ## 1時間間隔のインデックスを作る
        new_index = pd.date_range(self.datetime_ini, self.datetime_end, freq='1H')
        ## List of column names to apply ffill to
        ## ffillを適用するカラム名のリスト
        cols_ffill=['KanID', 'Kname', 'KanID_1', 'lhpaRMK', 'shpaRMK', 'kionRMK',
                    'stemRMK', 'rhumRMK', 'mukiRMK', 'spedRMK', 'clodRMK', 'tnkiRMK',
                    'humdRMK', 'lghtRMK', 'slhtRMK', 'kousRMK']
        ## Define a lambda function `dellist` to remove columns
        ## to which ffill is to be applied from the entire list of columns.
        ## カラム全体のリストからffillを適用するカラムを削除するラムダ関数dellistを定義

        dellist = lambda items, sublist: [item for item in items if item not in sublist]
        ## Creatie a list of column names to which interpolation should be applied
        ## instead of applying ffill
        ## ffillではなく，内挿を適用するカラム名のリストをつくる
        cols_interp=dellist(df_interp.columns, cols_ffill)

        ## Apply indexes at 1-hour intervals and perform completion on columns
        ## to which ffill should be applied.
        ## Put the result into df_ffill.
        ## 1時間間隔のインデックスを適用し，ffillを適用すべきカラムを対象に補完実行
        ## 結果をdf_ffillに入れる
        df_ffill = df_interp.reindex(new_index).loc[:, cols_ffill].fillna(method='ffill')

        ## Apply indexes at one-hour intervals and perform interpolation for columns
        ## that should be interpolated in time.
        ## Put the result into df_interp_1H.
        ## 1時間間隔のインデックスを適用し，時間内挿すべきカラムを対象に内挿実行
        ## 結果をdf_interp_1Hに入れる
        df_interp_1H = df_interp.reindex(new_index).loc[:, cols_interp].interpolate(
            method='time').copy()
        ### df_ffillとdf_interpを連結し，カラムの順序をdf1と同じとしたデータフレームdfを作る
        ### これで一応完成だが，1990年以前の全天日射量，日照時間，降水量への対応を今後進める
        df_interp_1H = pd.concat([df_ffill, df_interp_1H], axis=1)[df_interp.columns]

        return df_interp, df_interp_1H


class Check(Hourly):
    '''
    Find missing values using RMK values; if found, set corresponding values as NaN.
    '''
    def __init__(self, datetime_ini = "2014-1-1 15:00:00", datetime_end = "2014-6-1 00:00:00",
                 stn = "Tokyo", dirpath = "../GWO/Hourly/"):
        ## Class Check inherits Class Met.
        Met.__init__(self, datetime_ini, datetime_end, stn, dirpath)
        self.names_jp = Hourly.col_names_jp
        self.names = Hourly.col_names
        self.items_jp = Hourly.col_items_jp
        self.items = Hourly.col_items

        ## The values of rmk to be set as NaN (RMK=0, 1, 2)
        ## Since RMK=2 for nighttime lght and slht, RMK=2 is not a missing value.
        ## lghtとslhtの夜間はRMK=2なので，RMK=2を欠損値としない
        self.rmk_nan01 = ["0", "1"]
        ## The data of clod and tnki are 3-hour intervals; thus, RMK=2 (unobserved)
        ## should be designated as missing values.
        ## clodとtnkiは3時間間隔データのため，観測なしのRMK=2を欠損値に指定する
        self.rmk_nan = ["0", "1", "2"]
        ## Be sure to set the following to false.
        ## ここは必ずFalseにする
        self.__df = super()._create_df(interp=False)

    @property
    def df(self):
        return self.__df


## The following is under construction.
class Daily(Hourly):
    '''
    Extraction of daily data from meteorological database DVD
    for ground weather observation (GWO)
    Note: Units for daily values of global horizontal irradiance are:
        1 cal/cm2 between 1961 and 1980; 0.1 MJ/m2 after 1980.
    Database directory path is suppoed to be f"{stn}/".
    Data file name is supposed to be f"{stn}{year}.csv".
    気象データベース・地上観測DVDの日別値データ抽出
    注意：全天日射量日別値の単位：1961-1980: 1cal/cm2，1981以降: 0.1MJ/m2
    '''
    def __init__(self, datetime_ini = "2014-1-10 15:00:00", datetime_end = "2014-6-1 00:00:00",
                 stn = "Tokyo", dirpath = "../../../met/GWO/Daily/"):
        super().__init__(datetime_ini, datetime_end, stn, dirpath)
        self.names_jp = ["観測所ID","観測所名","ID1","年","月","日",
                         "平均現地気圧(0.1hPa)","ID2","平均海面気圧(0.1hPa)","ID3",
                         "最低海面気圧","ID4","平均気温(0.1degC)","ID5",
                         "最高気温(0.1degC)","ID6","最低気温(0.1degC)","ID7",
                         "平均蒸気圧(0.1hPa)","ID8","平均相対湿度(%)", "ID9",
                         "最小相対湿度(%)", "ID10","平均風速(0.1m/s)","ID11",
                         "最大風速(0.1m/s)","ID12","最大風速風向(1(NNE)～16(N))","ID13",
                         "最大瞬間風速(0.1m/s)","ID14", "最大瞬間風向(1(NNE)～16(N))", "ID15",
                         "平均雲量(0.1)","ID16", "日照時間(0.1時間)", "ID17",
                         "全天日射量(0.1MJ/m2)", "ID18","蒸発量(0.1mm)", "ID19",
                         "日降水量(0.1mm)", "ID20" , "最大1時間降水量(0.1mm)", "ID21",
                         "最大10分間降水量(0.1mm)", "ID22", "降雪の深さ日合計(cm)", "ID23",
                         "日最深積雪(cm)","ID24","天気概況符号1","ID25","天気概況符号2","ID26",
                         "大気現象コード1","大気現象コード1","大気現象コード2",
                         "大気現象コード3", "大気現象コード4", "大気現象コード5",
                         "降水強風時間", "ID27"]

        self.names = ["KanID","Kname","KanID_1","YYYY","MM","DD","avrLhpa","avrLhpaRMK",
                      "avrShpa","avrShpaRMK","minShpa","minShpaRMK","avrKion", "avrKionRMK",
                      "maxKion","maxKionRMK","minKion","minKionRMK","avrStem","avrStemRMK",
                      "avrRhum","avrRhumRMK","minRhum","minRhumRMK", "avrSped", "avrSpedRMK",
                      "maxSped","maxSpedRMK","maxMuki","maxMukiRMK","maxSSpd", "maxSSpdRMK",
                      "maxSMuk","maxSMukRMK","avrClod","avrClodRMK","daylght","daylghtRMK",
                      "sunlght","sunlghtRMK","amtEva","amtEvaRMK","dayPrec","dayPrecRMK",
                      "maxHPrc","maxHPrcRMK","maxMPrc","maxMPrcRMK","talSnow","talSnowRMK",
                      "daySnow","daySnowRMK","tenki1","tenki1RMK","tenki2","tenki2RMK",
                      "apCode1","apCode2","apCode3","apCode4","apCode5","strgTim","strgTimRMK"]
                      
        self.items_jp = ["現地気圧(hPa)","海面気圧(hPa)","気温(degC)","蒸気圧(hPa)",
                         "相対湿度(0-1)","風向(0-360)","風速(m/s)","雲量(0-1)","現在天気",
                         "露天温度(degC)","日照時間(時間)","全天日射量(W/m2)","降水量(mm/h)",
                         "風速u(m/s)","風速v(m/s)"]
        self.items = ["lhpa","shpa","kion","stem","rhum","muki","sped","clod","tnki",
                      "humd","lght","slht","kous","u","v"]
        ## the values of rmk to be set as NaN (RMK=0, 1, 2)
        ## Since RMK=2 for nighttime lght and slht, RMK=2 is not a missing value.
        ## lghtとslhtの夜間はRMK=2なので，RMK=2を欠損値としない
        self.rmk_nan01 = ["0", "1"]
        ## should be designated as missing values.
        ## clodとtnkiは3時間間隔データのため，観測なしのRMK=2を欠損値に指定する
        self.rmk_nan = ["0", "1", "2"]
        ## Create pandas.DataFrame by _create_df()
        ## _create_df()でDataFrameを作る
        self.__df_org, self.__df = self._create_df()

    ## Define properties for explicit and easy access to the DataFrames.
    ## DataFrameへのアクセスを明確かつ簡単にするため，propertyを定義
    @property
    def df_org(self):
        '''
        Accessing a DataFrame without missing value handling
        欠損値未処理のDataFrameにアクセスする
        '''
        return self.__df_org

    @property
    def df(self):
        '''
        Accessing a DataFrame with missing value handling
        欠損値処理したDataFrameにアクセスする
        '''
        return self.__df

    ## debug Class Met と同一のmethodなら削除する
    def to_csv(self, df, fo_path='./df.csv'):
        '''
        Export DataFrame df to a CSV file.
                   
        Parameters
        ----------
        df (pandas.DataFrame) : 
        fo_path (str) : CSV file path
        '''
        ## Shift-JIS is used on Windows, thus, UTF-8 should be specified.
        df.to_csv(fo_path, encoding='utf-8', )
        ## Force conversion of line feed code to Linux LF.
        ## 改行コードをLinuxのLFに強制変換（Linuxとの互換性維持）
        cmd = 'nkf -w -Lu --overwrite ' + fo_path
        subprocess.call(cmd)

    ## debug Class Met と同一のmethodなら削除する
    def read_csv(self, fi_path, **kwargs):
        '''
        Read a CSV file into pandas.DataFrame.
        Useful for reading a CSV file exported by to_csv().
                
        Parameters
        ----------
        fi_path (str) : CSV file path
        
        Returns
        -------
        pandas.DataFrame

        '''
        try:
            return pd.read_csv(fi_path, index_col=0, parse_dates=True, **kwargs)
        except:
            print(f"Error in reading csv of {fi_path}")


    ########################################################
    ## The following are hidden methods.
    ########################################################

    def _unit_conversion(self, df):
        '''DataFrameを受け取り，カラムの単位を変換する．風速ベクトルを定義する．'''
        df['avrLhpa']=df['avrLhpa']/1.0e1  # [0.1hPa] -> [hPa]
        df['avrShpa']=df['avrShpa']/1.0e1  # [0.1hPa] -> [hPa]
        df['minShpa']=df['minShpa']/1.0e1  # [0.1hPa] -> [hPa]
        df['avrKion']=df['avrKion']/1.0e1  # [0.1degC] -> [degC]
        df['maxKion']=df['maxKion']/1.0e1  # [0.1degC] -> [degC]
        df['minKion']=df['minKion']/1.0e1  # [0.1degC] -> [degC]
        df['avrStem']=df['avrStem']/1.0e1  # [0.1hPa] -> [hPa]
        df['avrRhum']=df['avrRhum']/1.0e2  # [%] -> [0-1]
        df['avrSped']=df['avrSped']/1.0e1  # [0.1m/s] -> [m/s]
        df['maxSped']=df['maxSped']/1.0e1  # [0.1m/s] -> [m/s]
        ## [0-16] -> [deg]  0=NA, 1=NNE, .., 8=S, ..., 16=N
        df['maxMuki']=-90.0 - df['maxMuki'] * 22.5
        df['maxMuki']=df['maxMuki'] % 360.0  # 0-360
        df['maxSSpd']=df['maxSSpd']/1.0e1  # [0.1m/s] -> [m/s]
        ##  [0-16] -> [deg]  0=NA, 1=NNE, .., 8=S, ..., 16=N
        df['maxSMuk']=-90.0 - df['maxSMuk'] * 22.5
        df['maxSMuk']=df['maxSMuk'] % 360.0  # 0-360
        df['avrClod']=df['avrClod']/1.0e1  # [0-10] -> [0-1]
        df['daylght']=df['daylght']/1.0e1  # 0.1h -> h
        ## Units for global horizontal irradiance:
        ##     1 cal/cm2 between 1961 and 1980; 0.1MJ/m2 after 1980
        ## 全天日射量日別値の単位について：1961-1980は1cal/cm2，1981以降は0.1MJ/m2
        ## 左辺は df.loc[,] とする必要がある
        ## [0.1MJ/m2/day] -> [J/m2/s] = [W/m2]
        df.loc['1981':, 'sunlght']=df['1981':]['sunlght']*1.0e5/3.6e3/24.0
        ## [1calcm2/day] -> [J/m2/s] = [W/m2]
        df.loc[:'1980', 'sunlght']=df[:'1980']['sunlght']*4.2*1.0e4/3.6e3/24.0
        df['amtEva']=df['amtEva']/1.0e1    # 0.1mm -> 1 mm
        df['dayPrec']=df['dayPrec']/1.0e1  # 0.1mm -> 1mm
        df['maxHPrc']=df['maxHPrc']/1.0e1  # 0.1mm -> 1mm
        df['maxMPrc']=df['maxMPrc']/1.0e1  # 0.1mm -> 1mm
        df['talSnow']=df['talSnow']        # cm
        df['daySnow']=df['daySnow']        # cm           
        ## wind vector (u,v)
        rad = df["maxMuki"].values * 2 * np.pi / 360.0
        (umax, vmax) = df["maxSped"].values * (np.cos(rad), np.sin(rad))
        df["umax"] = umax
        df["vmax"] = vmax
        (u, v) = df["avrSped"].values * (np.cos(rad), np.sin(rad))
        df["u"] = u
        df["v"] = v

        return df

    ## debug Class Met のmethodと同一なら削除
    def _create_df(self):
        start = self.datetime_ini
        end   = self.datetime_end
        ## Ys, Yeの妥当性と読込みcsvファイルの存在チェック
        if start >= end:
            print("ERROR: start >= end is incorrect.")
            sys.exit()
        Ys, Ye = int(start.strftime("%Y")), int(end.strftime("%Y"))
        fdir = self.dirpath + self.stn + "/"  # data dirpath
        print("Data directory path = ", fdir)
        fstart = glob.glob(fdir + self.stn + str(Ys) + ".csv")
        if len(fstart) != 1:
            print(fstart)
            print('ERROR: fstart file does not exist or has more than one file.')
            sys.exit()
        fend = glob.glob(fdir + self.stn + str(Ye) + ".csv")
        if len(fend) != 1:
            print('fend= ', fend)
            print('ERROR: fend file does not exist or has more than one file.')
            sys.exit()

        tsa = []  ### RMKの欠損値を考慮しない，オリジナルと同一
        fyears = list(range(Ys, Ye+1)) # from Ys to Ye

        for year in fyears:
            file = fdir + self.stn + str(year) + ".csv"
            print("Reading csv file of ", file)
            tsa.append(pd.read_csv(file, header = None, names = self.names,
                                   parse_dates=[[3,4,5]]))

        def create_df(tsa):
            '''
            Create df from tsa
            '''
            df = pd.concat(tsa)
            # df.index = [x + y * Hour() for x,y in zip(df['YYYY_MM_DD'],df['HH'])]
            df.index = df['YYYY_MM_DD']
            df.drop("YYYY_MM_DD", axis=1, inplace=True)
            # df.drop("HH", axis=1, inplace=True)
            df=df[start:end]
            return df
        df_org = create_df(tsa)  # 欠損値を無視した，元データと同じDataFrame

        df_org = self._unit_conversion(df_org)
        
        ## 欠損値を考慮したDataFrame
        df = df_org.copy()  ### df = df_org はcopyではない！
        rmk_cols = [col for col in df.columns if 'RMK' in col]  # RMKカラムのリストを作成
        missing_rmk = [0, 1]
        df = super().set_missing_values(df, rmk_cols, missing_rmk)

        return df_org, df

class Data1Ds:
    '''
    Class for 1-D scalar data
    Use subclass Data1D for constructing instance for scalar or vector
    '''
    def __init__(self, df=None, col_1 = None, xrange: tuple = None, yrange: tuple = None):
        '''
        df: pandas DataFrame, col_1: selected column of df and its value is v1, 
        xrange: apply x range of df index or full range of index (0, -1)
        '''
        if xrange is None:  # default x full range
            self.df = df[df.index[0]:df.index[-1]]
        else:  # x range set via argument
            self.df = df[xrange[0]:xrange[1]]
        self.xrange = (self.df.index[0], self.df.index[-1])
        (self.xmin, self.xmax) = self.xrange
        if type(self.df.index) == pd.DatetimeIndex:
            self.x = self.df.index.to_pydatetime()  # pandas datetimeindex => datetime.datetime
        else:
            self.x = self.df.index
        self.s1 = self.df[col_1]  # Series
        self.v1 = self.df[col_1].values  # NumPy
        self.v1max = max(self.v1)
        self.v1min = min(self.v1)
        self.v1range = (self.v1min, self.v1max)
        if yrange is None:
            self.vrange = self.v1range  # to set y-range automatically by default
        else:
            self.yrange = yrange

class Data1D(Data1Ds):
    '''
    Organize 1-D scalar or vector data.
    初期化メソッドの引数にcol_2が存在しない場合はスカラー，存在する場合はベクトルと自動判定
    '''
    def __init__(self, df=None, col_1 = None, col_2 = None, xrange: tuple = None):
        '''
        USAGE: scalar = Data1D(df, 'kion'), vector = Data1D(df, 'u', 'v')
        '''
        super().__init__(df, col_1, xrange)
        if col_2:
            self.s2 = self.df[col_2]  # Series
            self.v2 = self.df[col_2].values  # numpy
            self.v2max = np.max(self.v2)
            self.v2min = np.min(self.v2)
            self.v2range = (self.v2min, self.v2max)
            ## Override in Plot1D for handling rolling mean
            self.v = np.sqrt(self.v1**2 + self.v2**2)  # v: magnitude of vector
            vmax = max(self.v)
            self.vrange = (-vmax, vmax)  # useful for scaling vertical axis of vector

class Data1D_PlotConfig:
    def __init__(self, fig_size: tuple=(10,2), title_size: int=14, label_size: int=12,
                 plot_color = 'b', xlabel = None, ylabel = None, v_color = 'k', vlabel = None,
                 vlabel_loc = 'lower right', xlim: tuple = None, ylim: tuple = None,
                 x_major_locator = None, y_major_locator = None,
                 x_minor_locator = None, y_minor_locator = None,
                 grid = False, format_xdata = None, format_ydata = None):
        self.fig_size = fig_size
        self.title_size = title_size
        self.label_size = label_size
        self.plot_color = plot_color
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.v_color = v_color  # color for fill_between of magnitude v
        self.vlabel = vlabel
        self.vlabel_loc = vlabel_loc
        self.xlim = xlim
        self.ylim = ylim
        self.x_major_locator = x_major_locator
        self.y_major_locator = y_major_locator
        self.x_minor_locator = x_minor_locator
        self.y_minor_locator = y_minor_locator
        self.grid = grid
        self.format_xdata = format_xdata
        self.format_ydata = format_ydata

class Plot1D:
    def __init__(self, plot_config, data, window=1, center=True):
        '''
        window: rolling mean window in odd integer, center: rolling mean at center
        '''
        self.cfg = plot_config
        self.data = data
        if window % 2 != 1:
            print('ERROR: Rolling window must be odd integer.')
        self.window = window
        self.center = center
        self.data.v1 = self.data.s1.rolling(window=self.window,
                                            center=self.center).mean().values
        ## 'v2'がself.dataの属性に含まれているかチェック
        if 'v2' in [key for key,value in data.__dict__.items()]:
            self.data.v2 = self.data.s2.rolling(window=self.window,
                                                center=self.center).mean().values
            self.data.v = np.sqrt(np.square(self.data.v1) + np.square(self.data.v2))
        self.figure = plt.figure(figsize=self.cfg.fig_size)

    def get_axes(self):
        return self.figure.add_subplot(1,1,1)
        
    def update_axes(self, axes):
        ## axes.set_ylim(self.data.v1range[0], self.data.v1range[1])
        ## axes.set_xlim(parse("2014-01-15"), parse("2014-01-16"))
        if self.cfg.xlim:
            axes.set_xlim(self.cfg.xlim)
        else:
            axes.set_xlim(self.data.x[0], self.data.x[-1])  # default: full range
        if self.cfg.ylim:
            axes.set_ylim(self.cfg.ylim)
        else:
            axes.set_ylim(self.data.vrange[0], self.data.vrange[1])  # default
        axes.grid(self.cfg.grid)
        if self.cfg.xlabel:
            axes.set_xlabel(self.cfg.xlabel, fontsize = self.cfg.label_size)
        if self.cfg.ylabel:
            axes.set_ylabel(self.cfg.ylabel, fontsize = self.cfg.label_size)
        if self.cfg.format_xdata:
            axes.format_xdata = self.cfg.format_xdata  # DateFormatter('%Y-%m-%d')
        if self.cfg.format_ydata:
            axes.format_ydata = self.cfg.format_ydata
        if self.cfg.x_major_locator:
            axes.xaxis.set_major_locator(self.cfg.x_major_locator)
        if self.cfg.y_major_locator:
            axes.yaxis.set_major_locator(self.cfg.y_major_locator)
        if self.cfg.x_minor_locator:
            axes.xaxis.set_minor_locator(self.cfg.x_minor_locator)  # days
        if self.cfg.y_minor_locator:
            axes.yaxis.set_minor_locator(self.cfg.y_minor_locator)
        ## fig.autofmt_xdate()
        return axes

    def make_plot(self, axes):
        return axes.plot(self.data.x, self.data.v1, color=self.cfg.plot_color)

    def make_quiver(self, axes):
        '''
        Plot vectors and unit vector
        '''
        #print(type(self.data.x[0]))
        ## Check whether dtype of x is datetime.datetime.
        if isinstance(self.data.x[0], datetime.datetime):
            x = date2num(self.data.x)
        else:
            x = self.data.x

        self.q = axes.quiver(x, 0, self.data.v1, self.data.v2, color=self.cfg.plot_color,
                             units='y', scale_units='y', scale=1, headlength=1,
                             headaxislength=1, width=0.1, alpha=0.5)
        return self.q

    def make_quiverkey(self, axes):
        return axes.quiverkey(self.q, 0.2, 0.1, 5, '5 m/s', labelpos='W')

    def make_fill_between(self, axes):
        fill = axes.fill_between(date2num(self.data.x), self.data.v, 0,
                                 color= self.cfg.v_color, alpha=0.1)
        ## Fake box to insert a legend
        p = axes.add_patch(plt.Rectangle((1,1), 1, 1, fc=self.cfg.v_color, alpha=0.1))
        leg = axes.legend([p], [self.cfg.vlabel], loc=self.cfg.vlabel_loc)
        leg._drawFrame=False

    def save_plot(self, filename, **kwargs):
        axes = self.update_axes(self.get_axes())
        plot = self.make_plot(axes)
        self.figure.savefig(filename, **kwargs)

    def save_vector_plot(self, filename, magnitude=None, **kwargs):
        axes = self.update_axes(self.get_axes())
        quiver = self.make_quiver(axes)
        quiverkey = self.make_quiverkey(axes)
        if magnitude:
            fill_between = self.make_fill_between(axes)
        self.figure.savefig(filename, **kwargs)
