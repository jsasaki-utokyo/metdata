# metdata
This repository contains tools for processing meteorological data, currently supporting the GWO (Ground Weather Observation) dataset provided by the Japan Meteorological Business Support Center ([JMBSC](http://www.jmbsc.or.jp/en/index-e.html)). The dataset was a commercial product in Japanese (see [web](http://www.roy.hi-ho.ne.jp/ssai/mito_gis/) for details).

（一財）日本気象業務支援センターから販売されていた，1961年から2015年までのアメダスや地上観測データDVDから必要なデータを切り出し，処理するツールです．[ウェザートーイ](http://www.roy.hi-ho.ne.jp/ssai/mito_gis/)がサポートしています．

気象庁公開の気象情報は[whp View](http://www.roy.hi-ho.ne.jp/ssai/mito_gis/whpview/index.html)を用いて閲覧可能です．フリー版でなければCSV形式でのダウンロードも可能です．詳細は下端にまとめます．

# GWO dataset

Each file contains meteorological data (see the columns below) from YYYY-01-01 01:00:00 to YYYY-12-31 24:00:00, JST in each year at each observation station. The time at 24:00:00 is converted to 00:00:00 following the Python `datetime` hour range.

### Columns of dataset

```Python
["KanID","Kname","KanID_1","YYYY","MM","DD","HH","lhpa","lhpaRMK",
 "shpa","shpaRMK","kion","kionRMK","stem","stemRMK","rhum","rhumRMK",
 "muki","mukiRMK","sped","spedRMK","clod","clodRMK","tnki","tnkiRMK",
 "humd","humdRMK","lght","lghtRMK","slht","slhtRMK","kous","kousRMK"]

["観測所ID","観測所名","ID1","年","月","日","時",
 "現地気圧(0.1hPa)","ID2","海面気圧(0.1hPa)","ID3",
 "気温(0.1degC)","ID4","蒸気圧(0.1hPa)","ID5",
 "相対湿度(%)","ID6","風向(1(NNE)～16(N))","ID7","風速(0.1m/s)","ID8",
 "雲量(10分比)","ID9","現在天気","ID10","露天温度(0.1degC)","ID11",
 "日照時間(0.1時間)","ID12","全天日射量(0.01MJ/m2/h)","ID13",
 "降水量(0.1mm/h)","ID14"]
```

# Install
Only local install is supported. Move into the directory of the local repository of *metdata* and execute the following. Do not forget the last period.

```bash
pip install -e .
```

## Extract data by specifying an observatory and period.

```Python
from metdata import gwo

datetime_ini = "2019-12-1 00:00:00"
datetime_end = "2020-12-30 00:00:00"
stn = "Chiba"
dirpath = "d:/dat/met/JMA_DataBase/GWO/Hourly/"

met = gwo.Hourly(datetime_ini=datetime_ini, datetime_end=datetime_end, stn=stn, dirpath=dirpath)
met.df  # pandas.DataFrame
```


# Plot

```Python
## Specify pandas.DataFrame and set an item to be plotted.
data = gwo.Data1D(df=met.df, col_1='kion')
ylabel='Temperature (degC)'
xlim = None
## xlim = (parse("1990-09-02"), parse("1992-09-03"))
dx = 7
ylim = None
dy = 2

## Set window=1 when no plot.
window=1
#try:
plot_config = gwo.Data1D_PlotConfig(xlim=xlim, ylim=ylim, 
                                    x_minor_locator=DayLocator(interval=dx),
                                    y_minor_locator = MultipleLocator(dy),
                                    format_xdata = DateFormatter('%Y-%m-%d'),
                                    ylabel = ylabel)
gwo.Plot1D(plot_config, data, window=window,
           center=True).save_plot('data.png', dpi=600)
```

## Class Data1D
1D scalar (`col_1`) plot or 1D vector `(col_1, col_2)` plot.

Parameters
----------
- `df` (pandas.DataFrame)
- `col_1` (str) : Scalar variable or 1st dimension of vector variable
- `col_2` (str) : 2nd dimension of vector variable
- `xrange` (tuple) : Default = None
- `yrange` (tuple) : Default = None

## Class Data1D_PlotConfig
Configuration setting for plot parameters

Parameters
----------
- `fig_size` (tuple) : Default = (10,2)
- `title_size` (int) : Default = 14
- `label_size` (int) : Default = 12
- `plot_color` (str) : Default = 'b'
- `xlabel` (str) : Default = None
- `ylabel` (str) : Default = None
- `v_color` (str) : Default = 'k'
- `vlabel` (str) 
- `vlabel_loc` (str) : Default = 'lower right'
- `xlim` (tuple)
- `ylim` (tuple)
- `x_major_locator`
- `y_major_locator`
- `x_minor_locator`
- `y_minor_locator`
- `grid` (bool) Default = False
- `format_xdata`
- `format_ydata`

## Class Plot1D

### Constructor

```Python
Plot1D(plot_config, data, window: int=1, center: bool=True)
```

**Parameters**
----------
- `plot_config` (instance of Data1D_PlotConfig)
- `data` (pandas.DataFrame)
- `window` (int) : Rolling mean window in odd integer
- `center` (bool): True: Rolling mean at center

### Methods

```Python
save_vector_plot(filename, magnitude, **kwargs)
```

**Parameters**
----------
- `filename` (str) : output file name
- `magnitude` (bool) : True: plot magnitude of vectors
- `**kwargs` (dict) : Transferred to `figure.savefig` in matplotlib


# whp View を用いた気象庁公開気象データの取得 (in Japanese)

[whp View](http://www.roy.hi-ho.ne.jp/ssai/mito_gis/whpview/index.html)を用いて気象庁公開の気象情報が閲覧できます．フリー版でなければCSVのダウンロードも可能です．1度のダウンロードは1測点1年間程度とする必要があります（読み飛ばしエラーが起こりやすい）．．
気象情報は **地上観測** と **アメダス**，および **高層** の3種類があります．

### 雲量の注意
雲量に `0+` といった表現があります．これを含め，値欄の記号の情報は [気象庁web] (https://www.data.jma.go.jp/obd/stats/data/mdrr/man/remark.html)にあります．また，雲量は時別値でも3時間間隔のデータとなっており，欠損値の扱いが要注意です．

## 地上観測
時別，日別，１０分が存在します．

### 地上観測時別
地上観測タブで **観測所** を選択，期間を指定し，ラジオボタンから **時別** を選択，デフォルトの主要素（詳細１，詳細２でもよい）が選択された状態で，**取得** ボタンをクリックします．**左下に進捗バーが現れ，完了するまで待ちます**．取得が完了したら，**ファイルメニュー** の **表示テーブル保存** をクリックし，CSVファイルとして保存します．テキストファイル形式は Shift-JISのCRLFです．

### 地上観測日別
時別と同様ですが，ラジオボタンから **日別** を選び，ラジオボタンの **主要素**，**詳細１**，**詳細２** を切り替えて選択します（それぞれ表示される気象要素が異なります）．

### 地上観測１０分
ラジオボタンから **１０分** を選択，デフォルトの主要素が選択された状態で，**取得** ボタンをクリックします．

## アメダス
時別，日別，１０分が存在します．

### アメダス時別
**アメダスタブ** で **観測所** を選択し，**期間** を指定します．ラジオボタンで **時別** を選択し，デフォルトの主要素が選択された状態で，**取得** ボタンをクリックします．

### アメダス日別
**アメダスタブ** で **観測所** を選択し，**期間** を指定します．ラジオボタンの **日別** を選択し，**主要素**，**詳細１**，**詳細2** を切り替えて選択し，**取得** ボタンをクリックします．

### アメダス１０分
**アメダスタブ** で **観測所** を選択し，**期間** を指定します．ラジオボタンの **１０分** を選択し，デフォルトの主要素が選択された状態で，**取得** ボタンをクリックします．
