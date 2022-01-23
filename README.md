# metdata
This repository contains tools for processing meteorological data, currently supporting the GWO (Ground Weather Observation) dataset provided by the Japan Meteorological Business Support Center ([JMBSC](http://www.jmbsc.or.jp/en/index-e.html)). The dataset was a commercial product in Japanese (see [web](http://www.roy.hi-ho.ne.jp/ssai/mito_gis/) for details).

（一財）日本気象業務支援センターから販売されていた，1961年から2015年までのアメダスや地上観測データDVDから必要なデータを切り出し，処理するツールです．[ウェザートーイ](http://www.roy.hi-ho.ne.jp/ssai/mito_gis/)がサポートしています．

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
Only local install is supported.

```bash
pip install -e .
```

# Extract data by specifying an observatory and period.

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

#### Class Data1D
1D scalar (`col_1`) plot or 1D vector `(col_1, col_2)` plot.

Parameters
----------
- df (pandas.DataFrame)
- col_1 (str) : scalar variable or 1st dimension of vector variable
- col_2 (str) : 2nd dimension of vector variable
- xrange: tuple = None
- yrange: tuple = None

#### Class Data1D_PlotConfig
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

#### Constructor

```Python
Plot1D(plot_config, data, window: int=1, center: bool=True)
```

**Parameters**
----------
- `plot_config` (instance of Data1D_PlotConfig)
- `data` (pandas.DataFrame)
- `window` (int) : Rolling mean window in odd integer
- `center` (bool): True: Rolling mean at center

#### Methods

```Python
save_vector_plot(filename, magnitude, **kwargs)
```

**Parameters**
----------
- `filename` (str) : output file name
- `magnitude` (bool) : True: plot magnitude of vectors
- `**kwargs` (dict) : Transferred to `figure.savefig` in matplotlib
