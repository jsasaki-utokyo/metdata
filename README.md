# metdata

This repository contains tools for processing meteorological data, currently supporting the GWO (Ground Weather Observation) dataset format originally provided by the Japan Meteorological Business Support Center ([JMBSC](http://www.jmbsc.or.jp/en/index-e.html)).

**Note:** The original GWO dataset was a commercial product available until 2021 (see [web](http://www.roy.hi-ho.ne.jp/ssai/mito_gis/) for details). For data from 2022 onwards, use the [GWO-AMD](https://github.com/jsasaki-utokyo/GWO-AMD) tool to download and convert JMA's publicly available CSV data into GWO format.

## Data Preparation

### For 2022 and Later (Current JMA Data)

1. Download JMA CSV data and convert to GWO format using [GWO-AMD](https://github.com/jsasaki-utokyo/GWO-AMD):
   ```bash
   # Install GWO-AMD
   git clone https://github.com/jsasaki-utokyo/GWO-AMD.git
   cd GWO-AMD
   conda env create -f environment.yml
   conda activate gwo-amd

   # Download and convert JMA data to GWO format
   jma-download --year 2023 --station Tokyo --gwo-format
   ```

2. Copy the converted GWO CSV files to your data directory:
   ```bash
   cp -r ./output/* $DATA_DIR/met/JMA_DataBase/GWO/Hourly/
   ```

3. Use metdata to process the GWO format files (same workflow as legacy data)

### For 2021 and Earlier (Legacy GWO Data)

If you have the commercial GWO dataset, preprocess it using [GWO-AMD](https://github.com/jsasaki-utokyo/GWO-AMD) to create yearly CSV files at each station.

## GWO dataset

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

## Installation

### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate metdata
```

### Using pip

```bash
pip install -r requirements.txt
pip install -e .
```

**Note**: The package requires `nkf` (Network Kanji Filter) to be installed on your system:
- Linux/WSL: `sudo apt install nkf`
- macOS: `brew install nkf`

## Usage

### Extract data by specifying an observatory and period

```Python
from metdata import gwo

datetime_ini = "2019-12-1 00:00:00"
datetime_end = "2020-12-30 00:00:00"
stn = "Chiba"
dirpath = "/mnt/d/dat/met/JMA_DataBase/GWO/Hourly/"

met = gwo.Hourly(datetime_ini=datetime_ini, datetime_end=datetime_end, stn=stn, dirpath=dirpath)
met.df  # pandas.DataFrame
```


### Plotting

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

### Create CF-compliant NetCDF

Use the helper script under `scripts/` to package a station/date range into a NetCDF file that follows the CF convention:

```bash
python scripts/gwo_to_cf_netcdf.py \
  --station Tokyo \
  --start "2019-01-01 00:00:00" \
  --end   "2020-01-01 00:00:00" \
  --output tokyo_2019.nc
```

The script expects hourly CSVs under `$DATA_DIR/met/JMA_DataBase/GWO/Hourly/` (or a custom `--data-dir`) and reports missing values before writing the file.

#### Wind Data Handling

The script implements intelligent handling of calm and missing wind conditions:

**Calm Conditions:**
- When wind direction is N/A (`muki=0`) with valid RMK and very low speed (≤ 0.3 m/s), the data represents calm conditions
- `wind_from_direction` is set to NaN (direction is undefined when wind is calm)
- `wind_speed`, `eastward_wind`, and `northward_wind` are all set to `0`
- This ensures vector components remain consistent with zero speed

**Missing Data:**
- When both `mukiRMK` and `spedRMK` indicate missing observations (RMK ∈ {0,1,2}), all wind variables are set to `_FillValue`
- This prevents mismatches between direction and speed availability

**Nighttime and No-Phenomenon Data:**
- Nighttime samples automatically set `duration_of_sunshine` and `surface_downwelling_shortwave_flux_in_air` to `0`
- Other "no phenomenon" indicators (e.g., dry hours for `precipitation_flux`) are also set to `0`
- Remaining missing values truly reflect gaps in the underlying data

#### Inspecting Missing Values

To inspect missing samples inside a generated NetCDF file:

```bash
# Uses Asia/Tokyo timezone by default
python scripts/check_netcdf_missing.py tokyo_2019.nc --limit 20

# Or specify a different timezone
python scripts/check_netcdf_missing.py tokyo_2019.nc --timezone UTC --limit 20
```

Each reported line lists the timestamp (in JST by default), variable name, stored missing value, and the closest previous/next valid measurements to help with QA. The RMK code from the original GWO data is also displayed when available.

### Interpolate Missing Values

Use the interpolation script to fill small gaps in NetCDF time series data:

```bash
# Basic usage: linear interpolation with max gap of 3 (overwrites input)
python scripts/interpolate_netcdf.py tokyo_2019.nc

# Save to a different file
python scripts/interpolate_netcdf.py tokyo_2019.nc --output tokyo_2019_interp.nc

# Use cubic interpolation with larger gap threshold
python scripts/interpolate_netcdf.py tokyo_2019.nc --method cubic --max-gap 5

# Interpolate specific variables only
python scripts/interpolate_netcdf.py tokyo_2019.nc \
  --variables eastward_wind northward_wind wind_speed

# Create backup before overwriting
python scripts/interpolate_netcdf.py tokyo_2019.nc --backup
```

**Features:**
- **Smart gap handling:** Only fills gaps up to `--max-gap` length (default: 3)
- **Multiple methods:** linear, time, cubic, quadratic, spline, pchip, akima, and more
- **Alerts:** Warns when gaps exceed the maximum threshold
- **Selective processing:** Can target specific variables or process all
- **Safe operation:** Optionally creates backup files with `.bak` extension
- **Metadata tracking:** Records interpolation details in NetCDF history attribute

**Examples:**

```bash
# Interpolate wind data only, allowing gaps up to 2 hours
python scripts/interpolate_netcdf.py tokyo_2019.nc \
  --variables eastward_wind northward_wind \
  --max-gap 2

# Time-weighted interpolation (better for non-uniform time spacing)
python scripts/interpolate_netcdf.py tokyo_2019.nc --method time

# Suppress alerts for long gaps
python scripts/interpolate_netcdf.py tokyo_2019.nc --no-alerts
```

The script reports statistics for each variable, showing how many missing values were filled vs. unfilled due to exceeding the gap threshold.

### Visualize Time Series

Generate comprehensive time series plots from NetCDF files:

```bash
# Plot entire dataset
python scripts/plot_netcdf_timeseries.py tokyo_2019.nc

# Plot specific time period
python scripts/plot_netcdf_timeseries.py tokyo_2019.nc \
  --start "2019-06-01" --end "2019-08-31"

# Apply 24-hour moving average
python scripts/plot_netcdf_timeseries.py tokyo_2019.nc --window 24

# Plot every 3 hours (reduce density)
python scripts/plot_netcdf_timeseries.py tokyo_2019.nc --skip 3

# Custom output directory
python scripts/plot_netcdf_timeseries.py tokyo_2019.nc --output-dir figures
```

**Features:**
- **Multi-panel layout:** Each variable plotted in its own panel with appropriate labels and units
- **Wind vector display:** Wind components (eastward/northward) automatically combined into vector plot using the metdata package style
  - All data points plotted by default (no automatic subsampling)
  - Thin vector lines with minimal arrow heads for clarity
  - Blue color with transparency (α=0.5)
  - Y-axis shows wind magnitude scale
- **Smart formatting:** LaTeX-style units (e.g., m s$^{-1}$, W m$^{-2}$), temperature in °C
- **Moving average:** Optional smoothing with configurable window size
- **Data subsampling:** Plot every N-th point to reduce density (applies to all variables including wind)
- **Time range selection:** Focus on specific periods
- **High-quality output:** PNG with 300 DPI (configurable), tight bounding box

**Output:**
- Saved as `PNG/{input_filename}_timeseries.png`
- PNG directory created automatically if it doesn't exist
- Each variable in separate panel (tall figure with independent y-axes)
- Wind vectors show both direction and magnitude clearly

**Variable Labels:**
The script uses concise, publication-quality labels:
- T$_{\mathrm{air}}$ (°C) - Air temperature
- T$_{\mathrm{dew}}$ (°C) - Dew point temperature
- P$_{\mathrm{sfc}}$ (hPa) - Surface pressure
- P$_{\mathrm{msl}}$ (hPa) - Mean sea level pressure
- e (hPa) - Water vapor partial pressure
- RH (1) - Relative humidity
- Wind (m s$^{-1}$) - Wind vectors
- SW$_{\downarrow}$ (W m$^{-2}$) - Downwelling shortwave radiation
- Precip (kg m$^{-2}$ s$^{-1}$) - Precipitation flux

## API Reference

### Class Data1D
1D scalar (`col_1`) plot or 1D vector `(col_1, col_2)` plot.

Parameters
----------
- `df` (pandas.DataFrame)
- `col_1` (str) : Scalar variable or 1st dimension of vector variable
- `col_2` (str) : 2nd dimension of vector variable
- `xrange` (tuple) : Default = None
- `yrange` (tuple) : Default = None

### Class Data1D_PlotConfig
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

### Class Plot1D

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


---

<details>
<summary><b>日本語情報 (Japanese Information)</b></summary>

## 概要

- このリポジトリでは（一財）日本気象業務支援センター([JMBSC](http://www.jmbsc.or.jp/en/index-e.html))が提供する，2021年までのGWO(地上気象観測)データセットを処理するツールを提供しています．詳細は[web](http://www.roy.hi-ho.ne.jp/ssai/mito_gis/)および[ブログ記事](https://estuarine.jp/2016/05/gwo/)を参照ください．
この2021年までのGWOデータセットは[GWO-AMD](https://github.com/jsasaki-utokyo/GWO-AMD)を用いて，各観測点における年別のCSVファイルに変換しておく前処理が必要です．

- （一財）日本気象業務支援センターから販売されていた，1961年から2015年までのアメダス（AMD）や地上観測データ（GWO）は，購入者限定で[ウェザートーイ](http://www.roy.hi-ho.ne.jp/ssai/mito_gis/)が2021年のデータまでサポートしています．2022年以降は気象庁互換形式のみ提供されています．

- 気象庁公開の気象情報は[whp View](http://www.roy.hi-ho.ne.jp/ssai/mito_gis/whpview/index.html)を用いて閲覧可能です．フリー版でなければCSV形式でのダウンロードも可能です．詳細は下端にまとめます．しかし，2024年7月現在，このダウンロードはできなくなったようです．上記データセット購入者は引き続き，気象庁互換形式のデータをサポートwebからダウンロード可能です．

- **気象庁互換形式**の値欄には記号が含まれる場合があります．[**値欄の詳細**](https://www.data.jma.go.jp/obd/stats/data/mdrr/man/remark.html)です．

- データーの2次利用には「気象庁提供」の明示が必要です．

## whp View を用いた気象庁公開気象データの取得

[whp View](http://www.roy.hi-ho.ne.jp/ssai/mito_gis/whpview/index.html)を用いて気象庁公開の気象情報が閲覧できます．フリー版でなければCSVのダウンロードも可能です．1度のダウンロードは1測点1年間程度とする必要があります（読み飛ばしエラーが起こりやすい）．．
気象情報は **地上観測** と **アメダス**，および **高層** の3種類があります．

### 雲量の注意
雲量に `0+` といった表記があります．これを含め，値欄の記号の情報は [気象庁web](https://www.data.jma.go.jp/obd/stats/data/mdrr/man/remark.html)にあります．また，雲量は時別値でも3時間間隔のデータとなっており，欠損値の扱いが要注意です．

### 地上観測
時別値，日別値，１０分値が存在します．

#### 地上観測時別
地上観測タブで **観測所** を選択，期間を指定し，ラジオボタンから **時別** を選択，デフォルトの主要素（詳細１，詳細２でもよい）が選択された状態で，**取得** ボタンをクリックします．**左下に進捗バーが現れ，完了するまで待ちます**．取得が完了したら，**ファイルメニュー** の **表示テーブル保存** をクリックし，CSVファイルとして保存します．テキストファイル形式は Shift-JISのCRLFです．

#### 地上観測日別
時別と同様ですが，ラジオボタンから **日別** を選び，ラジオボタンの **主要素**，**詳細１**，**詳細２** を切り替えて選択します（それぞれ表示される気象要素が異なります）．

#### 地上観測１０分
ラジオボタンから **１０分** を選択，デフォルトの主要素が選択された状態で，**取得** ボタンをクリックします．

### アメダス
時別，日別，１０分が存在します．

#### アメダス時別
**アメダスタブ** で **観測所** を選択し，**期間** を指定します．ラジオボタンで **時別** を選択し，デフォルトの主要素が選択された状態で，**取得** ボタンをクリックします．

#### アメダス日別
**アメダスタブ** で **観測所** を選択し，**期間** を指定します．ラジオボタンの **日別** を選択し，**主要素**，**詳細１**，**詳細2** を切り替えて選択し，**取得** ボタンをクリックします．

#### アメダス１０分
**アメダスタブ** で **観測所** を選択し，**期間** を指定します．ラジオボタンの **１０分** を選択し，デフォルトの主要素が選択された状態で，**取得** ボタンをクリックします．

</details>
