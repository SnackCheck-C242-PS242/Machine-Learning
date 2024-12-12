## BIKE-SHARING

# Proyek Data Analisis: Bike Sharing Dataset

Pada [**Bike Sharing Dataset**](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) terdapat 2 dataset yaitu **day.csv** dab **hour.csv**, dimana berisi rekap laporan penggunaan Bike Sharing secara daily (per-hari) dan hourly (per-jam) pada tahun 2011 dan 2012 yang ada pada sistem [**Capital Bikeshare**](https://capitalbikeshare.com) disertai dengan informasi cuaca dan musim.

**Dataset Information**

- instant: record index
- dteday : date
- season : season (1:spring, 2:summer, 3:fall, 4:winter)
- yr : year (0: 2011, 1:2012)
- mnth : month ( 1 to 12)
- hr : hour (0 to 23)
- holiday : weather day is holiday or not (extracted from [Web Link])
- weekday : day of the week
- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
- weathersit :
  - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
  - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
  - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
  - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
- atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
- hum: Normalized humidity. The values are divided to 100 (max)
- windspeed: Normalized wind speed. The values are divided to 67 (max)
- casual: count of casual users
- registered: count of registered users
- cnt: count of total rental bikes including both casual and registered


## How to Run This Project ?

1. Clone this repository

```
git clone https://github.com/airtaprtw/BIKE-SHARING.git
```

2. Install all library

```
pip install numpy pandas matplotlib seaborn jupyter streamlit babel
```

or

```
pip install -r requirements.txt
```

3. Go to dashboard folder

```
cd dashboard
```

4. Run with Streamlit

```
streamlit run dashboard.py
```

Atau dapat langsung mengunjungi link streamlit berikut:
#### [**Click here to view Bike Sharing Dashboard**](https://bike-sharing-7nppgfroixweaumaa2ujir.streamlit.app/)
