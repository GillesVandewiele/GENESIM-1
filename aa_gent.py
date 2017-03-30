import requests

import pandas as pd
from dateutil.parser import parse
from datetime import datetime
import urllib
import urllib2
from lxml import html
import numpy as np
import math

# http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=fdafef39ee2140cf931100707163010&q=51.0161,3.7337&format=json&date=2016-10-01&enddate=2016-10-30&tp=1
#
# dates = [('2015-07-20', '2016-07-31'), ('2015-08-01', '2016-08-27'), ('2015-09-07', '2016-08-10'),
#          ('2016-09-10', '2016-09-12'), ('2016-11-08', '2016-11-27')]
# dates = [('2016-11-08', '2016-11-28')]
# for date_range in dates:
#     weather_entries = []
#     str_id = '-'.join(date_range[0].split('-')[:2])
#     url_prefix = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=bf257600769c459f9b1141304163010&q=51.0161,3.7337&format=json&tp=1'
#     url_date = '&date='+date_range[0]+'&enddate='+date_range[1]
#     weather_json = requests.get(url_prefix+url_date).json()
#     print weather_json
#     for weather_json_entry in weather_json['data']['weather']:
#         weather_date = weather_json_entry['date']
#         weather_year, weather_month, weather_day = list(map(int, weather_date.split('-')))
#         for hourly_weather_entry in weather_json_entry['hourly']:
#             weather_time = hourly_weather_entry['time']
#             humidity = hourly_weather_entry['humidity']
#             windspeed = hourly_weather_entry['windspeedKmph']
#             visibility = hourly_weather_entry['visibility']
#             temperature = hourly_weather_entry['tempC']
#             weather_type = hourly_weather_entry['weatherCode']
#             weather_hour = int(int(weather_time)/100)
#             weather_date = datetime(year=weather_year, month=weather_month, day=weather_day,
#                                     hour=weather_hour)
#
#             weather_entries.append([weather_date, temperature, humidity, windspeed, visibility, weather_type])
#     weather_df = pd.DataFrame(weather_entries)
#     weather_df.columns = ['Date', 'Temperature', 'Humidity', 'Windspeed', 'Visibility', 'Weather Type']
#     weather_df.to_csv('weather_features'+str_id+'.csv', index=False)

# df = pd.read_csv('data/aa_gent.csv', sep=";")
#
# label_col = 'RPE'
# feature_cols = ['S1', 'S2', 'S3', 'S4', 'S5', 'HF-zone 80-90',
#                 'HF-zone 70-80', 'HF-zone 90-100', 'Aantal sprints',
#                 'Gem v', 'Tijd (s)', 'Afstand', 'Datum training', 'ID', 'Naam speler']
# cols = feature_cols + [label_col] + ['temperature', 'humidity', 'windspeed', 'visibility', 'weather_type']
# print df.isnull().sum()
# print df[label_col].value_counts()
# df['ID'] = df['ID-nummer.1'].astype(int)
# df = df[feature_cols + [label_col]]

# weather_df = pd.concat([pd.read_csv('weather/weather_data.csv'), pd.read_csv('weather/weather_data_july_1.csv'),
#                         pd.read_csv('weather/weather_data_july_2.csv'), pd.read_csv('weather/weather_data_aug_1.csv'),
#                         pd.read_csv('weather/weather_data_aug_2.csv'), pd.read_csv('weather/weather_data_sep_1.csv'),
#                         pd.read_csv('weather/weather_data_sep_2.csv')])

# weather_df = weather_df[(weather_df['station_name'] == 'Gent-Sint-Pieters')]# | (weather_df['station_name'] == 'Gent-Dampoort')]
# weather_df['date_time'] = weather_df['date_time'].apply(lambda x: parse(x))
#
# new_vectors = []
#
player_ids = {}
not_found = []

def get_fifa_rating(player_name, season):
    # http://sofifa.com/api.php?action=history&type=player&id=<201043> --> Get the player ID
    try:
        if player_name not in player_ids:
            url = "http://sofifa.com/players?keyword=" + urllib.quote(player_name) + "&v=" + season + "&hl=en-US"
            req = urllib2.Request(url, headers={'User-Agent': "Magic Browser"})
            page = urllib2.urlopen(req)
            s = page.read().decode("utf-8")
            # print(s)
            tree = html.fromstring(s).getroottree()

            table_elts = tree.findall('//tbody/tr/')
            team_id = table_elts[6].find('a').attrib['href'].split('/')[-1]
            player_id = table_elts[0].find('a').attrib['href'].split('/')[-1]

            # if team_id != '674': raise   # Player not from AA Ghent
            # else:
            player_ids[player_name] = player_id

    except:
        if player_name not in not_found:
            print 'Info not found for', player_name
            not_found.append(player_name)

    # for table_cell in tree.findall('//tbody/tr/'):
    #     if 'data-title' in table_cell.attrib and table_cell.attrib['data-title'] == "Overall rating":
    #         rating = table_cell.find('./span').text
    #         print((url, int(rating)))
    #         return 1, int(rating)
    #
    # print((url, 50))
    # return 0, 50

# print(player_ids)
# print(not_found)
#
# new_df = pd.DataFrame(new_vectors)
# new_df.columns = cols
# new_df = pd.get_dummies(new_df, ['ID'])
# # new_df.to_csv('aa_gent_features.csv', index=False)
#
#
#
# print(weather_df.head(5))

# df = pd.read_csv('data/aa_gent.csv', sep=';')
# print list(df.columns)
# id_player_mapping = {}
# player_id_df = df[['ID-nummer.1', 'Naam speler']].drop_duplicates()
# for i in range(len(player_id_df)):
#     entry = player_id_df.iloc[i, :]
#     id_player_mapping[entry['ID-nummer.1']]  = entry['Naam speler']
#

player_sofifa_mapping = {'Nana Asare': '158375', 'Thomas Matton': '176019', 'Marko Poletanovic': '227150',
                         'Kalifa Coulibaly': '206141', 'Uros Vitas': '221891', 'Rob Schoofs': '205369',
                         'Lasse Nielsen': '177373', 'Anderson Esiti': '220921', 'Yaya Soumahoro': '199818',
                         'Renato Neto': '199663', 'Thomas Foket': '208509', 'Jeremy Perbet': '150594',
                         'Erik Johansson': '206022', 'Haris Hajradinovic': '226517', 'Brecht Dejaegere': '201043',
                         'Moses Simon': '216820', 'Emir Kujovic': '187915', 'Hannes Van Der Bruggen': '201438',
                         'Serge Tabekou': '228506', 'Mustapha Oussalah': '45362', 'Dieumerci Ndongala': '204908',
                         'Nicklas Pedersen': '172470', 'Danijel Milicevic': '168732', 'Laurent Depoitre': '212933',
                         'Rami Gershon': '194858', 'Sven Kums': '176009', 'Benito Raman': '200429',
                         'Kenneth Saief': '224813', 'Stefan Mitrovic': '209547', 'Kenny Saief': '224813',
                         'Lucas Schoofs': '230933', 'Hatem Elhamed': '202601', 'Siebe Horemans': '235222',
                         'Rafael Rafinha': '205951', 'Ibrahim Rabiu': '197359', 'Ofir Davidzada': '217525',
                         'Jeremy Taravel': '183283', 'Perbet': '150594'}
#
# #
# vectors = []
# for player in player_sofifa_mapping:
#     url = 'http://2016.sofifa.com/api.php?action=history&type=player&id='+player_sofifa_mapping[player]
#     fifa_json = requests.get(url).json()
#     for entry in fifa_json['chart']:
#         vectors.append([player, entry['date'], entry['overall'], entry['phy'], entry['pac']])
# player_df = pd.DataFrame(vectors)
# player_df.columns = ['player', 'date', 'overall', 'phy', 'pac']
# player_df.to_csv('player_stats.csv', index=False)

df = pd.read_csv('aa_gent_revised.csv', sep=',')

# for i in range(len(df)):
#     entry = df.iloc[i,:]
#     get_fifa_rating(entry['Speler'], '15')
#     get_fifa_rating(entry['Speler'], '16')
#     # date = parse(entry['Datum training'])
# print(player_ids)
# print(not_found)

print list(df.columns)
id_player_mapping = {}
player_id_df = df[['Speler', 'Idnummer']].drop_duplicates()
for i in range(len(player_id_df)):
    entry = player_id_df.iloc[i, :]
    if not math.isnan(entry['Idnummer']):
        id_player_mapping[int(entry['Idnummer'])] = entry['Speler']

print id_player_mapping

weather_df = pd.concat([pd.read_csv('weather_features2015-07.csv'), pd.read_csv('weather_features2015-08.csv'),
                        pd.read_csv('weather_features2015-09.csv'), pd.read_csv('weather_features2016-09.csv'),
                        pd.read_csv('weather_features2016-11.csv')])
print list(weather_df.columns)
weather_df['Date'] = weather_df['Date'].apply(lambda x: parse(x))

player_df = pd.read_csv('player_stats.csv')
player_df['date'] = player_df['date'].apply(lambda x: parse(x))

aa_gent_df = pd.read_csv('aa_gent_revised.csv')
aa_gent_df['Datum'] = aa_gent_df['Datum'].apply(lambda x: parse(x))

train_times = pd.read_csv('train_times.csv')

train_times['Datum'] = train_times['Datum'].apply(lambda x: parse(x))
train_times['Start training'] = train_times['Start training'].apply(lambda x: parse(x))

aa_gent_features = []
for i in range(len(aa_gent_df)):
    entry = aa_gent_df.iloc[i, :]
    training_date = entry['Datum']
    train_time = train_times[train_times['Datum'] == training_date].iloc[0, :]['Start training']
    train_hour = train_time.hour if train_time.minute < 30 else train_time.hour + 1
    training_date_time = datetime(year=training_date.year, month=training_date.month, day=training_date.day,
                                  hour=train_hour, minute=0)
    weather_features = weather_df[weather_df['Date'] == training_date_time].iloc[0,:][['Temperature','Humidity','Windspeed','Visibility','Weather Type']]
    game_lag = train_times[train_times['Datum'] == training_date].iloc[0, :]['GameLag']

    if int(entry['Idnummer']) in id_player_mapping and entry['Speler'] in player_sofifa_mapping.keys():
        filtered_player_df = player_df[player_df['player'] == id_player_mapping[int(entry['Idnummer'])]]
        min_date_diff = float('inf')
        player_features = None
        for j in range(len(filtered_player_df)):
            player_entry = filtered_player_df.iloc[j, :]
            date_diff = abs((player_entry['date'] - entry['Datum']).total_seconds())
            if date_diff < min_date_diff:
                min_date_diff = date_diff
                player_features = list(player_entry[['overall','phy','pac']].values)

        print id_player_mapping[int(entry['Idnummer'])], player_features, entry['Datum'], min_date_diff
        aa_gent_features.append(list(entry.values) + list(weather_features.values) + player_features + game_lag)

        # aa_gent_features.append(list(entry.values) + list(weather_features.values))
aa_gent_features_df = pd.DataFrame(aa_gent_features)
aa_gent_features_df.columns = list(aa_gent_df.columns) + ['Temperature','Humidity','Windspeed','Visibility','Weather Type'] + ['overall', 'phy', 'pac']
aa_gent_features_df.to_csv('aa_gent_with_player_features.csv', index=False)

