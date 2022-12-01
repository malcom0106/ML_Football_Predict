import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sys

# print("Python version: {}".format(sys.version))
# print("JobLib version: {}".format(joblib.__version__))

model = joblib.load('./assets/modeleFinal.joblib')
countryEncoder = joblib.load('./assets/countryLabelEncoder.joblib')
tournamentEncoder = joblib.load('./assets/tournamentLabelEncoder.joblib')


def make_prediction(df):
    df['away_team'] = countryEncoder.transform(df['away_team'])
    df['home_team'] = countryEncoder.transform(df['home_team'])
    df['tournament'] = tournamentEncoder.transform(df['tournament'])
    return model.predict(df)


st.title('ML Football Results Predictions')

pays = ['Padania', 'Székely Land', 'Sápmi', 'Artsakh', 'Turkmenistan',
        'Tajikistan', 'Maldives', 'Pakistan', 'Sri Lanka', 'Philippines',
        'Syria', 'Bangladesh', 'Kyrgyzstan', 'India', 'Nepal', 'Malaysia',
        'Bhutan', 'North Korea', 'Palestine', 'Mongolia', 'Laos', 'Kuwait',
        'Cambodia', 'Thailand', 'Singapore', 'Japan', 'Lebanon', 'Oman',
        'Jordan', 'Vietnam', 'Qatar', 'Indonesia', 'Hong Kong', 'Germany',
        'Turkey', 'Malta', 'Cyprus', 'Georgia', 'Norway', 'Latvia',
        'Slovenia', 'Greece', 'North Macedonia', 'Finland', 'England',
        'Bahrain', 'Luxembourg', 'Mexico', 'Andorra', 'Russia', 'Bulgaria',
        'Republic of Ireland', 'Switzerland', 'Portugal', 'Estonia',
        'Kazakhstan', 'Poland', 'Denmark', 'Ukraine', 'San Marino',
        'Belarus', 'Netherlands', 'Romania', 'Hungary', 'Azerbaijan',
        'Sweden', 'Montenegro', 'Bosnia and Herzegovina', 'Argentina',
        'France', 'Iceland', 'Serbia', 'Italy', 'Armenia', 'Kosovo',
        'Austria', 'Liechtenstein', 'Israel', 'Spain', 'Scotland',
        'Moldova', 'Lithuania', 'Wales', 'Czech Republic', 'Saudi Arabia',
        'Jersey', 'Isle of Man', 'Hitra', 'Orkney', 'Menorca',
        'Falkland Islands', 'Ivory Coast', 'Tanzania', 'Egypt',
        'Burkina Faso', 'Cameroon', 'Liberia', 'Tunisia', 'DR Congo',
        'South Africa', 'Morocco', 'Angola', 'Burundi', 'Senegal',
        'Namibia', 'Mali', 'Belgium', 'Uganda', 'Chad', 'Niger',
        'Zimbabwe', 'Nigeria', 'Rwanda', 'Gabon', 'Guinea', 'Cape Verde',
        'Gambia', 'Zambia', 'Malawi', 'United States',
        'Central African Republic', 'Libya', 'Benin', 'South Korea',
        'Ethiopia', 'Ghana', 'Lesotho', 'Seychelles', 'Botswana',
        'Djibouti', 'Iran', 'Australia', 'Samoa', 'Fiji',
        'Solomon Islands', 'Tonga', 'Tuvalu', 'New Caledonia', 'Albania',
        'Croatia', 'Slovakia', 'Faroe Islands', 'Gibraltar', 'Eswatini',
        'Mozambique', 'Mauritius', 'Equatorial Guinea', 'Togo', 'Algeria',
        'Congo', 'Eritrea', 'Kenya', 'Venezuela', 'United Arab Emirates',
        'Guinea-Bissau', 'Sudan', 'Madagascar', 'Mauritania',
        'Saint Martin', 'British Virgin Islands', 'Dominican Republic',
        'Antigua and Barbuda', 'Barbados', 'El Salvador', 'Martinique',
        'Puerto Rico', 'Cayman Islands', 'French Guiana',
        'Trinidad and Tobago', 'Saint Vincent and the Grenadines',
        'Sint Maarten', 'Nicaragua', 'Bahamas', 'Guatemala', 'Dominica',
        'Panama', 'Saint Lucia', 'Saint Kitts and Nevis', 'Bermuda',
        'Haiti', 'Cuba', 'Guadeloupe', 'Curaçao', 'Guyana', 'Aruba',
        'Grenada', 'Jamaica', 'United States Virgin Islands', 'Paraguay',
        'Chile', 'Brazil', 'Uruguay', 'Colombia', 'Peru', 'Bolivia',
        'Ecuador', 'Honduras', 'Catalonia', 'Costa Rica',
        'Northern Ireland', 'Suriname', 'New Zealand', 'China PR', 'Iraq',
        'Uzbekistan', 'Guam', 'Taiwan', 'Turks and Caicos Islands',
        'Belize', 'Bonaire', 'Myanmar', 'Yemen', 'Afghanistan',
        'Montserrat', 'Corsica', 'Canada', 'Sierra Leone',
        'São Tomé and Príncipe', 'South Sudan', 'Åland Islands',
        'Basque Country', 'Comoros', 'Anguilla', 'Macau', 'Timor-Leste',
        'Zanzibar', 'Somalia', 'Kárpátalja', 'Panjab', 'Abkhazia',
        'Andalusia', 'Tahiti', 'Papua New Guinea', 'American Samoa',
        'Northern Cyprus', 'Brittany', 'Vanuatu', 'Saare County',
        'Gotland', 'Western Isles', 'Guernsey', 'Alderney', 'Frøya',
        'Isle of Wight', 'Ynys Môn', 'South Ossetia', 'Cook Islands',
        'Réunion', 'Shetland', 'Rhodes', 'Greenland',
        'Republic of St. Pauli', 'Mayotte', 'Northern Mariana Islands',
        'Kernow', 'Brunei', 'Galicia', 'Tamil Eelam', 'Occitania',
        'United Koreans in Japan', 'Iraqi Kurdistan', 'Barawa',
        'County of Nice', 'Western Armenia', 'Chagos Islands',
        'Ellan Vannin', 'Provence', 'Cascadia', 'Chameria', 'Crimea',
        'Felvidék', 'Kiribati', 'Micronesia', 'Saint Pierre and Miquelon',
        'Tibet']

tournament = ['FIFA World Cup', 'CONIFA European Football Cup', 'FIFA World Cup qualification',
              'Friendly', 'SAFF Cup', 'AFC Challenge Cup',
              'AFC Challenge Cup qualification', 'Merdeka Tournament',
              'AFC Asian Cup qualification', 'Malta International Tournament',
              'UEFA Euro qualification', "Prime Minister's Cup", 'UEFA Euro',
              'UEFA Nations League', 'Island Games',
              'Inter Games Football Tournament',
              'African Cup of Nations qualification', 'African Cup of Nations',
              'African Nations Championship',
              'African Nations Championship qualification', 'Arab Cup',
              'Oceania Nations Cup qualification', 'South Pacific Games',
              'Pacific Games', 'COSAFA Cup qualification', 'COSAFA Cup',
              'CFU Caribbean Cup qualification',
              'CONCACAF Nations League qualification', 'CONCACAF Nations League',
              'Gold Cup qualification', 'CFU Caribbean Cup', 'Copa América',
              'Kirin Cup', 'Confederations Cup', 'Superclásico de las Américas',
              'CONMEBOL–UEFA Cup of Champions',
              'Cyprus International Tournament', 'ABCS Tournament',
              'Oceania Nations Cup', 'AFC Asian Cup', 'EAFF Championship',
              'Millennium Cup', 'Gulf Cup', 'WAFF Championship', 'Nehru Cup',
              'Three Nations Cup', 'Mahinda Rajapaksa Cup',
              'SKN Football Festival', 'Windward Islands Tournament',
              "King's Cup", 'UNCAF Cup', 'Gold Cup', 'Amílcar Cabral Cup',
              'USA Cup', 'Copa Paz del Chaco', 'Kirin Challenge Cup',
              'Tournoi de France', 'Lunar New Year Cup', 'AFF Championship',
              'AFF Championship qualification', 'CECAFA Cup',
              'Nile Basin Tournament', 'CONIFA World Football Cup',
              'UNIFFAC Cup', 'Copa del Pacífico', 'Dunhill Cup', 'Dynasty Cup',
              'Cup of Ancient Civilizations', 'Arab Cup qualification',
              'ELF Cup', 'Korea Cup', 'Copa América qualification',
              'Nordic Championship', 'King Hassan II Tournament',
              'United Arab Emirates Friendship Tournament', 'Baltic Cup',
              'Pacific Mini Games', 'FIFI Wild Cup', 'Viva World Cup',
              'Intercontinental Cup', 'Navruz Cup', 'VFF Cup', 'Dragon Cup',
              'Copa Confraternidad', "MSG Prime Minister's Cup", 'OSN Cup',
              'TIFOCO Tournament', 'Nations Cup',
              'Jordan International Tournament', 'Afro-Asian Games',
              'World Unity Cup']

col1, col2 = st.columns(2)

col1.header('Home Team')
optionHomeTeam = col1.selectbox("Home Team", sorted(pays))
pointHomeTeam = col1.number_input("Point Home Team", 0, 9)


col2.header('Away Team')
optionAwayTeam = col2.selectbox("Away Team", sorted(pays))
pointAwayTeam = col2.number_input("Point Away Team", 0, 9)

pred = []

container = st.container()
container.header("Tournament")
optionTournament = container.selectbox("Tournament", tournament)

col3, col4, col5, col6, col7 = container.columns(5)

with col3:
    pass
with col4:
    pass
with col6:
    pass
with col7:
    pass
with col5:
    if container.button('prediction', type='primary'):
        df = pd.DataFrame({
            'home_team': [optionHomeTeam],
            'away_team': [optionAwayTeam],
            'tournament': [optionTournament],
            'home_point_rolling': [pointHomeTeam],
            'away_point_rolling': [pointAwayTeam],
        })

        pred = make_prediction(df)

if len(pred) > 0:
    if (pred[0] == 1):
        container.warning(f"{optionHomeTeam} gagne !", icon="✔")
    else:
        container.warning(f"{optionAwayTeam} gagne !", icon="✔")
