import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template
import math

app = Flask(__name__)

# Function to categorize AQI values into categories
def categorize_aqi(aqi_value):
    if 0 <= aqi_value <= 50:
        return 'Good'
    elif 51 <= aqi_value <= 100:
        return 'Satisfactory'
    elif 101 <= aqi_value <= 200:
        return 'Moderate'
    elif 201 <= aqi_value <= 300:
        return 'Poor'
    elif 301 <= aqi_value <= 400:
        return 'Very Poor'
    else:
        return 'Severe'


# Function to retrieve AQI data for a specific city using WAQI API
def get_aqi_data(city, token):
    url = f'https://api.waqi.info/feed/{city}/?token={token}'
    response = requests.get(url)
    data = response.json()

    if 'data' in data and 'forecast' in data['data'] and 'daily' in data['data']['forecast']:
        daily_data = data['data']['forecast']['daily']
        all_dates = []
        all_aqi_values = []

        for pollutant_data in daily_data.values():
            dates = [entry['day'] for entry in pollutant_data]
            avg_aqi_values = [entry['avg'] for entry in pollutant_data]

            all_dates.extend(dates)
            all_aqi_values.extend(avg_aqi_values)

        df = pd.DataFrame({'date': all_dates, 'avg_aqi': all_aqi_values})
        df['date'] = pd.to_datetime(df['date'])
        df['category'] = df['avg_aqi'].apply(categorize_aqi)
        return df
    else:
        return None

# Function to train a Random Forest Regressor model and make predictions
def forecast_aqi(df):
    # Feature Engineering: Extracting date features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    # Splitting data into train and test sets
    X = df[['month', 'day_of_week']]
    y = df['avg_aqi']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Machine Learning Model (Random Forest Regressor)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Forecasting AQI for future dates (next 7 days)
    future_dates = pd.date_range(df['date'].max() + timedelta(days=1), periods=7, freq='D')
    future_features = pd.DataFrame({
        'date': future_dates,
        'month': future_dates.month,
        'day_of_week': future_dates.dayofweek
    })

    predicted_aqi = model.predict(future_features[['month', 'day_of_week']])
    forecast_df = pd.DataFrame({'date': future_dates, 'forecasted_aqi': predicted_aqi})
    forecast_df['forecasted_aqi'] = forecast_df['forecasted_aqi'].apply(lambda x: round(x, 0))
    forecast_df['forecasted_category'] = forecast_df['forecasted_aqi'].apply(categorize_aqi)
    return forecast_df

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city = request.form['city']
        token = 'fe6070578597b078349d9a5e4b8adf507128312e'
        city_df = get_aqi_data(city, token)
        
        if city_df is not None:
            forecast_result = forecast_aqi(city_df)
            forecast_data = forecast_result[['date', 'forecasted_aqi', 'forecasted_category']].to_dict(orient='records')
            return render_template('index.html', city=city, forecast_data=forecast_data)
        else:
            error_message = f"Failed to retrieve data for {city}."
            return render_template('index.html', error_message=error_message)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
