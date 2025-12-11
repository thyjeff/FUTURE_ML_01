import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. Generate Dummy Data (History)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = [100 + (x * 0.5) + (10 if x % 7 == 0 else 0) for x in range(len(dates))]
df = pd.DataFrame({'ds': dates, 'y': values})

# 2. Train the AI Model
model = Prophet()
model.fit(df)

# 3. Predict the Future (Next 90 Days)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# 4. Visualize
print("Plotting the forecast...")
model.plot(forecast)
plt.title("Sales Forecast: 2023 History + 90 Day Prediction")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# 5. Save Data
forecast.to_csv('sales_forecast_data.csv', index=False)
print("Forecast data saved to sales_forecast_data.csv")