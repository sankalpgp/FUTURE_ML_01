import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(r"C:\Users\sanka\Desktop\FUTURE_ML_01\Sample - Superstore.csv", encoding='latin-1')

df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.sort_values('Order Date')

df.head()

# Remove missing values
df = df.dropna(subset=['Sales'])

# Aggregate sales by date
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

daily_sales['Day'] = daily_sales['Order Date'].dt.day
daily_sales['Month'] = daily_sales['Order Date'].dt.month
daily_sales['Year'] = daily_sales['Order Date'].dt.year

train = daily_sales[daily_sales['Order Date'] < '2017-01-01']
test = daily_sales[daily_sales['Order Date'] >= '2017-01-01']

X_train = train[['Day', 'Month', 'Year']]
y_train = train['Sales']

X_test = test[['Day', 'Month', 'Year']]
y_test = test['Sales']

model = LinearRegression()
model.fit(X_train, y_train)

test = test.copy()
test.loc[:, 'Predicted Sales'] = model.predict(X_test)


mae = mean_absolute_error(y_test, test['Predicted Sales'])
rmse = np.sqrt(mean_squared_error(y_test, test['Predicted Sales']))

print("MAE:", mae)
print("RMSE:", rmse)

plt.figure(figsize=(12,6))
plt.plot(test['Order Date'], y_test, label='Actual Sales')
plt.plot(test['Order Date'], test['Predicted Sales'], label='Forecasted Sales')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecast vs Actual Sales")
plt.legend()
plt.show()

future_dates = pd.date_range(start=daily_sales['Order Date'].max(), periods=30)
future_df = pd.DataFrame({
    'Day': future_dates.day,
    'Month': future_dates.month,
    'Year': future_dates.year
})

future_sales = model.predict(future_df)

plt.figure(figsize=(10,5))
plt.plot(future_dates, future_sales)
plt.title("Next 30 Days Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.show()
