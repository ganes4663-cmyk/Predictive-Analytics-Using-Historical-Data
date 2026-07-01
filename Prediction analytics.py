import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("sales_data.csv")

# Remove missing values
df = df.dropna()

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Create numeric feature from Date
df["Days"] = (df["Date"] - df["Date"].min()).dt.days

# Features and Target
X = df[["Days"]]
y = df["Sales"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Predict Next 30 Days
future_days = pd.DataFrame({
    "Days": range(df["Days"].max()+1, df["Days"].max()+31)
})

future_sales = model.predict(future_days)

future_dates = pd.date_range(
    df["Date"].max() + pd.Timedelta(days=1),
    periods=30
)

# Plot
plt.figure(figsize=(10,6))
plt.scatter(df["Date"], df["Sales"], label="Historical Sales")
plt.plot(future_dates, future_sales, color="red", linewidth=2,
         label="Predicted Sales")

plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# Save Predictions
forecast = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_sales
})

forecast.to_csv("sales_forecast.csv", index=False)

print("Forecast saved as sales_forecast.csv")
