import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("data/sample.csv")

# Convert Month to numbers
data["Month_Number"] = range(1, len(data) + 1)

# Prepare input (X) and output (y)
X = data[["Month_Number"]]
y = data["Sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next month sales
next_month = [[len(data) + 1]]
prediction = model.predict(next_month)

print("Past Sales Data:")
print(data[["Month", "Sales"]])

print("\nPredicted sales for next month:", int(prediction[0]))

import matplotlib.pyplot as plt

# Plot sales graph
plt.plot(data["Month"], data["Sales"], marker='o')
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales Trend")
plt.show()
