import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Housing.csv')

# Convert categorical columns to numeric
cat_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
for col in cat_cols:
    df[col] = df[col].replace({'yes': 1, 'no': 0, 'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

# Define target variable
y = df['price']

# --- Simple Linear Regression: price vs area ---

X_simple = df[['area']]

# Split into train and test sets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Initialize and fit the model
lr_simple = LinearRegression()
lr_simple.fit(X_train_s, y_train_s)

# Predict on test set
y_pred_s = lr_simple.predict(X_test_s)

# Evaluation metrics
mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)

print("Simple Linear Regression Results:")
print(f"MAE: {mae_s}")
print(f"MSE: {mse_s}")
print(f"R^2: {r2_s}")
print(f"Slope (Coefficient): {lr_simple.coef_[0]}")
print(f"Intercept: {lr_simple.intercept_}")

# Plot regression line
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_test_s['area'], y=y_test_s)
plt.plot(X_test_s['area'], y_pred_s, color='red', linewidth=2)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression: Price vs Area')
plt.tight_layout()
plt.savefig('simple_regression.png')
plt.show()

# --- Multiple Linear Regression: price vs all features ---

X_multi = df.drop('price', axis=1)

# Split into train and test sets
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)

# Initialize and fit the model
lr_multi = LinearRegression()
lr_multi.fit(X_train_m, y_train_m)

# Predict on test set
y_pred_m = lr_multi.predict(X_test_m)

# Evaluation metrics
mae_m = mean_absolute_error(y_test_m, y_pred_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)

print("\nMultiple Linear Regression Results:")
print(f"MAE: {mae_m}")
print(f"MSE: {mse_m}")
print(f"R^2: {r2_m}")
print(f"Intercept: {lr_multi.intercept_}")

# Print coefficients for each predictor
coefs = pd.DataFrame({'Feature': X_multi.columns, 'Coefficient': lr_multi.coef_})
print("\nCoefficients:")
print(coefs)

# Save metrics and coefficients to CSV files
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'R2', 'Slope', 'Intercept'],
    'Simple': [mae_s, mse_s, r2_s, lr_simple.coef_[0], lr_simple.intercept_],
    'Multiple': [mae_m, mse_m, r2_m, None, lr_multi.intercept_]
})

metrics_df.to_csv('regression_metrics.csv', index=False)
coefs.to_csv('multi_regression_coefs.csv', index=False)
