import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'Housing.csv'
df = pd.read_csv(file_path)

# Preprocessing
# Convert categorical columns to numerical
cat_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
for col in cat_cols:
    df[col] = df[col].replace({'yes':1, 'no':0, 'furnished':2, 'semi-furnished':1, 'unfurnished':0})

# Check columns and nulls
df_info = df.info()
df_desc = df.describe()

# Simple Linear Regression: price vs area
X_simple = df[['area']]
y = df['price']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)
lr_simple = LinearRegression()
lr_simple.fit(X_train_s, y_train_s)
y_pred_s = lr_simple.predict(X_test_s)
mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)
slope_s = lr_simple.coef_[0]
intercept_s = lr_simple.intercept_

# Plot simple regression
plt.figure(figsize=(8,5))
sns.scatterplot(x=X_test_s['area'], y=y_test_s)
plt.plot(X_test_s['area'], y_pred_s, color='red', linewidth=2)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression: Price vs Area')
plt.tight_layout()
plt.savefig('simple_regression.png')
plt.close()

# Multiple Linear Regression: price vs all predictors
X_multi = df.drop('price', axis=1)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)
lr_multi = LinearRegression()
lr_multi.fit(X_train_m, y_train_m)
y_pred_m = lr_multi.predict(X_test_m)
mae_m = mean_absolute_error(y_test_m, y_pred_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)
coefs_m = dict(zip(X_multi.columns, lr_multi.coef_))
intercept_m = lr_multi.intercept_

# Return metrics and coefficients
output = {
    'simple': {
        'MAE': mae_s,
        'MSE': mse_s,
        'R2': r2_s,
        'Slope': slope_s,
        'Intercept': intercept_s
    },
    'multiple': {
        'MAE': mae_m,
        'MSE': mse_m,
        'R2': r2_m,
        'Coefficients': coefs_m,
        'Intercept': intercept_m
    }
}
output_df = pd.DataFrame({'Metric':['MAE','MSE','R2','Slope','Intercept'],
                         'Simple':[mae_s, mse_s, r2_s, slope_s, intercept_s],
                         'Multiple':[mae_m, mse_m, r2_m, None, intercept_m]})
output_df.to_csv('regression_metrics.csv', index=False)
output_coefs = pd.DataFrame({'Feature': list(coefs_m.keys()), 'Coefficient': list(coefs_m.values())})
output_coefs.to_csv('multi_regression_coefs.csv', index=False)
output_df, output_coefs
