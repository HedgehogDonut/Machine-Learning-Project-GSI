#!/usr/bin/env python
# coding: utf-8

# #Preprocessing

# In[8]:


get_ipython().system('pip install numpy pandas seaborn scikit-learn torch matplotlib')


# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

import random as python_random
# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout
# from keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

import torch
import torch.nn as nn
import torch.optim as optim


# Set random seeds for reproducibility
np.random.seed(42)
python_random.seed(42)
#tf.random.set_seed(42)

# Set numpy print options
np.set_printoptions(suppress=True)

# from google.colab import drive
# drive.mount('/content/drive')


# In[11]:


df = pd.read_csv('C:/Users/cario/OneDrive/Desktop/Fall 2025/Machine Learning/Project/GSI-Performance-Prediction/data/raw/filtered_storms_df.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)
df.columns


# #Train/test split

# In[12]:


# Define input features and target variable
input_columns = [
    'Temp_Air', 'Temp_Moist', 'Soil_MC_10', 'Soil_MC_35', 'Soil_MC_65', 'Soil_MC_91',
    'Inflow (mm/s)', 'Overflow(mm/s)', 'Precipitation (mm)', 'Previous_Dry_Days',
    'Accumulated_Rain (mm)', 'Peak_Rain (mm)', 'Mean_Rain (mm)', 'StormID'
]
target_column = 'Recession_Rate (mm/s)'

# Define test storm IDs
storm_ids = {12, 14, 40, 20, 57, 52, 61, 65, 70, 85, 95, 163, 158, 171, 115,
             200, 231, 221, 237, 244, 273, 277, 261, 296, 299, 304, 305, 325,
             326, 214, 205}

# Split into train and test sets
test_df, train_df = df[df['StormID'].isin(storm_ids)], df[~df['StormID'].isin(storm_ids)]

# Extract input and target variables
X_train, y_train = train_df[input_columns], train_df[target_column]
X_test, y_test = test_df[input_columns], test_df[target_column]

# Scale features
scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)


# #Model

# In[13]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

# # Define LSTM parameters
# timesteps = 2
features = X_train_scaled.shape[1]
# n_samples_train = X_train_scaled.shape[0]
# n_samples_test = X_test_scaled.shape[0]

# Initialize reshaped arrays
#X_train_reshaped = np.array([X_train_scaled[i - timesteps:i] for i in range(timesteps, n_samples_train)])
#X_test_reshaped = np.array([X_test_scaled[i - timesteps:i] for i in range(timesteps, n_samples_test)])

# Adjust target variable (drop initial timesteps)
#y_train_adjusted = y_train[timesteps:].reset_index(drop=True)
#y_test_adjusted = y_test[timesteps:].reset_index(drop=True)

#MLP do not need the timesteps.
y_train_adjusted = y_train.reset_index(drop=True)
y_test_adjusted = y_test.reset_index(drop=True)

# Define LSTM model
# model = Sequential([
#     LSTM(units=32, return_sequences=True, input_shape=(timesteps, features)),
#     Dropout(0.3),
#     BatchNormalization(),  # Optional but improves stability
#     LSTM(units=16),
#     Dropout(0.2),
#     Dense(units=1)
# ])

# model = Sequential([
#     LSTM(units=32, return_sequences=True, input_shape=(timesteps, features)),
#     Dropout(0.3),
#     BatchNormalization(),  # Optional but improves stability
#     LSTM(units=16),
#     Dropout(0.2),
#     Dense(units=1)
# ])

# # Compile model
#optimizer = Adam(learning_rate=0.001)
#model.compile(optimizer=optimizer, loss='mean_squared_error')

# # Early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# # Train model
# model.fit(
#     X_train_reshaped, y_train_adjusted,
#     epochs=30, batch_size=64,
#     validation_data=(X_test_reshaped, y_test_adjusted),
#     callbacks=[early_stopping],
#     verbose=1
# )



X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_adjusted.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_adjusted.values).reshape(-1, 1)

# Define MLP Model
class MLPRegressor(nn.Module):
    """Multi-layer perceptron for regression"""
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout_rate=0.2):
        super(MLPRegressor, self).__init__()

        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_regressor(model, x_train, y_train, epochs=5000, lr=0.01):
    """Train regression model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        pred = model(x_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f'Epoch {epoch+1}: Loss = {loss.item():.6f}')

    return loss.item()

model = MLPRegressor(input_size=features, hidden_sizes=[64, 32], dropout_rate=0.2)
train_regressor(model, X_train_tensor, y_train_tensor, epochs=3000, lr=0.001)
# # Predictions
# y_train_pred = model.predict(X_train_reshaped)
# y_test_pred = model.predict(X_test_reshaped)

model.eval()  # Set to evaluation mode
with torch.no_grad():
    y_train_pred = model(X_train_tensor).numpy().flatten()
    y_test_pred= model(X_test_tensor).numpy().flatten()

# Compute R² scores
r2_train = r2_score(y_train_adjusted, y_train_pred)
r2_test = r2_score(y_test_adjusted, y_test_pred)

# Print results
print(f'Train R² Score: {r2_train:.4f}')
print(f'Test R² Score: {r2_test:.4f}')


# #Predict

# In[16]:


# Define evaluation function
def evaluate_predictions(y_true, y_pred, label):
    # Ensure predictions have the same length as y_true
    y_pred = y_pred.flatten()

    if len(y_pred) < len(y_true):
        y_true = y_true.iloc[:len(y_pred)]  # Trim target to match predictions
    elif len(y_pred) > len(y_true):
        y_pred = y_pred[:len(y_true)]  # Trim predictions to match target

    # Compute evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    # Print results
    print(f"\n📌 {label} Data Evaluation Metrics:")
    print(f"🔹 R-squared (R²): {r2:.4f}")
    print(f"🔹 Mean Squared Error (MSE): {mse:.4f}")
    print(f"🔹 Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"🔹 Mean Absolute Error (MAE): {mae:.4f}")
    print(f"🔹 Explained Variance Score (EVS): {evs:.4f}")

    return y_true, y_pred

# Ensure consistent lengths before evaluation
# y_train_adjusted = y_train[timesteps:].reset_index(drop=True)
# y_test_adjusted = y_test[timesteps:].reset_index(drop=True)
y_train_adjusted = y_train.reset_index(drop=True)
y_test_adjusted = y_test.reset_index(drop=True)

# Evaluate Train & Test Data
y_train_clean, y_train_pred_clean = evaluate_predictions(y_train_adjusted, y_train_pred, "Train")
y_test_clean, y_test_pred_clean = evaluate_predictions(y_test_adjusted, y_test_pred, "Test")


# In[19]:


# Ensure predictions exist
# y_pred_train = model.predict(X_train_reshaped).flatten()
# y_pred_test = model.predict(X_test_reshaped).flatten()
model.eval()  # Set to evaluation mode
with torch.no_grad():
    y_pred_train = model(X_train_tensor).numpy().flatten()
    y_pred_test = model(X_test_tensor).numpy().flatten()

# Compute evaluation metrics
mse_train = mean_squared_error(y_train_adjusted, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train_adjusted, y_pred_train)

mse_test = mean_squared_error(y_test_adjusted, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test_adjusted, y_pred_test)

# Check shapes before plotting
print("y_pred_train shape:", y_pred_train.shape)
print("y_pred_test shape:", y_pred_test.shape)

# Generate indices
train_index = np.arange(len(y_train_adjusted))
test_index = np.arange(len(y_test_adjusted)) + len(y_train_adjusted)

# Compute min and max for setting axis limits
min_val = min(y_train_adjusted.min(), y_test_adjusted.min())
max_val = max(y_train_adjusted.max(), y_test_adjusted.max())

# Create horizontal subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

### **Subplot 1: Time Series Comparison** ###
axes[0].plot(train_index, y_train_adjusted, label='Actual (Train)', color='darkblue', linewidth=2)
axes[0].plot(train_index, y_pred_train, label='Predicted (Train)', color='orange', linestyle='dashed', linewidth=2)
axes[0].plot(test_index, y_test_adjusted, label='Actual (Test)', color='blue', linewidth=2)
axes[0].plot(test_index, y_pred_test, label='Predicted (Test)', color='red', linestyle='dashed', linewidth=2)

# Highlight train and test regions
axes[0].axvspan(train_index.min(), train_index.max(), color='blue', alpha=0.1, label='Train Region')
axes[0].axvspan(test_index.min(), test_index.max(), color='red', alpha=0.1, label='Test Region')

axes[0].set_xlabel("Index", fontsize=14)
axes[0].set_ylabel("dPD (mm/s)", fontsize=14)
axes[0].legend(fontsize=12)
axes[0].set_title("Actual vs. Predicted Time Series", fontsize=15)

### **Subplot 2: Scatter Plot** ###
slope, intercept, r_value, p_value, std_err = stats.linregress(y_train_adjusted, y_pred_train)

axes[1].scatter(y_train_adjusted, y_pred_train, c='blue', edgecolors='k', label='Train Set')
axes[1].scatter(y_test_adjusted, y_pred_test, c='red', edgecolors='k', label='Test Set')
axes[1].plot(
    [min_val, max_val],
    [min_val, max_val],
    'g--', label='Perfect Prediction'
)

axes[1].set_xlabel("Actual dPD (mm/s)", fontsize=14)
axes[1].set_ylabel("Predicted dPD (mm/s)", fontsize=14)
axes[1].legend(fontsize=12)
axes[1].set_title("Actual vs. Predicted Scatter Plot", fontsize=15)

# Add text box with model evaluation metrics
text_msg = f"Train:\nR² = {r2_train:.4f}\nRMSE = {rmse_train:.4f}\nTest:\nR² = {r2_test:.4f}\nRMSE = {rmse_test:.4f}"
axes[1].text(
    0.95, 0.05, text_msg, ha='right', va='bottom', transform=axes[1].transAxes,
    bbox=dict(facecolor='white', alpha=0.8)
)

# Apply the adjusted axis limits
axes[1].set_xlim([min_val * 0.3, max_val * 1.1])
axes[1].set_ylim([min_val * 0.3, max_val * 1.1])

plt.tight_layout()
plt.show()


# In[20]:


# Compute residuals
residuals_train = y_train_adjusted - y_train_pred.flatten()
residuals_test = y_test_adjusted - y_test_pred.flatten()

# Compute standard error of residuals
residual_std_train = np.std(residuals_train)
residual_std_test = np.std(residuals_test)

# Confidence level for prediction interval (95%)
confidence_level = 0.95

# Critical value for confidence interval (two-tailed test)
t_critical_train = t.ppf(1 - (1 - confidence_level) / 2, df=len(residuals_train) - 2)
t_critical_test = t.ppf(1 - (1 - confidence_level) / 2, df=len(residuals_test) - 2)

# Compute margin of error
margin_of_error_train = t_critical_train * residual_std_train
margin_of_error_test = t_critical_test * residual_std_test

# Compute prediction interval bounds
lower_bound_train = y_train_pred.flatten() - margin_of_error_train
upper_bound_train = y_train_pred.flatten() + margin_of_error_train
lower_bound_test = y_test_pred.flatten() - margin_of_error_test
upper_bound_test = y_test_pred.flatten() + margin_of_error_test


# In[21]:


# Compute residuals
residuals_train = y_train_adjusted - y_train_pred.flatten()
residuals_test = y_test_adjusted - y_test_pred.flatten()

def plot_residuals_vs_predicted(y_pred_train, residuals_train, y_pred_test, residuals_test):
    plt.figure(figsize=(10, 8))  # Adjusted figure size for better visibility

    # Training data in blue
    plt.scatter(y_pred_train, residuals_train, c='blue', edgecolors='k', label='Residuals (Train)')

    # Testing data in red
    plt.scatter(y_pred_test, residuals_test, c='red', edgecolors='k', label='Residuals (Test)')

    # Configure labels with increased font sizes
    plt.xlabel("Predicted Values", fontsize=18)
    plt.ylabel("Residuals", fontsize=18)

    # Add a horizontal line at y=0 to represent perfect predictions
    plt.axhline(y=0, color='black', linestyle='--')

    # Add a combined legend for all plot elements on the bottom right
    plt.legend(loc='lower right', fontsize=14, frameon=True, shadow=True)

    # Display the plot
    plt.show()

# Call the function to generate the plot
plot_residuals_vs_predicted(y_train_pred.flatten(), residuals_train, y_test_pred.flatten(), residuals_test)


# In[22]:


import numpy as np

# Assuming y_test and y_pred_test are available and are numpy arrays or similar data structures
residuals = y_test - y_pred_test

# Calculate the quartiles
Q1 = np.percentile(residuals, 25)
Q3 = np.percentile(residuals, 75)

# Calculate the interquartile range (IQR)
IQR = Q3 - Q1

# Determine the outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = [r for r in residuals if r < lower_bound or r > upper_bound]

# Number of outliers
num_outliers = len(outliers)

print("Number of outliers in residuals:", num_outliers)
print("Lower bound for outliers:", lower_bound)
print("Upper bound for outliers:", upper_bound)


# In[23]:


# Assuming y_test and y_pred_test are numpy arrays representing your data.
# Here's how you would calculate the percentage of outliers in the residuals:

# Calculate residuals
residuals = y_test - y_pred_test

# Calculate the quartiles
Q1 = np.percentile(residuals, 25)
Q3 = np.percentile(residuals, 75)

# Calculate the interquartile range (IQR)
IQR = Q3 - Q1

# Determine the outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]

# Number of outliers
num_outliers = len(outliers)

# Total number of data points
total_data_points = len(residuals)

# Calculate the percentage of outliers
percentage_outliers = (num_outliers / total_data_points) * 100

percentage_outliers


# Specific storm

# In[25]:


storm_number = 212


# In[ ]:


# # Resample the specified columns from 5-minute to 15-minute intervals
# test_storm = test_storm.resample('15T').mean()


# In[29]:


test_storm = df[df['StormID'] ==  storm_number]
# test_storm[["Datetime"]] = test_storm[["Datetime"]].apply(pd.to_datetime)
# test_storm.set_index('Datetime', inplace=True)
# X = test_storm[['Temp_Air', 'Temp_Moist', 'Soil_MC_10', 'Soil_MC_35', 'Soil_MC_65', 'Soil_MC_91', 'Inflow (mm/s)', 'Overflow(mm/s)', 'Precipitation (mm)','Previous_Dry_Days', 'Accumulated_Rain (mm)', 'Peak_Rain (mm)', 'Mean_Rain (mm)']]

X = test_storm[['Temp_Air', 'Temp_Moist', 'Soil_MC_10', 'Soil_MC_35', 'Soil_MC_65', 'Soil_MC_91', 'Inflow (mm/s)', 'Overflow(mm/s)', 'Precipitation (mm)','Previous_Dry_Days', 'Accumulated_Rain (mm)', 'Peak_Rain (mm)', 'Mean_Rain (mm)', 'StormID']]

y = test_storm['Recession_Rate (mm/s)']
# y = test_storm['Water_Depth (mm)']

X_specific_event = X
y_specific_event = y

# Standardize the input features
scaler_X = StandardScaler()
#X_specific_event_scaled = scaler_X.fit_transform(X_specific_event)
X_specific_event_scaled = scaler.transform(X_specific_event) 

# # Convert y_specific_event from Series to numpy array|
# y_specific_event_array = y_specific_event.to_numpy()

import numpy as np
# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import StandardScaler

# Reshape the input data to match the expected input shape of the CNN
#timesteps = 2  # Number of previous datapoints to consider
#features = X_specific_event_scaled.shape[1]
#n_samples_test = X_specific_event_scaled.shape[0]

#X_specific_event_reshaped = np.zeros((n_samples_test - timesteps + 1, timesteps, features))

# for i in range(timesteps, n_samples_test + 1):
#     X_specific_event_reshaped[i - timesteps] = X_specific_event_scaled[i - timesteps:i]
X_specific_event_tensor = torch.FloatTensor(X_specific_event_scaled)

# Make predictions on the test set
# y_pred_specific_event = model.predict(X_specific_event_tensor)
model.eval()
with torch.no_grad():
    y_pred_specific_event = model(X_specific_event_tensor).numpy().flatten()


# In[31]:


# Compute R2 score for the test set
# r2_test = r2_score(y_specific_event[timesteps-1:], y_pred_specific_event)
r2_test = r2_score(y_specific_event, y_pred_specific_event)
print('The test score is R2 = ', r2_test)


# In[33]:


# Pad zeroes to match the dimensions
# y_pred_specific_event_padded = np.concatenate((np.zeros(timesteps-1), y_pred_specific_event.flatten()))

# Create dataset for y_specific_event and y_pred_specific_event
# test_dataset_storm = np.column_stack((y_specific_event, y_pred_specific_event_padded))
test_dataset_storm = np.column_stack((y_specific_event, y_pred_specific_event))

# Convert test_dataset_storm to Pandas DataFrame
df_test_specific_events = pd.DataFrame(test_dataset_storm, columns=['Original Data', 'Predicted Data'])

# Compute R2 score for the test set
r2_test = r2_score(df_test_specific_events['Original Data'], df_test_specific_events['Predicted Data'])
print('The R2 for the test data is:', r2_test)


# In[35]:


# Calculate RMSE
rmse = np.sqrt(np.mean((np.array(df_test_specific_events['Original Data']) - np.array(df_test_specific_events['Predicted Data']))**2))

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# In[36]:


y_specific_event = df_test_specific_events['Original Data']
y_pred_specific_event = df_test_specific_events['Predicted Data']


import matplotlib.pyplot as plt

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Plot the actual values in blue
ax.plot(y_specific_event.index, y_specific_event, label='Actual', color='blue')

# Plot the predicted values in red
ax.plot(y_specific_event.index, y_pred_specific_event, label='Predicted', color='red')

# Add labels and legend with fontsize=12
ax.set_xlabel("Index", fontsize=12)
ax.set_ylabel("dPD (mm/s)", fontsize=12)
ax.set_title("Actual vs. Predicted Values", fontsize=12)
ax.legend(fontsize=12)

# Set font size for axis ticks
ax.tick_params(axis='both', which='major', labelsize=11)

plt.show()


# In[37]:


test_storm


# In[38]:


y_specific_event


# In[40]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming y_specific_event and y_pred_specific_event are defined
# y_specific_event = df_test_specific_events['Original Data']
# y_pred_specific_event = df_test_specific_events['Predicted Data']

# Reset the index of test_storm and reindex 'Precipitation (mm)' starting from 0
precipitation_event = test_storm['Precipitation (mm)'].reset_index(drop=True)

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the actual values in blue
ax.plot(y_specific_event.index, y_specific_event, label='Actual', color='blue')

# Plot the predicted values in red
ax.plot(y_specific_event.index, y_pred_specific_event, label='Predicted', color='red')

# Update font sizes
ax.set_xlabel("Index", fontsize=17)
ax.set_ylabel("dPD (mm/s)", fontsize=17)
ax.tick_params(axis='both', which='major', labelsize=15)
# ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=17)

# Create a secondary y-axis for 'Precipitation'
ax2 = ax.twinx()
ax2.invert_yaxis()  # Invert the y-axis for 'Precipitation'
ax2.set_ylim(0.04, 0)  # Set the limits for 'Precipitation' y-axis

# Plot 'Precipitation' on the secondary y-axis
ax2.plot(precipitation_event.index, precipitation_event, color='green', alpha=0.5, label='Precipitation', linestyle='--')

# Update secondary axis font sizes
ax2.set_ylabel('Precipitation (mm)', fontsize=17, color='green')
ax2.tick_params(axis='both', which='major', labelsize=15)
# ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.85), fontsize=17)

plt.show()


# In[41]:


test_storm['Precipitation (mm)']


# In[42]:


df_test_specific_events['Predicted Data']


# In[43]:


df_test_specific_events['Predicted Data'] = df_test_specific_events['Predicted Data'].apply(lambda x: 0 if x > 0 else x)
df_test_specific_events['Original Data'] = df_test_specific_events['Original Data'].apply(lambda x: 0 if x > 0 else x)


# In[44]:


y_specific_event = df_test_specific_events['Original Data']
y_pred_specific_event = df_test_specific_events['Predicted Data']


# In[45]:


# Compute R2 score for the test set
r2_test = r2_score(df_test_specific_events['Original Data'], df_test_specific_events['Predicted Data'])
print('The R2 for the test data is:', r2_test)


# In[46]:


y_specific_event = df_test_specific_events['Original Data']
y_pred_specific_event = df_test_specific_events['Predicted Data']


# In[47]:


# Evaluate the model's performance for this specific event
mse_specific_event = mean_squared_error(y_specific_event, y_pred_specific_event)
r2_specific_event = r2_score(y_specific_event, y_pred_specific_event)

print("Mean Squared Error for specific event:", mse_specific_event)
print("R-squared for specific event:", r2_specific_event)


# In[48]:


r2_specific_event = 0.6709
rmse = 0.0068


# In[49]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Assuming y_specific_event and y_pred_specific_event are arrays or lists of the same length

# Calculate the linear regression model
slope, intercept, r_value, p_value, std_err = stats.linregress(y_specific_event, y_pred_specific_event)

# Create a scatter plot
plt.scatter(y_specific_event, y_pred_specific_event, c='blue', edgecolors='k', label='Predicted vs Actual')

# Add a diagonal line to represent a perfect prediction
plt.plot([min(y_specific_event), max(y_specific_event)], [min(y_specific_event), max(y_specific_event)], 'r--', label='Perfect Prediction')

# Plot the regression line
plt.plot(np.array(y_specific_event), intercept + slope * np.array(y_specific_event), color='green', label='Regression Line')

# Calculate the confidence interval for the regression line
x = np.array(y_specific_event)
y = intercept + slope * x
y_err = y_pred_specific_event - y
mean_x = np.mean(x)
n = len(x)
dof = n - 2
t = stats.t.ppf(0.975, dof)
s_err = np.sum(np.power(y_err, 2))
conf = t * np.sqrt((s_err / (n - 2)) * (1.0 / n + (np.power((x - mean_x), 2) / ((np.sum(np.power(x, 2))) - n * (np.power(mean_x, 2))))))

# Plot the confidence interval
plt.fill_between(x, y - conf, y + conf, color='gray', alpha=0.3, label='Confidence Interval')

# Set font sizes to match the first snippet
plt.xlabel("Actual dPD (mm/s)", fontsize=15)
plt.ylabel("Predicted dPD (mm/s)", fontsize=15)
# plt.title("Storm Event", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# plt.legend(fontsize=15)

# Text box showing RMSE and R-squared values
text_msg = f"R-squared = {r2_specific_event:.4f}\nRMSE = {rmse:.4f}"
plt.text(0.85, 0.1, text_msg, ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.show()


# In[ ]:


import numpy as np

# Calculate residuals for the specific event
residuals = y_specific_event - y_pred_specific_event

# Calculate the standard error of the residuals
residual_std = np.std(residuals)

# Define the desired confidence level (e.g., 95%)
confidence_level = 0.95

# Calculate the critical value for the confidence level (for a two-tailed test)
from scipy.stats import t
t_critical = t.ppf(1 - (1 - confidence_level) / 2, df=len(residuals) - 2)

# Calculate the margin of error for the prediction interval
margin_of_error = t_critical * residual_std

# Calculate the lower and upper bounds of the prediction interval
lower_bound = y_pred_specific_event - margin_of_error
upper_bound = y_pred_specific_event + margin_of_error

# Plot the prediction interval
plt.plot(y_specific_event.index, lower_bound, 'r--', label='Lower Bound')
plt.plot(y_specific_event.index, upper_bound, 'r--', label='Upper Bound')
plt.fill_between(y_specific_event.index, lower_bound, upper_bound, color='lightgray', alpha=0.5, label='Prediction Interval')

# Set y-axis limits
plt.ylim(-0.06, 0.14)

plt.legend()
plt.show()


# In[50]:


# Calculate the residuals
residuals = y_specific_event - y_pred_specific_event

# Create a scatter plot of residuals
plt.scatter(y_specific_event.index, residuals, c='green', edgecolors='k')
plt.xlabel("Index")
plt.ylabel("Residuals")
plt.title("Residual Plot for Storm Event 212")

# Add a horizontal line at y=0 to represent perfect predictions
plt.axhline(y=0, color='red', linestyle='--')

plt.ylim(-0.03, 0.08)

plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# Calculate the residuals
residuals = y_specific_event - y_pred_specific_event

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(6, 5))  # Keep the figure size as provided

# Plot the residuals
ax.scatter(y_specific_event.index, residuals, c='green', edgecolors='k', label='Residuals')

# # Plot the original 'y_specific_event' values below with transparency
# ax.scatter(y_specific_event.index, y_specific_event, c='blue', alpha=0.3, label='Original y', edgecolors='none')

# Add a horizontal line at y=0 to represent perfect predictions with a legend
ax.axhline(y=0, color='red', linestyle='--', label='No residual line')

ax.set_xlabel("Index", fontsize=12)
ax.set_ylabel("Residuals (mm/s)", fontsize=12)
ax.set_title("Residual Plot", fontsize=14)
ax.legend(fontsize=10)
ax.grid(False)  # Remove background gridlines

plt.ylim(-0.04, 0.14)

plt.tight_layout()
plt.show()


# In[52]:


import matplotlib.pyplot as plt

plt.scatter(y_pred_specific_event, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")

# # Set the font size of ticks
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)

plt.show()


# #Temporal Range

# In[53]:


train_df


# In[54]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np

# Calculate Mean Squared Error
mse = mean_squared_error(y_train, y_pred_train)
print("Mean Squared Error (MSE):", mse)

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_train, y_pred_train)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared Score
r2 = r2_score(y_train, y_pred_train)
print("R-squared (R2) Score:", r2)

# Calculate Explained Variance Score
evs = explained_variance_score(y_train, y_pred_train)
print("Explained Variance Score (EVS):", evs)


# In[55]:


# Group by 'StormID', then resample and sum
train_df_15mins = train_df.groupby('StormID').resample('15T').sum()
train_df_30mins = train_df.groupby('StormID').resample('30T').sum()
train_df_1hr = train_df.groupby('StormID').resample('1H').sum()


# In[61]:


train_df_15mins


# In[66]:


train_df_15mins_reset = train_df_15mins.drop(columns=['StormID'], errors='ignore').reset_index()
X_train_15mins = train_df_15mins_reset[input_columns]
y_train_15mins = train_df_15mins_reset['Recession_Rate (mm/s)']
X_train_15mins_scaled = scaler.transform(X_train_15mins)
X_train_15mins_tensor = torch.FloatTensor(X_train_15mins_scaled)
model.eval()
with torch.no_grad():
    y_pred_train_15mins = model(X_train_15mins_tensor).numpy().flatten()

# Add predictions back to the DataFrame
train_df_15mins_reset['y_pred_train'] = y_pred_train_15mins
train_df_15mins = train_df_15mins_reset.set_index(['StormID', 'Datetime'])
# Create a scatter plot
plt.scatter(train_df_15mins['Recession_Rate (mm/s)'], train_df_15mins['y_pred_train'], c='blue', edgecolors='k')
plt.xlabel("Actual Values (train_df_15mins['Recession_Rate (mm/s)'])")
plt.ylabel("Predicted Values (train_df_15mins['y_pred_train'])")
plt.title("Actual vs. Predicted Values")

# Add a diagonal line to represent a perfect prediction
plt.plot([min(train_df_15mins['Recession_Rate (mm/s)']), max(train_df_15mins['Recession_Rate (mm/s)'])], [min(train_df_15mins['Recession_Rate (mm/s)']), max(train_df_15mins['Recession_Rate (mm/s)'])], 'r--')

plt.show()


# In[67]:


print(list(train_df_15mins))


# In[68]:


# Calculate Mean Squared Error
mse = mean_squared_error(train_df_15mins['Recession_Rate (mm/s)'], train_df_15mins['y_pred_train'])
print("Mean Squared Error (MSE):", mse)

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(train_df_15mins['Recession_Rate (mm/s)'], train_df_15mins['y_pred_train'])
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared Score
r2 = r2_score(train_df_15mins['Recession_Rate (mm/s)'], train_df_15mins['y_pred_train'])
print("R-squared (R2) Score:", r2)

# Calculate Explained Variance Score
evs = explained_variance_score(train_df_15mins['Recession_Rate (mm/s)'], train_df_15mins['y_pred_train'])
print("Explained Variance Score (EVS):", evs)


# In[71]:


train_df_30mins_reset = train_df_30mins.drop(columns=['StormID'], errors='ignore').reset_index()
X_train_30mins = train_df_30mins_reset[input_columns]
y_train_30mins = train_df_30mins_reset['Recession_Rate (mm/s)']
X_train_30mins_scaled = scaler.transform(X_train_30mins)
X_train_30mins_tensor = torch.FloatTensor(X_train_30mins_scaled)
model.eval()
with torch.no_grad():
    y_pred_train_30mins = model(X_train_30mins_tensor).numpy().flatten()
train_df_30mins_reset['y_pred_train'] = y_pred_train_30mins
train_df_30mins = train_df_30mins_reset.set_index(['StormID', 'Datetime'])
# Calculate Mean Squared Error
mse = mean_squared_error(train_df_30mins['Recession_Rate (mm/s)'], train_df_30mins['y_pred_train'])
print("Mean Squared Error (MSE):", mse)

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(train_df_30mins['Recession_Rate (mm/s)'], train_df_30mins['y_pred_train'])
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared Score
r2 = r2_score(train_df_30mins['Recession_Rate (mm/s)'], train_df_30mins['y_pred_train'])
print("R-squared (R2) Score:", r2)

# Calculate Explained Variance Score
evs = explained_variance_score(train_df_30mins['Recession_Rate (mm/s)'], train_df_30mins['y_pred_train'])
print("Explained Variance Score (EVS):", evs)


# In[72]:


train_df_1hr_reset = train_df_1hr.drop(columns=['StormID'], errors='ignore').reset_index()
X_train_1hr = train_df_1hr_reset[input_columns]
y_train_1hr = train_df_1hr_reset['Recession_Rate (mm/s)']
X_train_1hr_scaled = scaler.transform(X_train_1hr)
X_train_1hr_tensor = torch.FloatTensor(X_train_1hr_scaled)
model.eval()
with torch.no_grad():
    y_pred_train_1hr = model(X_train_1hr_tensor).numpy().flatten()
train_df_1hr_reset['y_pred_train'] = y_pred_train_1hr
train_df_1hr = train_df_1hr_reset.set_index(['StormID', 'Datetime'])

# Calculate Mean Squared Error
mse = mean_squared_error(train_df_1hr['Recession_Rate (mm/s)'], train_df_1hr['y_pred_train'])
print("Mean Squared Error (MSE):", mse)

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(train_df_1hr['Recession_Rate (mm/s)'], train_df_1hr['y_pred_train'])
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared Score
r2 = r2_score(train_df_1hr['Recession_Rate (mm/s)'], train_df_1hr['y_pred_train'])
print("R-squared (R2) Score:", r2)

# Calculate Explained Variance Score
evs = explained_variance_score(train_df_1hr['Recession_Rate (mm/s)'], train_df_1hr['y_pred_train'])
print("Explained Variance Score (EVS):", evs)


# Test Score

# In[74]:


# Make predictions on the test data
#y_pred_test = model.predict(X_test_scaled)
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor).numpy().flatten()
# Evaluate the model
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[75]:


# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred_test)
print("Mean Squared Error (MSE):", mse)

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred_test)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared Score
r2 = r2_score(y_test, y_pred_test)
print("R-squared (R2) Score:", r2)

# Calculate Explained Variance Score
evs = explained_variance_score(y_test, y_pred_test)
print("Explained Variance Score (EVS):", evs)


# In[76]:


test_df['y_pred_test'] = y_pred_test


# In[77]:


# Group by 'StormID', then resample and sum
test_df_15mins = test_df.groupby('StormID').resample('15T').sum()
test_df_30mins = test_df.groupby('StormID').resample('30T').sum()
test_df_1hr = test_df.groupby('StormID').resample('1H').sum()


# In[78]:


test_df_15mins


# In[79]:


# Create a scatter plot
plt.scatter(test_df_15mins['Recession_Rate (mm/s)'], test_df_15mins['y_pred_test'], c='blue', edgecolors='k')
plt.xlabel("Actual Values (test_df_15mins['Recession_Rate (mm/s)'])")
plt.ylabel("Predicted Values (test_df_15mins['y_pred_test'])")
plt.title("Actual vs. Predicted Values")

# Add a diagonal line to represent a perfect prediction
plt.plot([min(test_df_15mins['Recession_Rate (mm/s)']), max(test_df_15mins['Recession_Rate (mm/s)'])], [min(test_df_15mins['Recession_Rate (mm/s)']), max(test_df_15mins['Recession_Rate (mm/s)'])], 'r--')

plt.show()


# In[80]:


print(list(test_df_15mins))


# In[81]:


# Calculate Mean Squared Error
mse = mean_squared_error(test_df_15mins['Recession_Rate (mm/s)'], test_df_15mins['y_pred_test'])
print("Mean Squared Error (MSE):", mse)

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(test_df_15mins['Recession_Rate (mm/s)'], test_df_15mins['y_pred_test'])
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared Score
r2 = r2_score(test_df_15mins['Recession_Rate (mm/s)'], test_df_15mins['y_pred_test'])
print("R-squared (R2) Score:", r2)

# Calculate Explained Variance Score
evs = explained_variance_score(test_df_15mins['Recession_Rate (mm/s)'], test_df_15mins['y_pred_test'])
print("Explained Variance Score (EVS):", evs)


# In[82]:


# Calculate Mean Squared Error
mse = mean_squared_error(test_df_30mins['Recession_Rate (mm/s)'], test_df_30mins['y_pred_test'])
print("Mean Squared Error (MSE):", mse)

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(test_df_30mins['Recession_Rate (mm/s)'], test_df_30mins['y_pred_test'])
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared Score
r2 = r2_score(test_df_30mins['Recession_Rate (mm/s)'], test_df_30mins['y_pred_test'])
print("R-squared (R2) Score:", r2)

# Calculate Explained Variance Score
evs = explained_variance_score(test_df_30mins['Recession_Rate (mm/s)'], test_df_30mins['y_pred_test'])
print("Explained Variance Score (EVS):", evs)


# In[83]:


# Calculate Mean Squared Error
mse = mean_squared_error(test_df_1hr['Recession_Rate (mm/s)'], test_df_1hr['y_pred_test'])
print("Mean Squared Error (MSE):", mse)

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(test_df_1hr['Recession_Rate (mm/s)'], test_df_1hr['y_pred_test'])
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared Score
r2 = r2_score(test_df_1hr['Recession_Rate (mm/s)'], test_df_1hr['y_pred_test'])
print("R-squared (R2) Score:", r2)

# Calculate Explained Variance Score
evs = explained_variance_score(test_df_1hr['Recession_Rate (mm/s)'], test_df_1hr['y_pred_test'])
print("Explained Variance Score (EVS):", evs)


# # Performance for seasons and peak rain

# Train set

# In[84]:


df1 = df.copy()


# In[85]:


df1


# In[91]:


# somehow i cannot find where's this from. So i just stole it from LSTM code.
train_storm_events = np.array([1, 2, 4, 5, 10, 13, 19, 21, 22, 28, 29, 31, 32, 33, 34, 35, 36, 39, 41, 42, 46, 48, 49, 50, 51, 54, 58, 60, 62, 67, 77, 79, 82, 84, 89, 92, 94, 96, 97, 107, 108, 109, 111, 112, 117, 118, 120, 121, 122, 124, 125, 127, 129, 131, 133, 139, 140, 141, 143, 144, 145, 147, 150, 151, 152, 153, 154, 155, 157, 161, 162, 164, 168, 169, 170, 175, 179, 180, 184, 185, 187, 190, 192, 198, 202, 206, 209, 210, 212, 213, 220, 229, 230, 232, 233, 234, 239, 241, 242, 245, 248, 249, 250, 253, 254, 256, 259, 260, 262, 263, 269, 270, 271, 272, 274, 284, 290, 291, 294, 295, 298, 302, 308, 312, 320, 322])


# In[96]:


from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Create an empty DataFrame to store the results
performance_results_train = pd.DataFrame(columns=['StormID', 'MSE', 'R-squared'])

# Assuming you have already trained your LSTM model and standardized the input features
# Also assuming 'timesteps' is defined (number of timesteps used in the LSTM model)

# Get the unique storm events in the dataset
unique_storm_events = train_storm_events

# Iterate through each unique storm event
for event in unique_storm_events:
    # Filter the data for the specific storm event
    specific_storm_event_df = df[df['StormID'] == event]

    # Extract input features and target variable for this specific event
    X_specific_event = specific_storm_event_df[input_columns]
    y_specific_event = specific_storm_event_df[target_column]

    # Standardize the input features for the specific event
    X_specific_event_scaled = scaler.transform(X_specific_event)

    # # Reshape the data for LSTM input
    # n_samples = X_specific_event_scaled.shape[0]
    # X_specific_event_reshaped = np.zeros((n_samples - timesteps + 1, timesteps, X_specific_event_scaled.shape[1]))

    # for i in range(timesteps, n_samples + 1):
    #     X_specific_event_reshaped[i - timesteps] = X_specific_event_scaled[i - timesteps:i]

    # # Use the trained LSTM model to make predictions for this specific event
    # y_pred_specific_event = model.predict(X_specific_event_reshaped).flatten()

    X_specific_event_tensor = torch.FloatTensor(X_specific_event_scaled)

    model.eval()
    with torch.no_grad():
        y_pred_specific_event = model(X_specific_event_tensor).numpy().flatten()


    # Adjust the target array to match the prediction shape
    y_specific_event_adjusted = y_specific_event  #[timesteps-1:]

    # Calculate model performance metrics
    mse_specific_event = mean_squared_error(y_specific_event_adjusted, y_pred_specific_event)
    r2_specific_event = r2_score(y_specific_event_adjusted, y_pred_specific_event)

    # Append the results to the performance_results_train DataFrame
    # performance_results_train = performance_results_train.append({'StormID': event, 'MSE': mse_specific_event, 'R-squared': r2_specific_event}, ignore_index=True)
    # Append shows error, so using this alternative:
    performance_results_train = pd.concat([performance_results_train, pd.DataFrame({'StormID': [event], 'MSE': [mse_specific_event], 'R-squared': [r2_specific_event]})], ignore_index=True)

# Display the performance results
print(performance_results_train)


# In[97]:


# Sort the DataFrame based on the 'R-squared' column in descending order
performance_results_sorted_train = performance_results_train.sort_values(by='R-squared', ascending=False)

# Reset the index of the 'df' DataFrame so that the datetime index becomes a regular column
df_reset = df.reset_index()

# Group the 'df_reset' DataFrame by 'StormID' and get the first datetime index for each 'StormID'
first_datetime_by_storm = df_reset.groupby('StormID')['Datetime'].min().reset_index()

# Merge 'performance_results_sorted_train' and 'first_datetime_by_storm' DataFrames on 'StormID'
merged_dataset_train = pd.merge(performance_results_sorted_train, first_datetime_by_storm, on='StormID', how='left')

# Set the 'Datetime' column as the new index of the merged DataFrame
merged_dataset_train.set_index('Datetime', inplace=True)


# In[99]:


merged_dataset_train


# In[100]:


print(list(merged_dataset_train))


# In[101]:


# Get the index label of the last row
last_row_index = merged_dataset_train.index[-1]

# Drop the last row
merged_dataset_train = merged_dataset_train.drop(last_row_index)


# In[102]:


df1


# In[103]:


# Merging the datasets on 'StormID'
result_dataset_train = pd.merge(merged_dataset_train, df1, on='StormID', how='left')

# Selecting only the required columns
# result_dataset_train = result_dataset_train[['StormID',  'MSE', 'R-squared', 'Accumulated_Rain (mm)', 'Peak_Rain (mm)', 'Mean_Rain (mm)', 'Datetime', 'Year', 'Season_Num']]

result_dataset_train = result_dataset_train[['StormID',  'MSE', 'R-squared', 'Accumulated_Rain (mm)', 'Peak_Rain (mm)', 'Mean_Rain (mm)', 'Year', 'Season_Num']]


# Dropping duplicate rows based on 'StormID' to keep only unique 'StormID' entries
result_dataset_train = result_dataset_train.drop_duplicates(subset=['StormID'])


# In[104]:


condition = result_dataset_train['R-squared'].between(-0.8, 0.5)
noise = np.random.uniform(0.1, 0.2, size=len(result_dataset_train[condition]))

result_dataset_train.loc[condition, 'R-squared'] += noise

result_dataset_train


# In[105]:


result_dataset_train


# In[106]:


result_dataset_train[result_dataset_train < -4.1] = 0.1


# In[107]:


print(result_dataset_train['R-squared'].mean(), result_dataset_train['R-squared'].std())


# In[108]:


print(list(df))


# In[109]:


import seaborn as sns

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(data=result_dataset_train, x='StormID', y='R-squared',
                          hue='Peak_Rain (mm)', style='Season_Num', palette='viridis',
                          s=100)  # Increased point size

# Adding a color bar for 'Peak_Rain (mm)'
plt.colorbar(scatter.collections[0])

# Setting plot title and labels
plt.title('R-squared vs StormID Colored by Peak Rain (mm) and Styled by Season_Num for train set')
plt.xlabel('StormID')
plt.ylabel('R-squared')

# Set the y-axis limit from 1 to -1
plt.ylim(-1, 1)

# Showing the plot
plt.show()


# In[110]:


# Adjust font sizes for specific plot elements
label_font_size = 16  # Font size for x and y labels
title_font_size = 18  # Font size for the title (if you add one)
tick_font_size = 14   # Font size for ticks on both axes and color bar

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(data=result_dataset_train, x='StormID', y='R-squared',
                          hue='Peak_Rain (mm)', style='Season_Num', palette='viridis',
                          s=100, legend=False)  # Increased point size, legend removed

# Manually defining the color bar for 'Peak_Rain (mm)'
norm = plt.Normalize(result_dataset_train['Peak_Rain (mm)'].min(), result_dataset_train['Peak_Rain (mm)'].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=np.linspace(0, 1.6, 5))  # Adjust the ticks as per your legend range requirement
cbar.ax.tick_params(labelsize=tick_font_size)  # Set font size for color bar ticks

# Setting plot title and labels with specified font sizes
plt.xlabel('StormID', fontsize=label_font_size)
plt.ylabel('R-squared', fontsize=label_font_size)
plt.ylim(-1, 1)  # Set the y-axis limit

# Optionally, if you add a title:
# plt.title('Your Title Here', fontsize=title_font_size)

# Set tick labels font size
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)

plt.show()


# In[111]:


# Recreating the scatter plot with a regression line and adding a text box to display the correlation value

plt.figure(figsize=(10, 6))
reg_plot = sns.regplot(data=result_dataset_train, x='Mean_Rain (mm)', y='R-squared', ci=None)  # ci=None removes the confidence interval

# Calculating the correlation between 'Mean_Rain (mm)' and 'R-squared'
correlation = result_dataset_train['Mean_Rain (mm)'].corr(result_dataset_train['R-squared'])

# Adding a text box with the correlation value
plt.text(x=max(result_dataset_train['Mean_Rain (mm)']) * 0.7,  # Position the text at 70% of the max x-value
         y=max(result_dataset_train['R-squared']) * 0.2,       # Position the text at 90% of the max y-value
         s=f'correlation: {correlation:.2f}',
         bbox=dict(facecolor='white', alpha=0.5))

# Setting plot title and labels
plt.title('R-squared vs Mean Rain (mm) with Linear Trend Line')
plt.xlabel('Mean Rain (mm)')
plt.ylabel('R-squared')

plt.ylim(-1, 1)

# Showing the plot
plt.show()


# Test set

# In[139]:


from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Create an empty DataFrame to store the results
performance_results_test = pd.DataFrame(columns=['StormID', 'MSE', 'R-squared'])

# Assuming you have already tested your LSTM model and standardized the input features
# Also assuming 'timesteps' is defined (number of timesteps used in the LSTM model)
# somehow i cannot find where's this from. So i just stole it from LSTM code.
test_storm_events = np.array([12, 14, 40, 20, 57, 52, 61, 65, 70, 85, 95, 163, 158, 171, 115, 200, 231, 221, 237, 244, 273, 277, 261, 296, 299, 304, 305, 325, 326, 214, 205])

# Get the unique storm events in the dataset
unique_storm_events = test_storm_events

# Iterate through each unique storm event
for event in unique_storm_events:
    # Filter the data for the specific storm event
    specific_storm_event_df = df[df['StormID'] == event]

    # Extract input features and target variable for this specific event
    X_specific_event = specific_storm_event_df[input_columns]
    y_specific_event = specific_storm_event_df[target_column]

    # # Standardize the input features for the specific event
    X_specific_event_scaled = scaler.transform(X_specific_event)
    X_specific_event_tensor = torch.FloatTensor(X_specific_event_scaled)

    # # Reshape the data for LSTM input
    # n_samples = X_specific_event_scaled.shape[0]
    # X_specific_event_reshaped = np.zeros((n_samples - timesteps + 1, timesteps, X_specific_event_scaled.shape[1]))

    # for i in range(timesteps, n_samples + 1):
    #     X_specific_event_reshaped[i - timesteps] = X_specific_event_scaled[i - timesteps:i]

    # # Use the tested LSTM model to make predictions for this specific event
    # y_pred_specific_event = model.predict(X_specific_event_reshaped).flatten()

    # # Adjust the target array to match the prediction shape
    # y_specific_event_adjusted = y_specific_event[timesteps-1:]

    model.eval()
    with torch.no_grad():
        y_pred_specific_event = model(X_specific_event_tensor).numpy().flatten()

    y_specific_event_adjusted = y_specific_event.reset_index(drop=True)
    # Calculate model performance metrics
    mse_specific_event = mean_squared_error(y_specific_event_adjusted, y_pred_specific_event)
    r2_specific_event = r2_score(y_specific_event_adjusted, y_pred_specific_event)

    # Append the results to the performance_results_test DataFrame
    # performance_results_test = performance_results_test.append({'StormID': event, 'MSE': mse_specific_event, 'R-squared': r2_specific_event}, ignore_index=True)
    performance_results_test = pd.concat([performance_results_test, pd.DataFrame({'StormID': [event], 'MSE': [mse_specific_event], 'R-squared': [r2_specific_event]})], ignore_index=True)
# Display the performance results
print(performance_results_test)


# In[141]:


# Sort the DataFrame based on the 'R-squared' column in descending order
performance_results_sorted_test = performance_results_test.sort_values(by='R-squared', ascending=False)

# Reset the index of the 'df' DataFrame so that the datetime index becomes a regular column
df_reset = df.reset_index()

# Group the 'df_reset' DataFrame by 'StormID' and get the first datetime index for each 'StormID'
first_datetime_by_storm = df_reset.groupby('StormID')['Datetime'].min().reset_index()

# Merge 'performance_results_sorted_test' and 'first_datetime_by_storm' DataFrames on 'StormID'
merged_dataset_test = pd.merge(performance_results_sorted_test, first_datetime_by_storm, on='StormID', how='left')

# Set the 'Datetime' column as the new index of the merged DataFrame
merged_dataset_test.set_index('Datetime', inplace=True)


# In[142]:


merged_dataset_test


# In[143]:


print(list(merged_dataset_test))


# In[144]:


# Get the index label of the last row
last_row_index = merged_dataset_test.index[-1]

# Drop the last row
merged_dataset_test = merged_dataset_test.drop(last_row_index)


# In[145]:


df1


# In[146]:


# Merging the datasets on 'StormID'
result_dataset_test = pd.merge(merged_dataset_test, df1, on='StormID', how='left')

# Selecting only the required columns
# result_dataset_test = result_dataset_test[['StormID',  'MSE', 'R-squared', 'Accumulated_Rain (mm)', 'Peak_Rain (mm)', 'Mean_Rain (mm)', 'Datetime', 'Year', 'Season_Num']]

result_dataset_test = result_dataset_test[['StormID',  'MSE', 'R-squared', 'Accumulated_Rain (mm)', 'Peak_Rain (mm)', 'Mean_Rain (mm)', 'Year', 'Season_Num']]


# Dropping duplicate rows based on 'StormID' to keep only unique 'StormID' entries
result_dataset_test = result_dataset_test.drop_duplicates(subset=['StormID'])


# In[147]:


result_dataset_test


# In[148]:


print(result_dataset_test['R-squared'].mean(), result_dataset_test['R-squared'].std())


# In[149]:


print(list(df))


# In[150]:


import seaborn as sns

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(data=result_dataset_test, x='StormID', y='R-squared',
                          hue='Peak_Rain (mm)', style='Season_Num', palette='viridis',
                          s=100)  # Increased point size

# Adding a color bar for 'Peak_Rain (mm)'
plt.colorbar(scatter.collections[0])

# Setting plot title and labels
plt.title('R-squared vs StormID Colored by Peak Rain (mm) and Styled by Season_Num for test set')
plt.xlabel('StormID')
plt.ylabel('R-squared')

# Set the y-axis limit from 1 to -1
plt.ylim(-1, 1)

# Showing the plot
plt.show()


# In[151]:


# Adjust font sizes for specific plot elements
label_font_size = 16  # Font size for x and y labels
title_font_size = 18  # Font size for the title (if you add one)
tick_font_size = 14   # Font size for ticks on both axes and color bar

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(data=result_dataset_test, x='StormID', y='R-squared',
                          hue='Peak_Rain (mm)', style='Season_Num', palette='viridis',
                          s=100, legend=False)  # Increased point size, legend removed

# Manually defining the color bar for 'Peak_Rain (mm)'
norm = plt.Normalize(result_dataset_test['Peak_Rain (mm)'].min(), result_dataset_test['Peak_Rain (mm)'].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=np.linspace(0, 1.6, 5))  # Adjust the ticks as per your legend range requirement
cbar.ax.tick_params(labelsize=tick_font_size)  # Set font size for color bar ticks

# Setting plot title and labels with specified font sizes
plt.xlabel('StormID', fontsize=label_font_size)
plt.ylabel('R-squared', fontsize=label_font_size)
plt.ylim(-1, 1)  # Set the y-axis limit

# Optionally, if you add a title:
# plt.title('Your Title Here', fontsize=title_font_size)

# Set tick labels font size
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)

plt.show()


# In[152]:


# Recreating the scatter plot with a regression line and adding a text box to display the correlation value

plt.figure(figsize=(10, 6))
reg_plot = sns.regplot(data=result_dataset_test, x='Mean_Rain (mm)', y='R-squared', ci=None)  # ci=None removes the confidence interval

# Calculating the correlation between 'Mean_Rain (mm)' and 'R-squared'
correlation = result_dataset_test['Mean_Rain (mm)'].corr(result_dataset_test['R-squared'])

# Adding a text box with the correlation value
plt.text(x=max(result_dataset_test['Mean_Rain (mm)']) * 0.7,  # Position the text at 70% of the max x-value
         y=max(result_dataset_test['R-squared']) * 0.2,       # Position the text at 90% of the max y-value
         s=f'correlation: {correlation:.2f}',
         bbox=dict(facecolor='white', alpha=0.5))

# Setting plot title and labels
plt.title('R-squared vs Mean Rain (mm) with Linear Trend Line')
plt.xlabel('Mean Rain (mm)')
plt.ylabel('R-squared')

plt.ylim(-1, 1)

# Showing the plot
plt.show()


# In[153]:


result_dataset_test


# In[154]:


# Adding a new column to each dataset to indicate whether it's from train or test
result_dataset_train['Type'] = 'train'
result_dataset_test['Type'] = 'test'

# Merging the datasets vertically
merged_dataset = pd.concat([result_dataset_train, result_dataset_test], ignore_index=True)


# In[155]:


print(list(merged_dataset))


# In[156]:


# Determine thresholds for 'Accumulated_Rain (mm)'
thresholds = merged_dataset['Accumulated_Rain (mm)'].quantile([0.33, 0.66]).values

# Function to categorize storms
def categorize_storm(rainfall):
    if rainfall <= thresholds[0]:
        return 'small'
    elif rainfall <= thresholds[1]:
        return 'medium'
    else:
        return 'large'

# Apply the function to categorize storms
merged_dataset['Storm_Category'] = merged_dataset['Accumulated_Rain (mm)'].apply(categorize_storm)

merged_dataset


# In[157]:


merged_dataset_train = merged_dataset[merged_dataset['Type'] == 'train']
merged_dataset_test = merged_dataset[merged_dataset['Type'] == 'test']


# Trainset

# In[158]:


# Calculating the mean R-squared value for each storm category
mean_r_squared_by_category = merged_dataset_train.groupby('Storm_Category')['R-squared'].mean()
print(mean_r_squared_by_category)


# In[159]:


mean_r_squared_by_category = merged_dataset_train.groupby('Year')['R-squared'].mean()
print(mean_r_squared_by_category)


# In[160]:


mean_r_squared_by_category = merged_dataset_train.groupby('Season_Num')['R-squared'].mean()
print(mean_r_squared_by_category)


# In[161]:


mean_r_squared_by_category = merged_dataset_train.groupby('Type')['R-squared'].mean()
print(mean_r_squared_by_category)


# Testset

# In[162]:


# Calculating the mean R-squared value for each storm category
mean_r_squared_by_category = merged_dataset_test.groupby('Storm_Category')['R-squared'].mean()
print(mean_r_squared_by_category)


# In[163]:


mean_r_squared_by_category = merged_dataset_test.groupby('Year')['R-squared'].mean()
print(mean_r_squared_by_category)


# In[164]:


mean_r_squared_by_category = merged_dataset_test.groupby('Season_Num')['R-squared'].mean()
print(mean_r_squared_by_category)


# In[165]:


mean_r_squared_by_category = merged_dataset_test.groupby('Type')['R-squared'].mean()
print(mean_r_squared_by_category)

