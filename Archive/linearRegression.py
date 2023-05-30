import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the data from Excel
df = pd.read_excel('badmintondata.xlsx')

# Identify consecutive rows of all zeros
mask = (df == 0).all(axis=1)
groups = mask.cumsum()

# Filter out consecutive rows of all zeros
filtered_df = df[~mask]

# Group the data before each group of zeros
grouped_df = filtered_df.groupby(groups)

# Split the groups into testing set and sample set
train_groups, test_groups = train_test_split(list(grouped_df.groups), test_size=0.2, random_state=42)

# Create the training set
train_df = pd.concat([grouped_df.get_group(group) for group in train_groups])

# Create the testing set
test_df = pd.concat([grouped_df.get_group(group) for group in test_groups])

# Function to add a Time field to a group
def add_time_field(group):
    group['Time'] = np.arange(len(group)) * 10  # Multiply by 10 to get time in ms
    return group

# Apply this function to each group in the train and test data
train_df = train_df.groupby(groups).apply(add_time_field)
test_df = test_df.groupby(groups).apply(add_time_field)

# Train a linear regression model for X, Y, and Z positions
regressor_X = LinearRegression()
regressor_Y = LinearRegression()
regressor_Z = LinearRegression()

regressor_X.fit(train_df[['Time']], train_df['SHUTTLECOCK POSITIION IN AIR(X) metres'])
regressor_Y.fit(train_df[['Time']], train_df['SHUTTLECOCK POSITIION IN AIR(Y) metres'])
regressor_Z.fit(train_df[['Time']], train_df['SHUTTLECOCK POSITIION IN AIR(Z) metres'])

# Predict the positions using the trained models
test_df['Predicted_X'] = regressor_X.predict(test_df[['Time']])
test_df['Predicted_Y'] = regressor_Y.predict(test_df[['Time']])
test_df['Predicted_Z'] = regressor_Z.predict(test_df[['Time']])

# Now let's plot X, Y, and Z as functions of time
fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(3, 1, 1)
ax1.scatter(train_df['Time'], train_df['SHUTTLECOCK POSITIION IN AIR(X) metres'], s=5, label='Train Data')
ax1.scatter(test_df['Time'], test_df['SHUTTLECOCK POSITIION IN AIR(X) metres'], s=5, label='Test Data')
ax1.plot(test_df['Time'], test_df['Predicted_X'], color='red', linewidth=2, label='Predicted X')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('X Position (metres)')
ax1.set_title('Shuttlecock X Position vs Time')
ax1.legend()

ax2 = fig.add_subplot(3, 1, 2)
ax2.scatter(train_df['Time'], train_df['SHUTTLECOCK POSITIION IN AIR(Y) metres'], s=5, label='Train Data')
ax2.scatter(test_df['Time'], test_df['SHUTTLECOCK POSITIION IN AIR(Y) metres'], s=5, label='Test Data')
ax2.plot(test_df['Time'], test_df['Predicted_Y'], color='red', linewidth=2, label='Predicted Y')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Y Position (metres)')
ax2.set_title('Shuttlecock Y Position vs Time')
ax2.legend()

ax3 = fig.add_subplot(3, 1, 3)
ax3.scatter(train_df['Time'], train_df['SHUTTLECOCK POSITIION IN AIR(Z) metres'], s=5, label='Train Data')
ax3.scatter(test_df['Time'], test_df['SHUTTLECOCK POSITIION IN AIR(Z) metres'], s=5, label='Test Data')
ax3.plot(test_df['Time'], test_df['Predicted_Z'], color='red', linewidth=2, label='Predicted Z')
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Z Position (metres)')
ax3.set_title('Shuttlecock Z Position vs Time')
ax3.legend()

plt.tight_layout()
plt.show()
