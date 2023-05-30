import pandas as pd
from sklearn.model_selection import train_test_split
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

# Print the number of groups in each set
print("Number of groups in the training set:", len(train_groups))
print("Number of groups in the testing set:", len(test_groups))

# # Extract the required columns from the training set
# x_train = train_df['SHUTTLECOCK POSITIION IN AIR(X) metres']
# y_train = train_df['SHUTTLECOCK POSITIION IN AIR(Y) metres']
# z_train = train_df['SHUTTLECOCK POSITIION IN AIR(Z) metres']

# # Create a 3D plot for the training set
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_train, y_train, z_train)

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Shuttlecock Position in Air (Training Set)')

# # Show the plots
# plt.show()

# Function to add a Time field to a group
def add_time_field(group):
    group['Time'] = np.arange(len(group)) * 10  # Multiply by 10 to get time in ms
    return group

# Apply this function to each group in the train and test data
train_df = train_df.groupby(groups).apply(add_time_field)
test_df = test_df.groupby(groups).apply(add_time_field)

# Now let's plot X, Y, and Z as functions of time
plt.figure(figsize=(15,10))

plt.subplot(3, 1, 1)
plt.scatter(train_df['Time'], train_df['SHUTTLECOCK POSITIION IN AIR(X) metres'], s=5)
plt.xlabel('Time (ms)')
plt.ylabel('X Position (metres)')
plt.title('Shuttlecock X Position vs Time')

plt.subplot(3, 1, 2)
plt.scatter(train_df['Time'], train_df['SHUTTLECOCK POSITIION IN AIR(Y) metres'], s=5)
plt.xlabel('Time (ms)')
plt.ylabel('Y Position (metres)')
plt.title('Shuttlecock Y Position vs Time')

plt.subplot(3, 1, 3)
plt.scatter(train_df['Time'], train_df['SHUTTLECOCK POSITIION IN AIR(Z) metres'], s=5)
plt.xlabel('Time (ms)')
plt.ylabel('Z Position (metres)')
plt.title('Shuttlecock Z Position vs Time')

plt.tight_layout()
plt.show()
