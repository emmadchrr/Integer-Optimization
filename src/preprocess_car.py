#%%
from ucimlrepo import fetch_ucirepo 
import pandas as pd


# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 
  
# metadata 
print(car_evaluation.metadata) 
  
# variable information 
print(car_evaluation.variables) 
#%%
# Specify the file path
file_path = "../data/car.data"

# Read the CSV file into a pandas DataFrame
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(file_path, names=column_names, header=None)

# Display the first few rows of the DataFrame
print(df.head())
print(df['class'].unique())

#%%
# Map class values to integers
class_mapping = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}
df['class'] = df['class'].map(class_mapping)
#%%
# Display the first few rows of the DataFrame after mapping
print(df.head(100))
print(df['class'].unique())
# %%
doors_mapping = {'2': 2, '3': 3, '4': 4, '5more': 5}
df['doors'] = df['doors'].map(doors_mapping)
# %%
# Display the first few rows of the DataFrame after mapping
print(df.head(100))
print(df['doors'].unique())
# %%
buying_mapping = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
df['buying'] = df['buying'].map(buying_mapping)
# %%
print(df.head(100))
print(df['buying'].unique())
#%%
persons_mapping = {'2': 2, '4': 4, 'more': 5}
df['persons'] = df['persons'].map(persons_mapping)
#%%
print(df.head(100))
print(df['persons'].unique())
# %%
maint_mapping = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
df['maint'] = df['maint'].map(maint_mapping)
# %%
print(df.head(100))
print(df['maint'].unique())
# %%
lug_boot_mapping = {'small': 1, 'med': 2, 'big': 3}
df['lug_boot'] = df['lug_boot'].map(lug_boot_mapping)
# %%
print(df.head(100))
print(df['lug_boot'].unique())
# %%
safe_mapping = {'low': 1, 'med': 2, 'high': 3}
df['safety'] = df['safety'].map(safe_mapping)
# %%
print(df.head(100))
print(df['safety'].unique())
# %%
from sklearn.preprocessing import MinMaxScaler
# Set the desired number of data points per class
desired_num_data = 50

# Get the counts of each class
class_counts = df['class'].value_counts()

# Get the names of the 3 most present classes
top_classes = class_counts.nlargest(4).index.tolist()

# Initialize an empty DataFrame to store the sampled data
sampled_df = pd.DataFrame()

# Iterate over the top classes
for class_label in top_classes:
    # Get the data points for the current class
    class_data = df[df['class'] == class_label]

    # Sample the desired number of data points for the current class
    sampled_data = class_data.sample(n=min(desired_num_data, len(class_data)), random_state=42)

    # Append the sampled data to the sampled_df DataFrame
    sampled_df = pd.concat([sampled_df, sampled_data])

# Update the df DataFrame with the sampled data
df = sampled_df
# %%
# Define the file path for the .txt file
txt_file_path = "../data/car.txt"
numerical_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
# Open the .txt file in write mode
with open(txt_file_path, 'w') as file:
    # Write the X line
    x_line = "X=["
    x_line += ";\n ".join([" ".join(map(str, row)) for row in df[numerical_columns].values])
    x_line += "]\n"
    file.write(x_line)

    # Write the Y line
    y_line = "Y = Vector{Any}([" + ", ".join(map(str, df['class'].values)) + "])"
    file.write(y_line)
# %%
