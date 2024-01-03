import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load Data
df = pd.read_csv("bread basket.csv")

# Data Preparation
df['Item'] = df['Item'].str.strip()
df['Item'] = df['Item'].str.lower()

df['date_time'] = pd.to_datetime(df['date_time'])

df['month']=df['date_time'].dt.month
df['month']=df['month'].replace((1,2,3,4,5,6,7,8,9,10,11,12), ('january','february','march','april','may','june','july','august','september','october','november','december'))

df['weekday'] = df['date_time'].dt.weekday
df['weekday'] = df['weekday'].replace((0,1,2,3,4,5,6), ('monday','tuesday','wednesday','thursday','friday','saturday','sunday'))
# Data Preparation
unique_items = df['Item'].unique()
period_day_options = df['period_day'].unique()
weekday_options = df['weekday'].unique()  # Add weekday options
month_options = df['month'].unique()  # Add month options

# Streamlit App
st.title('Sistem Rekomendasi Produk')

# Sidebar for filters
st.sidebar.header('Filters')

# User Input
selected_item = st.sidebar.selectbox('Select an item:', unique_items)
selected_period_day = st.sidebar.selectbox('Select a period day:', period_day_options)
selected_weekday = st.sidebar.selectbox('Select a weekday:', weekday_options)  # Add weekday input
selected_month = st.sidebar.selectbox('Select a month:', month_options)  # Add month input

# Filter data based on user input
filtered_data = df[(df['period_day'] == selected_period_day) & (df['weekday'] == selected_weekday) & (df['month'] == selected_month)]


transactions = filtered_data.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Number of Items')

# Table Transformation
table = transactions.pivot_table(index='Transaction', columns='Item', values='Number of Items', aggfunc='sum').fillna(0)

def hot_encode(x):
    if x == 0:
        return False
    if x > 0:
        return True

final_table = table.applymap(hot_encode)

# Import Machine Learning
frequence = apriori(final_table, min_support=0.015, use_colnames=True)
product_association = association_rules(frequence, metric='confidence', min_threshold=1).sort_values('confidence', ascending=False).reset_index(drop=True)

# Convert frozenset to list
product_association['antecedents'] = product_association['antecedents'].apply(lambda x: list(x))
product_association['consequents'] = product_association['consequents'].apply(lambda x: list(x))

# Filter rules based on selected item
selected_item_rules = product_association[
    (product_association['antecedents'].apply(lambda x: selected_item in x)) |
    (product_association['consequents'].apply(lambda x: selected_item in x))]

# Display the rules only if there are any
if not selected_item_rules.empty:
    selected_item_rules=selected_item_rules.iloc[0]
    st.success(f'Hasil Rekomendasi {selected_item}')
    st.write(selected_item_rules[['antecedents', 'consequents']])#tampilkan hanya top 1 dari selected item
else:
    st.subheader(f'No rules found with the selected criteria for {selected_item}.')