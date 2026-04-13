# ============================================================
# CELL 5 — DROP UNNECESSARY COLUMNS & RENAME
# ============================================================
# 'Unnamed: 0' is just a leftover index column from the CSV — safe to drop.
# 'merch_zipcode' has too many missing values and isn't useful for our model.
# We also rename columns to be more readable and consistent.

df.drop(['Unnamed: 0', 'merch_zipcode'], axis=1, inplace=True, errors='ignore')

df.rename(columns={
    'trans_date_trans_time': 'trans_date_time',
    'cc_num': 'credit_card_number',
    'amt': 'amount'
}, inplace=True)

print("Columns cleaned. Remaining columns:")
print(df.columns.tolist())


# ============================================================
# CELL 15 — DROP COLUMNS NOT USEFUL FOR MODELING
# ============================================================
# These columns are either identifiers (like card number, name, address)
# or redundant after feature engineering (like raw lat/long after distance is made).
# Keeping them would confuse the model or cause privacy issues in production.

cols_to_drop = [
    'trans_date_time',
    'credit_card_number',
    'merchant',
    'first',
    'last',
    'street',
    'city',
    'zip',
    'job',
    'trans_num',
    'unix_time',
    'city_pop',   # replaced by city_pop_log
    'lat',        # replaced by distance feature
    'long',
    'merch_lat',
    'merch_long'
]

df = df.drop(columns=cols_to_drop, errors='ignore')
print("Columns after dropping identifiers:")
print(df.columns.tolist())

# ============================================================
# CELL 16 — ENCODING CATEGORICAL COLUMNS
# ============================================================
# Machine learning models require numeric input — they can't process text.
# gender: simple binary map (M=1, F=0)
# category & state: one-hot encoding (get_dummies) creates a binary column
# for each unique value. drop_first=True avoids the "dummy variable trap".

df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df = pd.get_dummies(df, columns=['category'], drop_first=True)
df = pd.get_dummies(df, columns=['state'], drop_first=True)

print("Encoding complete. Final shape:", df.shape)


