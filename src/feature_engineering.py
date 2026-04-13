# CELL 9 — FRAUD BY TRANSACTION HOUR
# ============================================================
# We extract the hour of day from the transaction timestamp and check
# if fraud tends to happen at certain times (e.g., late at night).
# This gives us the `trans_hour` feature we'll use later in the model.

df['trans_date_time'] = pd.to_datetime(df['trans_date_time'])
df['trans_hour'] = df['trans_date_time'].dt.hour

hour_fraud = df.groupby('trans_hour')['is_fraud'].mean()


# ============================================================
# CELL 10 — FRAUD BY AGE GROUP
# ============================================================
# We temporarily calculate age here just for the EDA visualization.
# (The permanent age feature will be engineered properly in the next section.)
# Grouping by age bins shows whether certain age groups are targeted more.

df['dob'] = pd.to_datetime(df['dob'])
df['age_temp'] = datetime.now().year - df['dob'].dt.year
df['age_group'] = pd.cut(df['age_temp'], bins=[0, 25, 35, 45, 55, 65, 100],
                          labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])

age_fraud = df.groupby('age_group')['is_fraud'].mean()


# ============================================================
# CELL 11 — FEATURE: AGE FROM DATE OF BIRTH
# ============================================================
# FIX: The original code used hardcoded 2020. We now use datetime.now().year
# so the age is always calculated relative to the current year.
# After extracting age, we drop the raw dob column since it's no longer needed.

df['age'] = datetime.now().year - df['dob'].dt.year
df = df.drop(columns=['dob'])

print("Age feature created. Sample values:")
print(df['age'].describe())


# ============================================================
# CELL 12 — FEATURE: DISTANCE BETWEEN CARDHOLDER AND MERCHANT
# ============================================================
# We calculate the Euclidean distance between the cardholder's location
# (lat/long) and the merchant's location (merch_lat/merch_long).
# A large distance can be a signal of fraud — the card is being used
# far away from where the cardholder is located.

df['distance'] = np.sqrt(
    (df['lat'] - df['merch_lat'])**2 +
    (df['long'] - df['merch_long'])**2
)

print("Distance feature created. Sample:")
print(df['distance'].describe())

# ============================================================
# CELL 13 — FEATURE: IS_NIGHT FLAG
# ============================================================
# Based on our EDA, fraud happens more often between 10 PM and 4 AM.
# We create a binary flag: 1 if the transaction was during those hours, 0 otherwise.
# Binary flags like this are simple but very effective for tree-based models.

df['is_night'] = df['trans_hour'].apply(lambda h: 1 if (h >= 22 or h <= 4) else 0)

print("is_night feature created.")
print(df['is_night'].value_counts())
print(f"\nFraud rate at night: {df[df['is_night']==1]['is_fraud'].mean()*100:.2f}%")
print(f"Fraud rate during day: {df[df['is_night']==0]['is_fraud'].mean()*100:.2f}%")


# ============================================================
# CELL 14 — FEATURE: LOG TRANSFORM CITY POPULATION 
# ============================================================
# city_pop (city population) is likely heavily skewed — a few very large cities
# vs many small ones. Log transformation compresses this scale so the model
# isn't dominated by extreme values. We keep the original column for reference.

df['city_pop_log'] = np.log1p(df['city_pop'])  # log1p = log(1 + x), safe for 0 values

print("city_pop_log feature created.")
print(df[['city_pop', 'city_pop_log']].describe())