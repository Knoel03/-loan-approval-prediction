import pandas as pd

file_path = "data/raw/default of credit card clients.xls"

df = pd.read_excel(file_path, header=1)

print("DATASET OVERVIEW")
print("----------------")
print("Number of records:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nFIRST 5 RECORDS")
print(df.head())

print("\nDEFAULT DISTRIBUTION")
print(df["default payment next month"].value_counts())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

print("\nDATA TYPES")
print(df.dtypes)

categorical_columns = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

print("\nUNIQUE VALUES IN CATEGORICAL COLUMNS")

for column in categorical_columns:
    print(f"{column}: {sorted(df[column].unique())}")

print("\nCATEGORY FREQUENCIES")

for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].value_counts().sort_index())

numerical_columns = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

print("\nNUMERICAL DATA SUMMARY")
print(df[numerical_columns].describe().T.to_string())

billing_columns = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6"
]

print("\nNEGATIVE BILLING AMOUNTS")

for column in billing_columns:
    negative_count = (df[column] < 0).sum()
    print(f"{column}: {negative_count}")

print("\nSAMPLE RECORDS WITH NEGATIVE BILLING AMOUNTS")

negative_records = df[df[billing_columns].lt(0).any(axis=1)]

print(negative_records[billing_columns].head(10).to_string())

print("\nDEFAULT RATE BY REPAYMENT STATUS")

for column in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
    print(f"\n{column}")
    print(df.groupby(column)["default payment next month"].mean().round(3))

print("\nDEFAULT RATE BY CREDIT LIMIT")

df["credit_limit_group"] = pd.qcut(
    df["LIMIT_BAL"],
    q=5,
    duplicates="drop"
)

print(
    df.groupby("credit_limit_group", observed=True)
    ["default payment next month"]
    .agg(["count", "mean"])
    .round(3)
)
print("\nAVERAGE PAYMENT AMOUNTS BY DEFAULT STATUS")

payment_columns = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

print(
    df.groupby("default payment next month")[payment_columns]
    .mean()
    .round(2)
)