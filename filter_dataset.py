import pandas as pd
import numpy as np
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/filter.log"),
        logging.StreamHandler()
    ]
)

def main():
    try:
        train_df = pd.read_csv("CSV/dataset.csv")
        logging.info("Dataset loaded successfully")
        
        columns_to_remove = [
            "ID", "Month", "Name", "SSN",
            "Type_of_Loan", "Delay_from_due_date",
            "Credit_Utilization_Ratio", "Payment_of_Min_Amount", "Amount_invested_monthly"
        ]
        train_df.drop(columns=columns_to_remove, inplace=True)
        logging.info("Unnecessary columns dropped")

        train_df = clean_data(train_df)
        logging.info("Data cleaning completed")

        train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        train_df = filter_emi(train_df)
        logging.info("EMI filtering applied")

        train_df = remove_outliers(train_df)
        logging.info("Outliers removed")

        max_notna_idx = train_df.drop(columns='Customer_ID').notna().sum(axis=1).groupby(train_df['Customer_ID']).idxmax()
        train_df = train_df.loc[max_notna_idx]
        train_df.drop(columns="Customer_ID", inplace=True)
        logging.info("Duplicates dropped while keeping rows with most information")

        train_df.to_csv('CSV/adjusted_dataset.csv', index=False)
        logging.info("Cleaned dataset saved to 'adjusted_dataset.csv'")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def convert_credit_score(score):
    score_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
    return score_mapping.get(score, np.nan)

def convert_credit_history(history):
    if pd.isna(history) or history == 'NA':
        return 0
    history = re.sub(r'\D', '', history.split(' ')[0])
    try:
        return int(history)
    except ValueError:
        return np.nan

def clean_age(age):
    age = str(age)
    age = re.sub(r'\D', '', age)
    try:
        age = int(age)
        if age < 13 or age > 100:
            return np.nan
    except (ValueError, TypeError):
        return np.nan
    return age

def clean_occupation(occupation):
    return 'None' if isinstance(occupation, str) and occupation == '_______' else str(occupation)

def clean_balance(bal):
    bal = re.sub(r'[^\d.-]', '', str(bal).split('.')[0])
    try:
        return float(bal)
    except ValueError:
        return np.nan

def clean_bank_accounts(ba):
    ba = re.sub(r'\D', '', str(ba))
    try:
        ba = int(ba)
        if ba < 0:
            return np.nan
    except ValueError:
        return np.nan
    return ba 

def clean_interest_rate(ir):
    ir = re.sub(r'\D', '', str(ir))
    try:
        ir = int(ir)
        if ir > 36 or ir < 0:
            return np.nan
    except ValueError:
        return np.nan
    return ir

def clean_loans(lo):
    lo = re.sub(r'\D', '', str(lo))
    try:
        lo = int(lo)
        if lo > 20:
            return np.nan
    except ValueError:
        return np.nan
    return lo

def adjust_values(row):
    if row['Num_Credit_Card'] < 1 or row['Num_Credit_Card'] > 10:
        row['Num_Credit_Card'] = np.nan

    if row['Num_Bank_Accounts'] < 1 or row['Num_Bank_Accounts'] > 10:
        row['Num_Bank_Accounts'] = np.nan

    if row['Num_of_Delayed_Payment'] > 50 or row['Num_of_Delayed_Payment'] < 0:
        row['Num_of_Delayed_Payment'] = np.nan

    if pd.notna(row['Monthly_Inhand_Salary']) and row['Monthly_Inhand_Salary'] > 1.5 * (row['Annual_Income'] / 12):
        row['Monthly_Inhand_Salary'] = row['Annual_Income'] / 12

    return row

def filter_emi(df):
    temp_monthly_inhand_salary = df['Monthly_Inhand_Salary'].copy()
    temp_monthly_inhand_salary.fillna(df['Annual_Income'] / 12, inplace=True)
    
    df['Total_EMI_per_month'] = np.where(
        df['Total_EMI_per_month'] > 0.5 * temp_monthly_inhand_salary,
        np.nan,
        df['Total_EMI_per_month']
    )
    
    return df

def remove_outliers(df):
    def is_outlier(s):
        lower_limit = s.mean() - (s.std() * 3)
        upper_limit = s.mean() + (s.std() * 3)
        return ~s.between(lower_limit, upper_limit)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[is_outlier(df[col]), col] = np.nan
    
    return df

def clean_data(df):
    df['Age'] = df['Age'].apply(clean_age)
    try:
        df['Credit_Score'] = df['Credit_Score'].apply(convert_credit_score)
    except:
        pass
    df['Occupation'] = df['Occupation'].apply(clean_occupation)
    df['Annual_Income'] = df['Annual_Income'].apply(clean_balance)
    df['Num_Bank_Accounts'] = df['Num_Bank_Accounts'].apply(clean_bank_accounts)
    df['Num_Credit_Card'] = df['Num_Credit_Card'].apply(clean_bank_accounts)
    df['Interest_Rate'] = df['Interest_Rate'].apply(clean_interest_rate)
    df['Num_of_Loan'] = df['Num_of_Loan'].apply(clean_loans)
    df['Outstanding_Debt'] = df['Outstanding_Debt'].apply(clean_balance)
    df['Total_EMI_per_month'] = df['Total_EMI_per_month'].apply(clean_balance)
    df['Monthly_Balance'] = df['Monthly_Balance'].apply(clean_balance)
    df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].apply(clean_balance)
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].apply(clean_balance)
    df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].apply(clean_balance)
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(convert_credit_history)

    df = df.apply(adjust_values, axis=1)

    payment_behaviour_mapping = {
        'Low_spent_Small_value_payments': 1,
        'Low_spent_Medium_value_payments': 2,
        'Low_spent_Large_value_payments': 3,
        'High_spent_Small_value_payments': 4,
        'High_spent_Medium_value_payments': 5,
        'High_spent_Large_value_payments': 6
    }

    credit_mix_mapping = {
        'Bad': 1,
        'Standard': 2,
        'Good': 3
    }

    occupation_mapping = {
        'Salesperson': 1,
        'Clerk': 2,
        'Farmer': 3,
        'Mechanic': 4,
        'Electrician': 5,
        'Plumber': 6,
        'Police_Officer': 7,
        'Firefighter': 8,
        'Nurse': 9,
        'Teacher': 10,
        'Artist': 11,
        'Writer': 12,
        'Journalist': 13,
        'Chef': 14,
        'Accountant': 15,
        'Architect': 16,
        'Developer': 17,
        'Manager': 18,
        'Entrepreneur': 19,
        'Media_Manager': 20,
        'Lawyer': 21,
        'Doctor': 22,
        'Engineer': 23,
        'Scientist': 24,
        'Pilot': 25,
        'Musician': 26,
        'None': 27
    }

    df['Payment_Behaviour'] = df['Payment_Behaviour'].map(payment_behaviour_mapping).astype('category')
    df['Credit_Mix'] = df['Credit_Mix'].map(credit_mix_mapping).astype('category')
    df['Occupation'] = df['Occupation'].map(occupation_mapping).astype('category')

    return df

if __name__ == '__main__':
    main()
