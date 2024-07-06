from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
from filter_dataset import clean_data, filter_emi
from train_models import CustomKerasClassifier, OneHotTransformer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        age = request.form.get('age')
        job_title = request.form.get('job_title')
        annual_income = request.form.get('annual_income')
        num_accounts = request.form.get('num_bank_accounts')
        num_cc = request.form.get('num_credit_cards')
        num_loans = request.form.get('num_loans')
        total_outstanding_debt = request.form.get('total_outstanding_debt')
        equated_monthly_installment = request.form.get('emi')
        interest_rate_cc = request.form.get('interest_rate_cc')
        monthly_balance = request.form.get('monthly-loan-balance')
        monthly_inhand_salary = request.form.get('monthly_inhand_salary')
        number_delayed_payments = request.form.get('number_delayed_payments')
        changed_credit_limit = request.form.get('changed_credit_limit')
        credit_mix = request.form.get('credit_mix')
        credit_history_age = request.form.get('credit_history_age')
        payment_behavior = request.form.get('payment_behavior')
        num_credit_inquiries = request.form.get('num_credit_inquiries')

        user_info = [age, job_title, annual_income, monthly_inhand_salary, num_accounts, num_cc, interest_rate_cc, num_loans, number_delayed_payments, changed_credit_limit, num_credit_inquiries, credit_mix, total_outstanding_debt, credit_history_age, equated_monthly_installment, payment_behavior, monthly_balance]
        predictions = predict_credit_score(user_info)
        
        return render_template('dashboard.html', model_data=json.dumps(predictions))

def load_models():
    keras_model = joblib.load("trained_models/keras_model.pkl")
    rf = joblib.load("trained_models/rf_model.pkl")
    svm = joblib.load("trained_models/svm_model.pkl")
    gb = joblib.load("trained_models/gb_model.pkl")
    ada = joblib.load("trained_models/ada_model.pkl")
    stacking = joblib.load("trained_models/stacking_model.pkl")
    voting = joblib.load("trained_models/voting_model.pkl")
    scaler = joblib.load("trained_models/scaler.pkl")
    imputer = joblib.load("trained_models/imputer.pkl")

    return keras_model, rf, svm, gb, ada, stacking, voting, scaler, imputer

def preprocess_input_data(input_data, scaler, imputer):
    input_df = pd.DataFrame([input_data], columns=['Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt', 'Credit_History_Age', 'Total_EMI_per_month', 'Payment_Behaviour', 'Monthly_Balance'])

    input_df = clean_data(input_df)
    input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    input_df = filter_emi(input_df)
    input_df = input_df.astype(float)

    input_df['Debt_to_Income_Ratio'] = input_df['Outstanding_Debt'] / input_df['Annual_Income']
    input_df['Credit_Utilization_Ratio'] = input_df['Outstanding_Debt'] / input_df['Changed_Credit_Limit']
    input_df['Monthly_Debt_Payment_Ratio'] = input_df['Total_EMI_per_month'] / input_df['Monthly_Inhand_Salary']
    input_df['Credit_Mix_Diversity'] = input_df['Num_Credit_Card'] + input_df['Num_Bank_Accounts'] + input_df['Num_of_Loan']
    input_df['Income_Stability'] = input_df['Annual_Income'] / input_df['Monthly_Inhand_Salary']
    input_df['Credit_Card_to_Bank_Account_Ratio'] = input_df['Num_Credit_Card'] / input_df['Num_Bank_Accounts']
    input_df['Credit_Age_Income_Ratio'] = input_df['Credit_History_Age'] / input_df['Annual_Income']
    input_df['Debt_to_Credit_Ratio'] = input_df['Outstanding_Debt'] / input_df['Changed_Credit_Limit']
    input_df['Credit_Card_Utilization'] = input_df['Num_Credit_Card'] / input_df['Annual_Income']
    input_df['Debt_Income_Credit_Ratio'] = (input_df['Outstanding_Debt'] + input_df['Total_EMI_per_month']) / input_df['Annual_Income']
    input_df['Credit_Age_Limit_Ratio'] = input_df['Credit_History_Age'] / input_df['Changed_Credit_Limit']
    input_df['Log_Outstanding_Debt'] = np.log1p(input_df['Outstanding_Debt'])
    input_df['Log_Monthly_Balance'] = np.log1p(input_df['Monthly_Balance'])
    input_df['Credit_Inquiries_to_Loan_Ratio'] = input_df['Num_Credit_Inquiries'] / input_df['Num_of_Loan']
    input_df['Credit_Inquiries_to_Credit_Card_Ratio'] = input_df['Num_Credit_Inquiries'] / input_df['Num_Credit_Card']
    input_df['Credit_Inquiries_to_Bank_Account_Ratio'] = input_df['Num_Credit_Inquiries'] / input_df['Num_Bank_Accounts']

    input_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

    categorical_features = ['Credit_Mix', 'Payment_Behaviour', 'Occupation']
    integer_features = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Age', 'Num_Credit_Inquiries']

    for feature in categorical_features + integer_features:
        input_df_imputed[feature] = input_df_imputed[feature].round().astype(int)

    scaled_input_df_imputed = scaler.transform(input_df_imputed)

    df = pd.DataFrame(scaled_input_df_imputed, columns=input_df.columns)

    final_df = df.drop(columns=['Occupation', 'Annual_Income', 'Credit_Age_Income_Ratio', 'Total_EMI_per_month', 'Monthly_Inhand_Salary', 'Credit_Inquiries_to_Loan_Ratio', 'Payment_Behaviour'])
    
    return final_df.values

def predict_credit_score(input_data):
    keras_model, rf, svm, gb, ada, stacking, voting, scaler, imputer = load_models()
    input_data = preprocess_input_data(input_data, scaler, imputer)
    
    models = {
        'keras': keras_model,
        'rf': rf,
        'svm': svm,
        'gb': gb,
        'ada': ada,
        'stacking': stacking,
        'voting': voting
    }
    predictions = {}
    for model_name, model in models.items():
        if model_name != 'voting':
            proba = model.predict_proba(input_data)[0]
            predictions[model_name] = {
                'category': map_credit_score_category(np.argmax(proba)),
                'probabilities': proba.tolist()
            }
        else:
            vote = model.predict(input_data)[0]
            predictions[model_name] = {
                'category': map_credit_score_category(vote),
                'individual_predictions': {
                    m: map_credit_score_category(np.argmax(models[m].predict_proba(input_data)[0]))
                    for m in ['keras', 'rf', 'gb', 'svm', 'ada']
                }
            }
    
    return predictions

def map_credit_score_category(value):
    mapping = {0: "Poor", 1: "Standard", 2: "Good"}
    return mapping.get(value, "Unknown")

@app.route('/templates/info_page.html')
def goToPage2():
    if request.method == "GET":
        return render_template('info_page.html')
    else:
        pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 
