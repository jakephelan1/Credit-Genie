import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scikeras.wrappers import KerasClassifier
from keras.utils import to_categorical 
from keras.models import Sequential 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Input
from keras.regularizers import L1L2 
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta 
import keras_tuner as kt
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from fancyimpute import IterativeImputer
import joblib
from joblib import Parallel, parallel_backend
import logging
import optuna

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)

class OneHotTransformer:
    def fit(self, X, y=None):
        self.num_classes_ = 3  
        return self

    def transform(self, y):
        return to_categorical(y, num_classes=3)

    def inverse_transform(self, y, return_proba=False):
        if return_proba:
            return y
        return np.argmax(y, axis=1)

class CustomKerasClassifier(KerasClassifier):
    @property
    def target_encoder(self):
        return OneHotTransformer()
    
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        if not hasattr(self, "classes_") or self.classes_ == None:
            self.classes_ = np.arange(3)

    def predict_proba(self, X, **kwargs):
        if not hasattr(self, "model_") or self.model_ is None:
            raise NotFittedError("This KerasClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        
        proba = self.model_.predict(X, **kwargs)
        if proba is None:
            raise ValueError("Keras model did not return probabilities.")
        
        if proba.shape[1] != 3:
            raise ValueError(f"Expected model output shape (num_classes) to be 3, but got {proba.shape[1]}")
        return proba

class MyHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(self.input_shape,)))
        model.add(Flatten())

        activation = hp.Choice('activation', values=['relu', 'tanh', 'elu', 'selu'])
        dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[0.1, 1e-2, 1e-3, 1e-4])
        l2 = hp.Float('l2', min_value=0.0, max_value=0.01, step=0.001)
        l1 = hp.Float('l1', min_value=0.0, max_value=0.01, step=0.001)
        optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'])
        num_layers = hp.Int('num_layers', 1, 4)
        
        for i in range(num_layers):
            units = hp.Int(f'layer_{i}_units', min_value=16, max_value=256, step=8)
            model.add(Dense(units=units, activation=activation, kernel_regularizer=L1L2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        
        model.add(Dense(self.num_classes, activation='softmax'))
        
        if optimizer_choice == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        elif optimizer_choice == 'adagrad':
            optimizer = Adagrad(learning_rate=learning_rate)
        elif optimizer_choice == 'adadelta':
            optimizer = Adadelta(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [32, 64, 128, 256]),
            **kwargs
        )

def train_keras_nn_with_tuning(x_train, y_train, x_val, y_val):
    input_shape = x_train.shape[1]
    tuner = kt.Hyperband(
        hypermodel=MyHyperModel(input_shape=input_shape),
        objective="val_accuracy",
        max_epochs=150,
        factor=2,
        directory='dir',
        project_name='nn_training'
    )

    early_stop = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

    y_train = to_categorical(y_train, num_classes=3)
    y_val = to_categorical(y_val, num_classes=3)

    tuner.search(
        x_train, y_train,
        epochs=150,
        validation_data=(x_val, y_val),
        callbacks=[early_stop, reduce_lr]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)

    batch_size = best_hps.values.get('batch_size')
    logging.info(f"Fitting Keras model with optimal batch size: {batch_size}")

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=300,
        verbose=True,
        callbacks=[early_stop, reduce_lr]
    )
    
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    logging.info(f"Keras NN validation accuracy: {val_accuracy}")

    return model


def objective_rf(trial, x_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    n_estimators = trial.suggest_int('n_estimators', 100, 300)
    max_depth = trial.suggest_int('max_depth', 5, 30)  
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)  
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)  
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=100
    )
    
    scores = []
    for train_index, val_index in skf.split(x_train, y_train):
        x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        rf.fit(x_fold_train, y_fold_train)
        scores.append(rf.score(x_fold_val, y_fold_val))
    
    rf.fit(x_train, y_train)
    return np.mean(scores)

def train_rf_with_optuna(x_train, y_train, x_val, y_val):
    study = optuna.create_study(direction='maximize')
    with parallel_backend('loky', n_jobs=-1):
        study.optimize(lambda trial: objective_rf(trial, x_train, y_train), n_trials=50)
    
    best_params = study.best_params
    best_rf_model = RandomForestClassifier(**best_params, random_state=100)
    best_rf_model.fit(x_train, y_train)
    
    val_accuracy = best_rf_model.score(x_val, y_val)
    logging.info(f"Random Forest validation accuracy: {val_accuracy}")
    
    return best_rf_model


def objective_gb(trial, x_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    n_estimators = trial.suggest_int('n_estimators', 100, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)  
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)  
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)  
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    subsample = trial.suggest_float('subsample', 0.8, 1.0)
    
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        subsample=subsample,
        random_state=100
    )
    
    scores = []
    for train_index, val_index in skf.split(x_train, y_train):
        x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        gb.fit(x_fold_train, y_fold_train)
        scores.append(gb.score(x_fold_val, y_fold_val))
    
    gb.fit(x_train, y_train)
    return np.mean(scores)

def train_gb_with_optuna(x_train, y_train, x_val, y_val):
    study = optuna.create_study(direction='maximize')
    with parallel_backend('loky', n_jobs=-1):
        study.optimize(lambda trial: objective_gb(trial, x_train, y_train), n_trials=50)
    
    best_params = study.best_params
    best_gb_model = GradientBoostingClassifier(**best_params, random_state=100)
    best_gb_model.fit(x_train, y_train)
    
    val_accuracy = best_gb_model.score(x_val, y_val)
    logging.info(f"Gradient Boosting validation accuracy: {val_accuracy}")
    
    return best_gb_model


def objective_svm(trial, x_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    C = trial.suggest_float('C', 1e-4, 1e2, log=True)
    gamma = trial.suggest_float('gamma', 0.001, 1.0)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'sigmoid'])
    degree = trial.suggest_int('degree', 2, 4)
    coef0 = trial.suggest_float('coef0', 0.0, 1.0)
    
    svm = SVC(C=C, gamma=gamma, kernel=kernel, degree=degree, coef0=coef0, random_state=100, probability=True)

    scores = []
    for train_index, val_index in skf.split(x_train, y_train):
        x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        svm.fit(x_fold_train, y_fold_train)
        scores.append(svm.score(x_fold_val, y_fold_val))
    
    svm.fit(x_train, y_train)
    return np.mean(scores)

def train_svm_with_optuna(x_train, y_train, x_val, y_val):
    study = optuna.create_study(direction='maximize')
    with parallel_backend('loky', n_jobs=-1):
        study.optimize(lambda trial: objective_svm(trial, x_train, y_train), n_trials=30, timeout=7200)
    
    best_params = study.best_params
    best_svm_model = SVC(**best_params, random_state=100, probability=True)
    best_svm_model.fit(x_train, y_train)
    
    val_accuracy = best_svm_model.score(x_val, y_val)
    logging.info(f"SVM validation accuracy: {val_accuracy}")
    
    return best_svm_model

def objective_ada(trial, x_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
    max_depth = trial.suggest_int('max_depth', 1, 10) 
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)  
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)  
    
    base_estimator = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=100
    )
    
    ada = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm='SAMME',
        random_state=100
    )
    
    scores = []
    for train_index, val_index in skf.split(x_train, y_train):
        x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        ada.fit(x_fold_train, y_fold_train)
        scores.append(ada.score(x_fold_val, y_fold_val))
    
    ada.fit(x_train, y_train)
    return np.mean(scores)

def train_ada_with_optuna(x_train, y_train, x_val, y_val):
    study = optuna.create_study(direction='maximize')
    with parallel_backend('loky', n_jobs=-1):
        study.optimize(lambda trial: objective_ada(trial, x_train, y_train), n_trials=50)
    
    best_params = study.best_params
    max_depth = best_params.pop('max_depth')
    min_samples_split = best_params.pop('min_samples_split')
    min_samples_leaf = best_params.pop('min_samples_leaf')
    base_estimator = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=100
    )
    best_ada_model = AdaBoostClassifier(estimator=base_estimator, **best_params, algorithm='SAMME', random_state=100)
    best_ada_model.fit(x_train, y_train)
    
    val_accuracy = best_ada_model.score(x_val, y_val)
    logging.info(f"AdaBoost validation accuracy: {val_accuracy}")
    
    return best_ada_model


def objective_voting(trial, estimators, x_train, y_train):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=100)
    voting = trial.suggest_categorical('voting', ['hard', 'soft'])
    weights = [trial.suggest_int(f'weights_{i}', 1, 3) for i in range(len(estimators))]
    
    voting_clf = VotingClassifier(estimators=estimators, voting=voting, weights=weights)
    
    scores = []
    for train_index, val_index in skf.split(x_train, y_train):
        x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        voting_clf.fit(x_fold_train, y_fold_train)
        scores.append(voting_clf.score(x_fold_val, y_fold_val))
    
    voting_clf.fit(x_train, y_train)
    return np.mean(scores)

def train_voting_with_optuna(estimators, x_train, y_train, x_val, y_val):
    study = optuna.create_study(direction='maximize')
    with parallel_backend('loky', n_jobs=-1):
        study.optimize(lambda trial: objective_voting(trial, estimators, x_train, y_train), n_trials=40)
    
    best_params = study.best_params
    weights = [best_params[f'weights_{i}'] for i in range(len(estimators))]
    best_voting_model = VotingClassifier(estimators=estimators, voting=best_params['voting'], weights=weights)
    
    best_voting_model.fit(x_train, y_train)
    
    val_accuracy = best_voting_model.score(x_val, y_val)
    logging.info(f"Voting validation accuracy: {val_accuracy}")
    
    return best_voting_model

def objective_stacking(trial, estimators, x_train, y_train):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)
    stack_method = trial.suggest_categorical('stack_method', ['auto', 'predict_proba'])
    final_estimator_C = trial.suggest_loguniform('final_estimator__C', 1e-3, 1e3)
    final_estimator_penalty = trial.suggest_categorical('final_estimator__penalty', ['l2'])
    
    final_estimator = LogisticRegression(C=final_estimator_C, penalty=final_estimator_penalty, random_state=100, max_iter=1000)
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method=stack_method
    )
    
    scores = []
    for train_index, val_index in skf.split(x_train, y_train):
        x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        stacking_clf.fit(x_fold_train, y_fold_train)
        scores.append(stacking_clf.score(x_fold_val, y_fold_val))
    
    stacking_clf.fit(x_train, y_train)
    return np.mean(scores)

def train_stacking_with_optuna(estimators, x_train, y_train, x_val, y_val):
    study = optuna.create_study(direction='maximize')
    with parallel_backend('loky', n_jobs=-1):
        study.optimize(lambda trial: objective_stacking(trial, estimators, x_train, y_train), n_trials=30)
    
    best_params = study.best_params
    final_estimator = LogisticRegression(C=best_params['final_estimator__C'], penalty=best_params['final_estimator__penalty'], random_state=100, max_iter=1000)
    
    best_stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method=best_params['stack_method']
    )
    
    best_stacking_model.fit(x_train, y_train)
    
    val_accuracy = best_stacking_model.score(x_val, y_val)
    logging.info(f"Stacking validation accuracy: {val_accuracy}")
    
    return best_stacking_model

def preprocess_data(df):
    df = df.astype(float)

    df['Debt_to_Income_Ratio'] = df['Outstanding_Debt'] / df['Annual_Income']
    df['Credit_Utilization_Ratio'] = df['Outstanding_Debt'] / df['Changed_Credit_Limit']
    df['Monthly_Debt_Payment_Ratio'] = df['Total_EMI_per_month'] / df['Monthly_Inhand_Salary']
    df['Credit_Mix_Diversity'] = df['Num_Credit_Card'] + df['Num_Bank_Accounts'] + df['Num_of_Loan']
    df['Income_Stability'] = df['Annual_Income'] / df['Monthly_Inhand_Salary']
    df['Credit_Card_to_Bank_Account_Ratio'] = df['Num_Credit_Card'] / df['Num_Bank_Accounts']
    df['Credit_Age_Income_Ratio'] = df['Credit_History_Age'] / df['Annual_Income']
    df['Debt_to_Credit_Ratio'] = df['Outstanding_Debt'] / df['Changed_Credit_Limit']
    df['Credit_Card_Utilization'] = df['Num_Credit_Card'] / df['Annual_Income']
    df['Debt_Income_Credit_Ratio'] = (df['Outstanding_Debt'] + df['Total_EMI_per_month']) / df['Annual_Income']
    df['Credit_Age_Limit_Ratio'] = df['Credit_History_Age'] / df['Changed_Credit_Limit']
    df['Log_Outstanding_Debt'] = np.log1p(df['Outstanding_Debt'])
    df['Log_Monthly_Balance'] = np.log1p(df['Monthly_Balance'])
    df['Credit_Inquiries_to_Loan_Ratio'] = df['Num_Credit_Inquiries'] / df['Num_of_Loan']
    df['Credit_Inquiries_to_Credit_Card_Ratio'] = df['Num_Credit_Inquiries'] / df['Num_Credit_Card']
    df['Credit_Inquiries_to_Bank_Account_Ratio'] = df['Num_Credit_Inquiries'] / df['Num_Bank_Accounts']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['Credit_Score'] = df['Credit_Score'].round().astype(int)

    credit_score = df.pop('Credit_Score')
    df['Credit_Score'] = credit_score

    return df

def find_top_features(x_train, x_train_smote, y_train_smote):
    rf = RandomForestClassifier(n_estimators=100, random_state=100)
    rf.fit(x_train_smote, y_train_smote)
    importances = rf.feature_importances_
    feature_names = x_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    logging.info(f"Feature Importances:\n{feature_importance_df}")

    plt.figure(figsize=(14, 8))
    plt.title("Feature Importances")
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.gca().invert_yaxis()
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout() 
    plt.savefig("plots/feature_importances.png")
    plt.close()

    feature_importance_df['Cumulative_Importance'] = feature_importance_df['Importance'].cumsum()
    logging.info(f"Cumulative Feature Importances:\n{feature_importance_df}")

    plt.figure(figsize=(14, 8))
    plt.title("Cumulative Feature Importances")
    plt.plot(feature_importance_df['Feature'], feature_importance_df['Cumulative_Importance'], marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout(pad=5) 
    plt.savefig("plots/cumulative_importance.png")
    plt.close()

    threshold = 0.90
    top_features = feature_importance_df[feature_importance_df['Cumulative_Importance'] <= threshold]['Feature'].values
    dropped_features = feature_importance_df[feature_importance_df['Cumulative_Importance'] > threshold]['Feature'].values
    logging.info(f"Features Dropped (threshold {threshold}): {dropped_features}")

    return top_features, feature_names

def get_params(model):
    model = joblib.load(f'trained_models/{model}_model.pkl')
    print(model.get_params())

def main():
    df = pd.read_csv("CSV/adjusted_dataset.csv")
    
    df = preprocess_data(df)
    logging.info('Engineered domain-specific features')
    
    y = df['Credit_Score'].astype(int)
    x = df.drop(columns=['Credit_Score'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=100, stratify=y_train)

    imputer = IterativeImputer(random_state=100, max_iter=100, tol=5e-4)
    x_train = imputer.fit_transform(x_train)
    x_val = imputer.transform(x_val)
    x_test = imputer.transform(x_test)
    logging.info('Applied Iterative Imputer for NaN values')

    x_train = pd.DataFrame(x_train, columns=x.columns)
    x_val = pd.DataFrame(x_val, columns=x.columns)
    x_test = pd.DataFrame(x_test, columns=x.columns)

    categorical_features = ['Credit_Mix', 'Payment_Behaviour', 'Occupation']
    integer_features = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Age', 'Num_Credit_Inquiries']

    for feature in categorical_features + integer_features:
        x_train[feature] = x_train[feature].round().astype(int)
        x_val[feature] = x_val[feature].round().astype(int)
        x_test[feature] = x_test[feature].round().astype(int)
    
    logging.info('Converted integer/categorical features to integers')

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    logging.info('Applied scaling using a standard scaler')

    smote = SMOTE(random_state=100)
    x_train_smote, y_train_smote = smote.fit_resample(x_train_scaled, y_train)

    logging.info('Applied SMOTE to handle class imbalance')

    top_features, feature_names = find_top_features(x_train, x_train_smote, y_train_smote)

    x_train_smote = pd.DataFrame(x_train_smote, columns=feature_names)[top_features].values
    x_val_scaled = pd.DataFrame(x_val_scaled, columns=feature_names)[top_features].values
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=feature_names)[top_features].values

    unwrapped_keras_model = train_keras_nn_with_tuning(x_train_smote, y_train_smote, x_val_scaled, y_val)
    clf = CustomKerasClassifier(model=unwrapped_keras_model, verbose=1)
    keras_model = Pipeline([
        ('clf', clf)
    ])
    keras_model.fit(x_train_smote, y_train_smote)
    rf_model = train_rf_with_optuna(x_train_smote, y_train_smote, x_val_scaled, y_val)
    gb_model = train_gb_with_optuna(x_train_smote, y_train_smote, x_val_scaled, y_val)
    ada_model = train_ada_with_optuna(x_train_smote, y_train_smote, x_val_scaled, y_val)
    svm_model = train_svm_with_optuna(x_train_smote, y_train_smote, x_val_scaled, y_val)

    models = [
        ('keras', keras_model),
        ('rf', rf_model),
        ('gb', gb_model),
        ('svm', svm_model),
        ('ada', ada_model)
    ]        

    accuracies = {}
    for name, model in models:
        y_pred = model.predict(x_test_scaled)
        accuracy = model.score(x_test_scaled, y_test)
        accuracies[name] = accuracy
        logging.info(f"{name.upper()} Classification Report:")
        logging.info(classification_report(y_test, y_pred))
        joblib.dump(model, f"trained_models/{name}_model.pkl")

    best_model_name = max(accuracies, key=accuracies.get)
    logging.info(f"Best base model: {best_model_name}")

    voting_model = train_voting_with_optuna(models, x_train_smote, y_train_smote, x_val_scaled, y_val)
    stacking_model = train_stacking_with_optuna(models, x_train_smote, y_train_smote, x_val_scaled, y_val)      

    ensemble_models = [
        ('voting', voting_model),
        ('stacking', stacking_model)
    ]

    ensemble_accuracies = {}
    for name, model in ensemble_models:
        y_pred = model.predict(x_test_scaled)
        accuracy = model.score(x_test_scaled, y_test)
        ensemble_accuracies[name] = accuracy
        logging.info(f"{name.upper()} Classification Report:")
        logging.info(classification_report(y_test, y_pred))
        joblib.dump(model, f"trained_models/{name}_model.pkl")

    logging.info(f"Ensemble Accuracies: {ensemble_accuracies}")

    joblib.dump(scaler, "trained_models/scaler.pkl")
    joblib.dump(imputer, "trained_models/imputer.pkl")

if __name__ == "__main__":
    main()