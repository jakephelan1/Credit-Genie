Created by: Jake Phelan & Daniel Sachs

https://github.com/jakephelan1

https://github.com/d3buggerdan

# Credit Score Prediction Project
This project is designed to train various machine learning models on a credit score database containing different information about individuals and their respective credit scores. It includes a GUI with a Flask backend to use the trained models for predicting a credit score based on user inputs. Additionally, it outputs a pie chart indicating each model's confidence in their predictions.

### Data Preperation
We began with a messy dataset containing over 70,000 inputs. The initial data required extensive cleaning and management to make it suitable for training. This involved handling missing values, normalizing data, and transforming categorical variables. Through a rigorous data preprocessing phase, we created a new, clean dataset, which significantly improved the performance of our machine learning models.

### Features
Trains multiple machine learning models on credit score data
Predicts credit scores using a Flask-based web interface
Displays model confidence using pie charts
Implements feature engineering and data preprocessing
Utilizes SMOTE for handling class imbalance
Supports hyperparameter tuning using Optuna and Keras Tuner
Provides logging for monitoring training and data processing

### Tools Used
Python: Programming language for the backend and model training

Flask: Micro web framework for the web application

Pandas: Data manipulation and analysis

NumPy: Numerical computing

Scikit-Learn: Machine learning library

Imbalanced-learn: Handling imbalanced datasets

Keras and TensorFlow: Deep learning library

Optuna: Hyperparameter optimization framework

Joblib: Serialization of models and other objects

Matplotlib: Plotting library for visualizations

FancyImpute: Advanced imputation of missing values

### Setup and Installation
To set up and run this project locally, follow these steps. Note that steps 5 and 6 are optional and are included for model training reproducibility. If you only want to test the app, you can skip steps 5 and 6 as they are already done here:

1. **Clone the Repository**

```bash
git clone https://github.com/jakephelan1/credit-genie.git
cd credit-genie
```

2. **Create a Virtual Environment**

```bash
python3 -m venv myenv
source myenv/bin/activate
```

3. **Install Dependencies**
   
Ensure you have Git LFS installed and configured:

```bash
brew install git-lfs
git lfs install
```

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

4. **Download Large Files**
   
If you have pushed large files using Git LFS, download them:

```bash
git lfs pull
```

5. **Set Up the Database and Data Files**
   
Place your dataset in the CSV/ directory with the name dataset.csv.

6. **Run Data Preprocessing and Model Training**

```bash
python filter_dataset.py
python train_models.py
```

7. **Start the Flask Application**

```bash
python app.py
```

### Usage
Once the Flask application is running, open your web browser and go to http://127.0.0.1:5000/. You will see a form where you can input various personal and financial details. Upon submission, the application will predict your credit score and display a pie chart showing each model's confidence.
