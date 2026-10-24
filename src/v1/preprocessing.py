import numpy as np, pandas as pd, joblib
from pandas import get_dummies
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# figuring out the project root directory dynamically, regardless of where the script is run from
# but since this file is in src/v1/, we can safely go up two levels (parents[2]) to reach the project root
# this can written in config, but this is a more robust way to ensure that the project root is always correctly identified
PROJECT_ROOT = Path(__file__).resolve().parents[2] if '__file__' in locals() else Path.cwd()

# creating a custom transformer for feature engineering
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        
        x_copy = X.copy()
        
        # creating new features based on existing ones
        # why mapping feature time into 24 hours time?
        # while researching i found out that model performs better in dataset with cycle like this.
        # because if we only send df_['hours'], the model may simply understand it as 0 being the least and 23 being the highest,
        # but if we make hour_sin and hour_cos they will act as cordinate thus provide model with mathematical clarity of circle of time.
        # more on that later (while model prototyping).
        x_copy['hour'] = (x_copy['time'] // 3600) % 24
        x_copy['hour_sin'] = np.sin(2 * np.pi * x_copy['hour'] / 24.0)
        x_copy['hour_cos'] = np.cos(2 * np.pi * x_copy['hour'] / 24.0)
        
        x_copy.drop(columns=['time', 'hour'], inplace=True)
        
        return x_copy
    
    def get_feature_names_out(self, input_features=None):
        
        if input_features is None: return None
        
        # we want to ensure that the new features are appended to the original feature list
        features = [f for f in input_features if f not in ['time', 'hour']]
        
        new_features = ["hour_sin",
                        "hour_cos"]
        
        # we return the combined list of original and new features, ensuring that the new features are added at the end
        return np.array(features + new_features)


# renaming col to make things easier for us
def col_rename(x):
    x_copy = x.copy()
    
    x_copy.columns = [col.lower().replace(' ', '_') for col in x_copy.columns]
    
    return x_copy



# the main logic behind the preprocessing pipeline,
# which includes imputation, feature engineering, and scaling
# to avoid data leakage, we fit the pipeline on the training data and then transform the test data using the fitted pipeline
# use fit_transform='yes' for training data and fit_transform='no' for test data
def preprocessor(data, fit_transform='yes', std='yes', ver=1):
    
    data_copy = col_rename(data)
    
    # we will save the pipeline and scaler to disk for later use,
    # so we can apply the same transformations to new data during inference
    pipeline_path = PROJECT_ROOT / f'models/pipelines/v{ver}/pipeline_normal_v{ver}.joblib'
    fea_scaler_path = PROJECT_ROOT / f'models/scalers/v{ver}/fea_scaler_v{ver}.joblib'
    amt_scaler_path = PROJECT_ROOT / f'models/scalers/v{ver}/amt_scaler_v{ver}.joblib'

    # if fit_transform is 'yes', we fit the pipeline and scaler on the training data and save them to disk
    if fit_transform.lower().strip() == 'yes':
        
        # imputer for the dataset
        impute_full = ColumnTransformer(transformers=[
            ('cat_', SimpleImputer(strategy='most_frequent'), make_column_selector(dtype_exclude='number', dtype_include=['category', 'string'])),
            ('num_', SimpleImputer(strategy='median'), make_column_selector(dtype_exclude=['category', 'string'], dtype_include='number'))
        ], verbose_feature_names_out=False, remainder='passthrough')

        
        pipeline_normal = Pipeline(steps=[
            ('imputer', impute_full),
            ('feature_eng', CustomFeatureEngineer())])
        
        # we set the output of the pipeline to be a pandas DataFrame,
        # so we can easily access the feature names and work with the data
        pipeline_normal.set_output(transform="pandas")
        
        # we fit the pipeline on the training data and transform it, which will apply all the steps defined in the pipeline
        data_transformed = pipeline_normal.fit_transform(data_copy)
        
        # we save the fitted pipeline to disk so we can use it later to transform new data during inference (or for testing)
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(pipeline_normal, pipeline_path)

        print('Data preprocessing complete.\n')
        
        # checking if we need to scale the data,
        # if yes we will fit the scaler on the training data and returning both the transformed and scaled data
        if std.lower().strip() == 'yes':
            data_transformed_std = data_transformed.copy()
            
            # for features starting v like v1 to v28
            pca_cols = [col for col in data_transformed.columns if col.startswith('v')]
            
            amt_scaler = RobustScaler()
            fea_scaler = StandardScaler()
            
            amt_scaler.set_output(transform="pandas")
            fea_scaler.set_output(transform="pandas")
            
            data_transformed_std['amount'] = amt_scaler.fit_transform(data_transformed_std[['amount']])
            data_transformed_std[pca_cols] = fea_scaler.fit_transform(data_transformed_std[pca_cols])
            
            fea_scaler_path.parent.mkdir(parents=True, exist_ok=True)
            amt_scaler_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(fea_scaler, fea_scaler_path)
            joblib.dump(amt_scaler, amt_scaler_path)
            
            print('Returning the processed dataset with scaling.\n')
            
            return data_transformed, data_transformed_std
        
        print('Returning the processed dataset without scaling.\n')
        
        return data_transformed
    
    # if fit_transform is 'no', we load the saved pipeline and scaler from disk and use them to transform the test data
    else:
        # ensuring that the pipeline exist before trying to load it, to avoid runtime errors
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found at {pipeline_path}.")
        
        pipeline_normal = joblib.load(pipeline_path)
        
        pipeline_normal.set_output(transform="pandas")
        
        data_transformed = pipeline_normal.transform(data_copy)
        
        print('Data preprocessing complete.\n')
        
        if std.lower().strip() == 'yes':
            # ensuring that the scaler file exists before trying to load it, to avoid runtime errors
            if not fea_scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found at {fea_scaler_path}.")
            
            if not amt_scaler_path.exists():
                raise FileNotFoundError(f"Scaler file found at {fea_scaler_path}. Scaler file not found at {amt_scaler_path}.\n")
            
            data_transformed_std = data_transformed.copy()
            
            pca_cols = [col for col in data_transformed.columns if col.startswith('v')]
            
            fea_scaler = joblib.load(fea_scaler_path)
            amt_scaler = joblib.load(amt_scaler_path)
            
            fea_scaler.set_output(transform="pandas")
            amt_scaler.set_output(transform="pandas")
            
            data_transformed_std['amount'] = amt_scaler.transform(data_transformed_std[['amount']])
            data_transformed_std[pca_cols] = fea_scaler.transform(data_transformed_std[pca_cols])
            
            print('Returning the processed dataset with scaling.\n')
            
            return data_transformed, data_transformed_std
        
        print('Returning the processed dataset without scaling.\n')
        
        return data_transformed


# preprocessing pipeline for the target variable (y), which may includes imputation and type conversion
def y_preprocessor(y, ver=1):
    
    y_copy = y.copy()
    
    y_copy = y_copy.astype('int8')
    
    return y_copy


# loading the raw data, splitting it into training and test sets, and returning them
# if base_only is 'yes', we return only the training set with the target variable (y) included,
# otherwise we return both training and test sets separately
# it is for monitoring purposes, where we want to have a baseline dataset to compare against the current data
def load_data(ver=1, base_only='no', only_weight='no'):
    
    data = pd.read_csv(PROJECT_ROOT / "data/raw/creditcard.csv")
    
    data = data.drop_duplicates()
    
    X = data.drop(columns='Class')
    y = data['Class'].copy()
    
    y = y_preprocessor(y, ver=ver)
    
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        X = X.loc[y.index]
    
    # using stratify=y here as the dataset is highly imbalanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # use this if necessary
    if only_weight == 'yes':
        neg, pos = np.bincount(y_train)
            
        total = neg + pos
        
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        
        weight_ratio = weight_for_1 / weight_for_0
        
        return weight_ratio
        
    print("Data loading complete. Moving on to the next step in the pipeline.\n")
    
    if base_only == 'yes':
        return pd.concat([X_train, pd.Series(y_train, name='class', index=X_train.index)], axis=1)
    
    else:
        return X_train, X_test, y_train, y_test