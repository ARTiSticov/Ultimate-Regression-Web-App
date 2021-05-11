import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder


dataset_name = st.sidebar.selectbox("Select Dataset", ("Car Price Prediction", "Combined Cycle Power Plant",
                                                       "Startup Company Profit", "Real Estate Price Prediction"))

# Importing the datasets :
def get_dataset(dataset_name):
    if dataset_name == "Car Price Prediction":
        dataset = pd.read_csv('Data/CarPrice_Assignment.csv')
        dataset.drop(['enginetype','fuelsystem'], inplace=True, axis=1)
        X = dataset.iloc[:, 3:-1].values
        y = dataset.iloc[:, -1].values

        le = LabelEncoder()
        X[:, 0] = le.fit_transform(X[:, 0])
        X[:, 1] = le.fit_transform(X[:, 1])
        X[:, 2] = le.fit_transform(X[:, 2])
        X[:, 3] = le.fit_transform(X[:, 3])
        X[:, 4] = le.fit_transform(X[:, 4])
        X[:, 5] = le.fit_transform(X[:, 5])
        X[:, 11] = le.fit_transform(X[:, 11])

    elif dataset_name == "Combined Cycle Power Plant":
       dataset = pd.read_csv('Data/Folds5x2_pp.csv')
       X = dataset.iloc[:, :-1].values
       y = dataset.iloc[:, -1].values
       
    elif dataset_name == "Startup Company Profit":
        dataset = pd.read_csv('Data/50_Startups.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        le = LabelEncoder()
        X[:, 3] = le.fit_transform(X[:, 3])

        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
        (X[:,0:3])=imputer.fit_transform(X[:,0:3])
    
    elif dataset_name == "Real Estate Price Prediction":
        dataset = pd.read_csv('Data/Real Estate.csv')
        dataset.drop(['No', 'Transaction Date'], inplace=True, axis=1)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

    return X, y


X, y = get_dataset(dataset_name)
st.title(f'{dataset_name}')
st.sidebar.write("shape of dataset", X.shape)
regressor_name = st.sidebar.selectbox("Select Regressor", ("Multiple Linear Regressor",  "SVR",
                                                             "Decision Tree Regressor", "Random Forest Regressor", "XGBoost"))

def dataset_input(dataset_name):
    inputs = dict()
    if dataset_name == "Car Price Prediction":
        user_inp1=st.selectbox("fueltype", ("Gas", "Diesel"))
        def switch(user_inp1):
            return{
                "Gas"     : 1,
                "Diesel"    : 0
                }.get(user_inp1, "invalid input")
        user_input1 = switch(user_inp1)
        
        user_inp2=st.selectbox("aspiration", ("STD", "Turbo")) 
        def switch(user_inp2):
            return{
                "STD"     : 0,
                "Turbo"    : 1
                }.get(user_inp2, "invalid input")
        user_input2 = switch(user_inp2)
        
        user_inp3=st.selectbox("doornumber", ("Four", "Two")) 
        def switch(user_inp3):
            return{
                "Four"     : 0,
                "Two"    : 1
                }.get(user_inp3, "invalid input")
        user_input3 = switch(user_inp3)
        
        user_inp4=st.selectbox("carbody", ("Sedan", "Hatchback", "Wagon", "Hardtop", "Convertible")) 
        def switch(user_inp4):
            return{
                "Sedan"         : 3,
                "Hatchback"     : 2,
                "Wagon"         : 4,
                "Hardtop"       : 1,
                "Convertible"   : 0
                }.get(user_inp4, "invalid input")
        user_input4 = switch(user_inp4)
        
        user_inp5=st.selectbox("drivewheel", ("FWD", "RWD", "4WD")) 
        def switch(user_inp5):
            return{
                "FWD"     : 1,
                "RWD"     : 2,
                "4WD"     : 0
                }.get(user_inp5, "invalid input")
        user_input5 = switch(user_inp5)
            
        user_inp6=st.selectbox("enginelocation", ("Front", "Rear")) 
        def switch(user_inp6):
            return{
                "Front"     : 0,
                "Rear"    : 1
                }.get(user_inp6, "invalid input")
        user_input6 = switch(user_inp6)
            
        user_input7=st.slider("wheelbase", 85, 125, step=1)
        user_input8=st.slider("carlenght", 140, 210, step=1)
        user_input9=st.slider("carwidth", 60.0, 73.0, step=0.1)
        user_input10=st.slider("carheight", 47.0, 60.0, step=0.1)
        user_input11=st.slider("curbweight", 1485, 4070, step=1)
        user_inp12=st.selectbox("cylindernumber", ("Two", "Three", "Four", "Five", "Six", "Eight", "Twelve")) 
        def switch(user_inp12):
            return{
                "Two"         : 6,
                "Three"       : 4,
                "Four"        : 2,
                "Five"        : 1,
                "Six"         : 3,
                "Eight"       : 0,
                "Twelve"      : 5
                }.get(user_inp12, "invalid input")
        user_input12 = switch(user_inp12)
            
        user_input13=st.slider("enginesize", 60, 330, step=1)
        user_input14=st.slider("boreratio", 2.50, 4.00, step=0.1)
        user_input15=st.slider("stroke", 2.00, 4.20, step=0.1)
        user_input16=st.slider("compressionratio", 7, 23, step=1)
        user_input17=st.slider("horsepower", 48, 300, step=1)
        user_input18=st.slider("peakrpm", 4150, 6700, step=1)
        user_input19=st.slider("citympg", 13, 49, step=1)
        user_input20=st.slider("highwaympg", 16, 54, step=1)
        
        inputs=[[user_input1, user_input2, user_input3, user_input4, user_input5, user_input6,
                 user_input7, user_input8, user_input9, user_input10, user_input11, user_input12,
                 user_input13, user_input14, user_input15, user_input16, user_input17, user_input18,
                 user_input19, user_input20]]
          
        
    if dataset_name == "Combined Cycle Power Plant":
        st.subheader("Predict the Electrical Energy Output based on the features that consist of hourly average ambient variables")
        user_input1=st.slider("Temperature (°C) ", 1.80, 38.00, step=0.1)
        user_input2=st.slider("Exhaust Vaccum (cm Hg)", 25.0, 82.0, step=0.1)
        user_input3=st.slider("Ambient Pressure (milibar)", 990, 1005, step=1)
        user_input4=st.slider("Relative Humidit (percentage %)", 25.0, 100.0, step=0.1)
        
        inputs=[[user_input1, user_input2, user_input3, user_input4]]
        
        
    if dataset_name == "Startup Company Profit":
        st.subheader("Determine the expected profit on startup of a company")
        user_input1=st.slider("Research & Development Spend", 542.00, 166000.00, step=10.0)
        user_input2=st.slider("Administration", 51283.00, 183000.00, step=10.0)
        user_input3=st.slider("Marketing Spend", 1903.00, 472000.00, step=10.0)
        user_inp4=st.selectbox("state", ("New York", "California", "Florida")) 
        def switch(user_inp4):
            return{
                "New York"     : 2,
                "California"     : 0,
                "Florida"     : 1
                }.get(user_inp4, "invalid input")
        user_input4 = switch(user_inp4)
        
        inputs=[[user_input1, user_input2, user_input3, user_input4]]
        
        
    if dataset_name == "Real Estate Price Prediction":
        
        user_input1=st.slider("House Age", 0, 44, step=1)
        user_input2=st.slider("Distance to the nearest MRT station", 23.00, 6500.00)
        user_input3=st.slider("Number of convenience stores", 0, 10, step=1)
        user_input4=st.slider("Latitude", 24.9320, 25.0146)
        user_input5=st.slider("Longitude", 121.470, 121.567)

        inputs=[[user_input1, user_input2, user_input3, user_input4,
                 user_input5]]
        
    return inputs

inputs = dataset_input(dataset_name)


    
# Adding the models parameters :
def add_parameter_ui(reg_name):
    params = dict()

    if reg_name == "SVR":
        kernel = st.sidebar.selectbox(("Choose Kernel function"), ('rbf', 'sigmoid', 'poly', 'linear'))
        C = st.sidebar.slider("C", 1, 70)
        params['kernel'] = kernel
        params["C"] = C
        
    elif reg_name =="Decision Tree Regressor":
        criterion = st.sidebar.selectbox(("Choose a Criterion method"), ('mse', 'mae'))
        max_depth = st.sidebar.slider("max_depth", 1, 10)
        params['criterion'] = criterion
        params["max_depth"] = max_depth
        
    elif reg_name == "Random Forest Regressor":
        criterion = st.sidebar.selectbox(("Choose a Criterion method"), ('mse', 'mae'))
        n_estimators = st.sidebar.slider("n_estimators", 100, 1000)
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params['criterion'] = criterion
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
        
    elif reg_name == "XGBoost":
        n_estimators = st.sidebar.slider("n_estimators", 100, 1000)
        max_depth = st.sidebar.slider("max_depth", 1, 10)
        learning_rate = st.sidebar.slider("learning_rate", 0.1, 0.01)
        subsample = st.sidebar.slider("subsample", 0.5, 1.0)
        min_child_weight = st.sidebar.slider("min_child_weight", 1, 10)
        colsample_bytree = st.sidebar.slider("colsample_bytree", 0.3, 1.0)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
        params["learning_rate"] = learning_rate
        params["subsample"] = subsample
        params["min_child_weight"] = min_child_weight
        params["colsample_bytree"] = colsample_bytree
        
    return params

params = add_parameter_ui(regressor_name)


# Choosing the regressors :
def get_regressor(reg_name, params):
    if reg_name == "Multiple Linear Regressor":
            reg = LinearRegression()

    elif reg_name == "SVR":
              reg = SVR(kernel = params['kernel'],
                              C = params['C'])
        
               
    elif reg_name == "Decision Tree Regressor":
        reg = DecisionTreeRegressor(max_depth = params['max_depth'],
                                    criterion = params['criterion'], random_state = 0)
                                     
                  
        
    elif reg_name == "Random Forest Regressor":
              reg = RandomForestRegressor(criterion = params['criterion'],
                                          max_depth = params["max_depth"],
                                          n_estimators = params["n_estimators"],
                                           random_state = 0)
              
    elif reg_name == "XGBoost":
              reg = XGBRegressor(max_depth = params["max_depth"],
                                  n_estimators = params['n_estimators'],
                                  learning_rate = params["learning_rate"],
                                  subsample = params["subsample"],
                                  min_child_weight = params["min_child_weight"],
                                  colsample_bytree = params["colsample_bytree"])
    return reg

reg = get_regressor(regressor_name, params)

#Splitting the dataset into the Training set and Test set :
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the model on the training set :
reg.fit(X_train, y_train)

#Making the perdictions :
user_pred = reg.predict(sc.transform(inputs))
user_pred = float(np.round(user_pred))
user_pred = '{:,.2f}'.format(user_pred)
print(user_pred)

if dataset_name == "Car Price Prediction":
    st.sidebar.info(f'The Predicted Price = {user_pred}$')
    
if dataset_name == "Combined Cycle Power Plant":
    st.sidebar.info(f'The Predicted Electrical Energy Output = {user_pred}MW')
    
if dataset_name == "Startup Company Profit":
    st.sidebar.info(f'The Predicted Profit = {user_pred}$')
    
if dataset_name == "Real Estate Price Prediction":
    st.sidebar.info(f'House Price of Unit Area = {user_pred} ')
        
#Showing the accuaracy results
from sklearn.metrics import r2_score
y_pred = reg.predict(X_test)

Score = r2_score(y_test, y_pred)
Score = float(np.round(Score,3))
st.sidebar.info(f'R2 Score = {Score}')

if st.sidebar.button("what is R2 Score ?"):
    result=st.sidebar.write("R2  compares the fit of the chosen model with that of a horizontal straight line (the null hypothesis).\
        If the chosen model fits worse than a horizontal line, then R2 is negative")


         

