import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



dataset_name = st.sidebar.selectbox("Select Dataset", ("Car Price Prediction", "50_Startups",
                                                       "Social_Network_Ads", "Person Body Type"))

# Importing the datasets :
def get_dataset(dataset_name):
    if dataset_name == "Car Price Prediction":
        dataset = pd.read_csv('Data/CarPrice_Assignment.csv')
        dataset.drop(['enginetype','fuelsystem'], inplace=True, axis=1)
        X = dataset.iloc[:, 3:-1].values
        y = dataset.iloc[:, -1].values

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 0] = le.fit_transform(X[:, 0])
        X[:, 1] = le.fit_transform(X[:, 1])
        X[:, 2] = le.fit_transform(X[:, 2])
        X[:, 3] = le.fit_transform(X[:, 3])
        X[:, 4] = le.fit_transform(X[:, 4])
        X[:, 5] = le.fit_transform(X[:, 5])
        X[:, 11] = le.fit_transform(X[:, 11])

    elif dataset_name == "Heart Disease Prediction":
       dataset = pd.read_csv('Data/heart.csv')
       X = dataset.iloc[:, :-1].values
       y = dataset.iloc[:, -1].values
    elif dataset_name == "Social_Network_Ads":
       dataset = pd.read_csv('Data/Social_Network_Ads.csv')
       X = dataset.iloc[:, :-1].values
       y = dataset.iloc[:, -1].values
       
    elif dataset_name == "Person Body Type":
       dataset = pd.read_csv('Data/500_Person_Gender_Height_Weight_Index.csv')
       X = dataset.iloc[:, :-1].values
       y = dataset.iloc[:, -1].values
       from sklearn.preprocessing import LabelEncoder
       le = LabelEncoder()
       X[:, 0] = le.fit_transform(X[:, 0])
       
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
        user_input1 = 1
        if user_inp1 == "Gas":
            user_input1 = 1
        elif user_inp1 == "Diesel":
            user_input1 = 0
        user_inp2=st.selectbox("aspiration", ("STD", "Turbo")) 
        user_input2 = 0
        if user_inp2 == "STD":
            user_input2 = 0
        elif user_inp2 == "Turbo":
            user_input2 = 1
        user_inp3=st.selectbox("doornumber", ("Four", "Two")) 
        user_input3 = 0
        if user_inp3 == "four":
            user_input3 = 0
        elif user_inp3 == "Two":
            user_input2 = 1
        user_inp4=st.selectbox("carbody", ("Sedan", "Hatchback", "Wagon", "Hardtop", "Convertible")) 
        user_input4 = 3
        if user_inp4 == "Sedan":
            user_input4 = 3
        elif user_inp4 == "Hatchback":
            user_input4 = 2
        elif user_inp4 == "Wagon":
            user_input4 == 4
        elif user_inp4 == "Hardtop":
            user_input4 = 1
        elif user_inp4 == "Convertible":
            user_input4 == 0
        user_inp5=st.selectbox("drivewheel", ("FWD", "RWD", "4WD")) 
        user_input5 = 1
        if user_inp5 == "FWD":
            user_input5 = 1
        elif user_inp5 == "RWD":
            user_input5 = 2
        elif user_inp5 == "4WD":
            user_input5 == 0
        user_inp6=st.selectbox("enginelocation", ("Front", "Rear")) 
        user_input6 = 0
        if user_inp3 == "Front":
            user_input6 = 0
        elif user_inp6 == "Rear":
            user_input6 = 1
        user_input7=st.slider("wheelbase", 85, 125, step=1)
        user_input8=st.slider("carlenght", 140, 210, step=1)
        user_input9=st.slider("carwidth", 60.0, 73.0, step=0.1)
        user_input10=st.slider("carheight", 47.0, 60.0, step=0.1)
        user_input11=st.slider("curbweight", 1485, 4070, step=1)
        user_inp12=st.selectbox("cylindernumber", ("Two", "Three", "Four", "Five", "Six", "Eight", "Twelve")) 
        user_input12 = 2
        if user_inp12 == "Two":
            user_input12 = 6
        elif user_inp12 == "Three":
            user_input12 = 4
        elif user_inp12 == "Four":
            user_input12 == 2
        elif user_inp12 == "Five":
            user_input12 = 1
        elif user_inp12 == "Six":
            user_input12 == 3
        elif user_inp12 == "Eight":
            user_input12 == 0
        elif user_inp12 == "Twelve":
            user_input12 == 5
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
    
if dataset_name == "Heart Disease Prediction":
    if user_pred == 0:
        st.info("Absence of Heart Disease")
    elif user_pred ==1:
        st.warning("Presence of Heart Disease")
if dataset_name == "Social_Network_Ads":
    if user_pred == 1:
        st.info("Purchased the item before")
    elif user_pred == 0:
        st.warning("Didn't purchase the item before")
if dataset_name == "Person Body Type":
    if user_pred == 0:
        st.error("Extremely Weak")
    elif user_pred == 1:
        st.info("Weak")
    elif user_pred == 2:
        st.success("Normal")
    elif user_pred == 3:
        st.info("Overweight")
    elif user_pred == 4:
        st.warning("Obesity")
    elif user_pred == 5:
        st.error("Extreme Obesity")
        
        
#Showing the accuaracy results
from sklearn.metrics import r2_score
y_pred = reg.predict(X_test)

Score = r2_score(y_test, y_pred)
Score = float(np.round(Score,3))
st.sidebar.info(f'Accuracy = {Score}')



    
         

