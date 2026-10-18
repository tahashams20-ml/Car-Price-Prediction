import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import re
df=pd.read_csv("Cars.csv",encoding="cp1252")
print(df.head(n=10))
print(df.info())
print(df.dtypes)
#Convert columns to appropriate data types
df['CC/Battery Capacity']=(df['CC/Battery Capacity'].str.replace(",","",regex=False).str.extract(r"(\d+\.?\d*)")[0].astype(float))
df['Total Speed']=(df['Total Speed'].str.extract(r"(\d+\.?\d*)")[0].astype(float))
df['Performance(0 - 100 )KM/H']=(df['Performance(0 - 100 )KM/H'].str.extract(r"(\d+\.?\d*)")[0].astype(float))
df["Engine Size"]=(df["Engines"].str.extract(r"(\d+\.?\d*)\s*[LI]")[0].astype(float))
df['Is_electric']=(df["Engines"].str.contains("electric",case=False,na=False).astype(int))
df["Seats"]=pd.to_numeric(df['Seats'],errors='coerce')
#r"(\d+\.?\d*)" is a regular expression (regex) used to find a number inside text.
#Extract ranges of Horse Power and Torque then convert to mean
def convert_range(value):
 value=str(value).replace(",","")
 numbers=re.findall(r"(\d+\.?\d*)",value)
 numbers=[float(n) for n in numbers]
 if len(numbers)==0:
    return None
 elif len(numbers)==1:
    return numbers[0]
 else :
   return (numbers[0]+numbers[1])/2
df["HorsePower"]=df["HorsePower"].apply(convert_range)
df["Torque"]=df["Torque"].apply(convert_range)

#Extract ranges of Price and convert to mean
def convert_price(price):
    price = str(price).replace("$", "").replace(",", "").strip()
    nums = [float(n) for n in re.findall(r"\d+\.?\d*", price)]
    if len(nums)==0:
       return None
    elif len(nums)==1:
       return nums[0]
    else:
       return (nums[0]+nums[1])/2

df["Cars Prices"] = df["Cars Prices"].apply(convert_price)
print(df.info())
#Remove duplicates
print(df.duplicated().sum())
df=df.drop_duplicates()
# Fill missing numeric values
numeric_columns = [
    "CC/Battery Capacity",
    "Total Speed",
    "Performance(0 - 100 )KM/H",
    "Engine Size",
    "HorsePower",
    "Torque",
    "Seats"
    
]

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())
df=df.dropna(subset=["Cars Prices"])
print(df['Cars Prices'].describe())
df.to_csv("cleaned_cars.csv", index=False)
# Log transform target to reduce the effect of highly expensive cars
df["Cars Prices"]=np.log1p(df["Cars Prices"])
#Train Test split
x=df.drop(columns=["Cars Prices"])
y=df["Cars Prices"]
#One hot encoding
x=pd.get_dummies(x,columns=["Cars Names","Engines","Fuel Types","Company Names"],drop_first=True)
x_train,x_test,y_train,y_test=train_test_split(
   x,y,test_size=0.20,random_state=42
)
#Create and Train model
model=RandomForestRegressor(
   n_estimators=100,
   n_jobs=-1,
   max_depth=None,
   min_samples_leaf=2,
   random_state=42
)
model.fit(x_train,y_train)
pred=model.predict(x_test)
pred_log=np.expm1(pred)
y_true=np.expm1(y_test)

mae = mean_absolute_error(y_true,pred_log)
rmse = np.sqrt(mean_squared_error(y_true,pred_log))
r2 = r2_score(y_true,pred_log)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
# =========================
# Feature Importance
# =========================

feature_importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": model.feature_importances_
})

# Sort from most important to least important
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


# =========================
# Save Model
# =========================
import joblib
joblib.dump(model,"Car_Price.pkl")


