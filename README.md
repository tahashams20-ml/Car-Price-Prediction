 Car Price Prediction Using Machine Learning

A machine learning web application that estimates a car's market price based on its specifications using a pre-trained **Random Forest Regressor**. The application provides an interactive interface for entering vehicle details, viewing predicted prices, and exploring dataset insights, model performance, and feature importance.

 Live Demo

[Try the Car Price Predictor](https://carprice-six.vercel.app/)
 About the Project

The goal of this project is to estimate vehicle prices from their technical specifications and other characteristics. The model is trained on a cleaned dataset of approximately 1,200 cars and deployed through an interactive web application.

Users can enter a car's specifications and receive an estimated price along with an indicative price range.

 Dataset Information

| Metric | Value |
|---|---:|
| Total Cars | 1,212 |
| Companies / Brands | 37 |
| Encoded Features | 1,618 |
| Median Price | $42,808 |
| Maximum Price | $18,000,000 |

The dataset contains vehicle specifications and pricing information used to train and evaluate the regression model.

### Key Features

- Car name and company
- Engine type and engine size
- Fuel type
- Engine displacement / battery capacity
- Horsepower and torque
- Maximum speed
- Acceleration time (0–100 km/h)
- Seating capacity
- Electric vehicle indicator

Categorical variables, including car name, company, engine type, and fuel type, are one-hot encoded before being passed to the model.

## 🤖 Machine Learning Approach

**Algorithm:** Random Forest Regressor

The model uses an ensemble of decision trees to learn relationships between vehicle characteristics and prices.

### Model Configuration

- Number of trees:** 100
- Minimum samples per leaf:** 2
- Target transformation:** `log1p(price)`
- Prediction conversion:** `expm1(prediction)`

The target price is log-transformed during training to help handle its wide range of values. Predictions are converted back to the original price scale before being displayed.

### Model Evaluation

The dataset is split into training and testing subsets using an 80/20 split with `random_state=42`.

Evaluation metrics are calculated on the same hold-out split used during model training. The application includes a model performance section for examining regression results.

Application Features

- **Car Price Prediction:** Estimate a vehicle's price using its specifications.
- **Dataset Information:** Explore dataset statistics and feature details.
- **Model Performance:** Review regression evaluation metrics.
- **Feature Importance:** Examine which features contribute most to the trained model's predictions.
- **Interactive Dashboard:** Navigate a user-friendly interface with visualizations.

 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Data processing and model development |
| Flask | Backend API |
| scikit-learn | Random Forest Regression and preprocessing |
| JavaScript | Frontend interactions |
| Chart.js | Data visualization |
| HTML / CSS | Web interface |
| Vercel | Application deployment |

 Project Workflow

1. Load and clean the vehicle dataset.
2. Prepare numerical and categorical features.
3. One-hot encode categorical variables.
4. Apply `log1p` transformation to target prices.
5. Train a Random Forest Regressor.
6. Evaluate the model on a hold-out test set.
7. Convert predicted log prices back to the original scale using `expm1`.
8. Serve predictions through a Flask API and display results in the web interface.

 Business Use Case

Car price estimation can help prospective buyers, sellers, and automotive businesses understand indicative vehicle valuations based on vehicle characteristics.

The model demonstrates how machine learning regression can be integrated into an interactive application to support data-driven price estimation.

 Disclaimer

Predictions are estimates based on the dataset and the trained model. They may differ from actual market prices due to vehicle condition, mileage, location, market demand, and other factors. The application is intended for educational and demonstration purposes only and should not be treated as a formal vehicle valuation or financial advice.

 Author

Developed as a Data Science and Machine Learning portfolio project.
