# Sales Prediction & Demand Forecasting (Machine Learning)

A machine learning project that predicts future sales using historical retail data.  
The system uses **Linear Regression** and provides an **interactive Gradio web interface** for business-friendly forecasting.

---

## Project Objective

Sales forecasting helps businesses:
- Plan inventory
- Manage cash flow
- Prepare staffing
- Reduce overstock and losses

This project demonstrates how machine learning supports **real-world business decisions**, not just model training.

---

## Dataset

**Sample – Superstore Dataset**

- Retail sales data with order dates and sales values
- Data aggregated at daily level
- Training data: before 2017
- Forecasting: 2017 and future dates (2018+)

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Gradio  

---

## Workflow

1. Load and clean historical sales data  
2. Convert order dates to datetime format  
3. Aggregate daily sales  
4. Create time-based features (Day, Month, Year)  
5. Train Linear Regression model  
6. Evaluate model using MAE and RMSE  
7. Forecast future sales  
8. Visualize results and deploy UI  

---

## Machine Learning Model

**Linear Regression**
- Simple and interpretable model
- Learns relationship between date features and sales
- Suitable for explaining forecasts to non-technical stakeholders

---

## Evaluation Metrics

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

These metrics measure prediction accuracy and error magnitude.

---

## Visual Output

- Actual vs Predicted Sales (2017)
- Future Sales Forecast (Next 30 Days)
- Interactive Gradio forecast dashboard

---

## Gradio Web Interface

The UI allows users to:
- Enter a start date
- Select number of forecast days
- View predictions in table and chart form

This makes the model usable for **business managers and decision-makers**.

---

## Author
Sankalp Gopal Patil
