# app.py

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_squared_error
import numpy as np
import base64
import bcrypt
import pyodbc
from sqlalchemy import create_engine
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(page_title="Hospital Demand & Capacity Modeling", layout="wide")

# Hide Streamlit style
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# User Authentication
users = {
    "admin": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()),
    "user": bcrypt.hashpw("user123".encode(), bcrypt.gensalt())
}

def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Login"):
        if username in users and bcrypt.checkpw(password.encode(), users[username]):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.sidebar.success(f"Logged in as {username}")
            st.rerun()  # Corrected function name
        else:
            st.sidebar.error("Invalid username or password")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login()
else:
    # Display Logo
    logo_path = "logo.jpeg"  # Replace with your logo file path
    try:
        st.sidebar.image(logo_path, use_column_width=True)
    except Exception as e:
        st.sidebar.write("")

    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Data Upload", "Data Visualization", "Forecasting", "Capacity Planning", "Reports", "Logout"],
            icons=["house", "cloud-upload", "bar-chart-line", "graph-up", "gear", "file-earmark-text", "box-arrow-right"],
            menu_icon="cast",
            default_index=0,
        )

    # Logout
    if selected == "Logout":
        st.session_state['logged_in'] = False
        st.experimental_rerun()

    # Global variables
    if 'data' not in st.session_state:
        st.session_state['data'] = None

    if 'forecast' not in st.session_state:
        st.session_state['forecast'] = None

    # Home Page
    if selected == "Home":
        if logo_path:
            try:
                st.image(logo_path, width=200)
            except:
                pass
        st.title("Hospital Demand & Capacity Modeling")
        st.markdown(f"""
        Welcome, **{st.session_state['username']}**, to the Hospital Demand & Capacity Modeling app. Use the sidebar to navigate through the app.
        - **Data Upload**: Upload your historical demand and capacity data.
        - **Data Visualization**: Explore your data through interactive charts.
        - **Forecasting**: Generate demand forecasts using statistical models.
        - **Capacity Planning**: Plan your capacity to meet future demand.
        - **Reports**: Generate comprehensive reports of your analysis.
        """)

    # Data Upload Page
    elif selected == "Data Upload":
        st.header("Data Upload")
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file, parse_dates=['Date'])
                st.session_state['data'] = data
                st.success("Data uploaded successfully!")
                st.write("Data Preview:")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error loading data: {e}")

        st.subheader("Or Load Data from SQL Server")
        if st.checkbox("Connect to SQL Server"):
            server = st.text_input("Server")
            database = st.text_input("Database")
            username = st.text_input("DB Username")
            password = st.text_input("DB Password", type='password')
            table = st.text_input("Table Name")
            if st.button("Load Data"):
                try:
                    engine = create_engine(f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server")
                    data = pd.read_sql_table(table, con=engine)
                    st.session_state['data'] = data
                    st.success("Data loaded successfully from SQL Server!")
                    st.write("Data Preview:")
                    st.dataframe(data.head())
                except Exception as e:
                    st.error(f"Error connecting to SQL Server: {e}")

    # Data Visualization Page
    elif selected == "Data Visualization":
        st.header("Data Visualization")
        if st.session_state['data'] is not None:
            data = st.session_state['data']
            departments = data['Department'].unique()
            dept_selection = st.multiselect("Select Departments", departments, default=departments)
            date_range = st.date_input("Select Date Range", [data['Date'].min(), data['Date'].max()])

            filtered_data = data[
                (data['Department'].isin(dept_selection)) &
                (data['Date'] >= pd.to_datetime(date_range[0])) &
                (data['Date'] <= pd.to_datetime(date_range[1]))
            ]

            # Interactive Plotly Chart
            fig = px.line(filtered_data, x='Date', y='Demand', color='Department', title='Demand Over Time')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please upload data first.")

    # Forecasting Page
    elif selected == "Forecasting":
        st.header("Forecasting")
        if st.session_state['data'] is not None:
            data = st.session_state['data']
            st.subheader("Select Forecasting Parameters")
            model_types = ["ARIMA", "Exponential Smoothing", "Prophet"]
            selected_models = st.multiselect("Select Models to Compare", model_types, default=model_types)
            periods = st.number_input("Forecast Periods (Days)", min_value=1, max_value=365, value=30)

            if st.button("Generate Forecast"):
                demand_series = data.groupby('Date')['Demand'].sum().reset_index()
                demand_series.columns = ['ds', 'y']
                demand_series = demand_series.sort_values('ds')
                results = {}
                metrics = {}
                model_errors = {}

                # Loop through selected models
                for model_name in selected_models:
                    try:
                        if model_name == "ARIMA":
                            arima_series = demand_series.set_index('ds')['y']
                            model = ARIMA(arima_series, order=(1, 1, 1))
                            model_fit = model.fit()
                            forecast = model_fit.forecast(steps=periods)
                            forecast = forecast.reset_index()
                            forecast.columns = ['ds', 'y']
                            results['ARIMA'] = forecast
                            # Compute AIC and RMSE
                            aic = model_fit.aic
                            bic = model_fit.bic
                            rmse = np.sqrt(mean_squared_error(arima_series[-periods:], model_fit.fittedvalues[-periods:]))
                            metrics['ARIMA'] = {'AIC': aic, 'BIC': bic, 'RMSE': rmse}

                        elif model_name == "Exponential Smoothing":
                            es_series = demand_series.set_index('ds')['y']
                            # Check if data has enough points for seasonal periods
                            if len(es_series) < 14:
                                raise ValueError("Not enough data points for Exponential Smoothing with seasonal components.")
                            model = ExponentialSmoothing(es_series, trend='add', seasonal='add', seasonal_periods=7)
                            model_fit = model.fit()
                            forecast = model_fit.forecast(steps=periods)
                            forecast = forecast.reset_index()
                            forecast.columns = ['ds', 'y']
                            results['Exponential Smoothing'] = forecast
                            # Compute AIC and RMSE
                            aic = model_fit.aic
                            bic = model_fit.bic
                            rmse = np.sqrt(mean_squared_error(es_series[-periods:], model_fit.fittedvalues[-periods:]))
                            metrics['Exponential Smoothing'] = {'AIC': aic, 'BIC': bic, 'RMSE': rmse}

                        elif model_name == "Prophet":
                            try:
                                # Updated Prophet code as above
                                # Split data into training and test sets
                                train_size = int(len(demand_series) * 0.8)
                                train_data = demand_series.iloc[:train_size]
                                test_data = demand_series.iloc[train_size:]

                                # Fit the Prophet model on training data
                                m = Prophet()
                                m.fit(train_data)

                                # Make predictions on test data
                                future = m.make_future_dataframe(periods=len(test_data), freq='D')
                                forecast = m.predict(future)

                                # Get the predictions for the test period
                                forecast_test = forecast.iloc[train_size:]

                        # Compute RMSE
                                y_true = test_data['y'].values
                                y_pred = forecast_test['yhat'].values
                                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                        # Store the forecast for the specified future periods
                                future_periods = periods
                                future_dates = m.make_future_dataframe(periods=future_periods)
                                forecast_full = m.predict(future_dates)
                                forecast_result = forecast_full[['ds', 'yhat']].tail(periods)
                                forecast_result.columns = ['ds', 'y']
                                results['Prophet'] = forecast_result

                        # Store the metrics
                                metrics['Prophet'] = {'AIC': None, 'BIC': None, 'RMSE': rmse}
                        except Exception as e:
                                st.error(f"Error loading data: {e}")
                    except Exception as e:
                        model_errors[model_name] = str(e)
                        st.warning(f"Model {model_name} encountered an error: {e}")

                if metrics:
                    # Display Metrics
                    st.subheader("Model Evaluation Metrics")
                    metrics_df = pd.DataFrame(metrics).T
                    st.write(metrics_df)

                    # Select Best Model based on AIC (lower is better)
                    # Filter out models with None AIC
                    valid_metrics = {k: v for k, v in metrics.items() if v['AIC'] is not None}
                    if valid_metrics:
                        best_model = min(valid_metrics.items(), key=lambda x: x[1]['AIC'])[0]
                    else:
                        # If all models have None AIC, fallback to RMSE
                        best_model = min(metrics.items(), key=lambda x: x[1]['RMSE'])[0]
                    st.success(f"The best model based on AIC is: {best_model}")

                    st.session_state['forecast'] = results[best_model]
                    st.session_state['best_model'] = best_model    

                    # Plot Forecast
                    st.subheader("Forecasted Demand")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=demand_series['ds'], y=demand_series['y'], mode='lines', name='Historical Demand'))
                    fig.add_trace(go.Scatter(x=st.session_state['forecast']['ds'], y=st.session_state['forecast']['y'], mode='lines', name='Forecasted Demand'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("All selected models failed to generate forecasts. Please check the data or select different models.")
            else:
                st.info("Select models and click 'Generate Forecast'")
        else:
            st.warning("Please upload data first.")

    # Capacity Planning Page
    elif selected == "Capacity Planning":
        st.header("Capacity Planning")
        if st.session_state['forecast'] is not None:
            forecast = st.session_state['forecast']
            st.subheader("Input Capacity Parameters")
            capacity = st.number_input("Available Capacity per Day", min_value=1, value=100)
            forecast_df = forecast.copy()
            forecast_df.columns = ['Date', 'Forecasted Demand']
            forecast_df['Capacity'] = capacity
            forecast_df['Surplus/Deficit'] = forecast_df['Capacity'] - forecast_df['Forecasted Demand']

            st.write(forecast_df)

            # Plotting with Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted Demand'], mode='lines', name='Forecasted Demand'))
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Capacity'], mode='lines', name='Capacity'))
            fig.update_layout(title='Capacity Planning', xaxis_title='Date', yaxis_title='Number of Patients')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please generate forecast first.")

    # Reports Page
    elif selected == "Reports":
        st.header("Generate Reports")
        if st.session_state['forecast'] is not None and st.session_state['data'] is not None:
            if st.button("Download Report"):
                # Generate a detailed PDF report using ReportLab
                buffer = BytesIO()
                p = canvas.Canvas(buffer, pagesize=letter)
                width, height = letter

                # Report Title
                p.setFont("Helvetica-Bold", 16)
                p.drawString(1 * inch, height - 1 * inch, "Hospital Demand & Capacity Report")

                # Insert Logo
                if logo_path:
                    try:
                        p.drawImage(logo_path, width - 2 * inch, height - 1.5 * inch, width=1.5 * inch, preserveAspectRatio=True)
                    except:
                        pass

                # User Information
                p.setFont("Helvetica", 12)
                p.drawString(1 * inch, height - 1.5 * inch, f"Prepared by: {st.session_state['username']}")
                p.drawString(1 * inch, height - 1.8 * inch, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Summary Section
                p.setFont("Helvetica-Bold", 14)
                p.drawString(1 * inch, height - 2.5 * inch, "Summary:")
                p.setFont("Helvetica", 12)
                text = p.beginText(1 * inch, height - 2.8 * inch)
                best_model = st.session_state.get('best_model', 'N/A')
                summary = f"This report provides a detailed analysis of the hospital's demand forecasts and capacity planning. The best forecasting model was determined to be {best_model} based on the evaluation metrics."
                for line in summary.split('\n'):
                    text.textLine(line)
                p.drawText(text)

                # Placeholder for Charts (In a real scenario, you can draw images of the charts)
                p.setFont("Helvetica-Bold", 14)
                p.drawString(1 * inch, height - 4 * inch, "Forecast Chart:")
                p.rect(1 * inch, height - 7 * inch, 6 * inch, 3 * inch)  # Placeholder rectangle

                # Finalize the PDF
                p.showPage()
                p.save()
                buffer.seek(0)
                b64 = base64.b64encode(buffer.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("Please complete forecasting and capacity planning first.")

