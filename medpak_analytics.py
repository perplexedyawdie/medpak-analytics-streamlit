from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import streamlit as st

def bg_pred(df, g_col, time_col, bg_limit, steps):
    try:
        progress_text="Training model to predict glucose levels, please wait."
        pred_prog = st.progress(0, text=progress_text)
        print(f"g_col:{g_col} bg_limit:{bg_limit}")
        
        # read csv path and create df
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(by=time_col, ascending=True)
        recent_15_mins_data = df.tail(3)
        df = df.iloc[:-3]

        # df = df.drop(columns=['id'])

        # create time windows
        window_interval = 5  # time in minutes, smallest possible interval is 5 minutes

        frame_1 = 'glucose_minus_' + str(window_interval)
        frame_2 = 'glucose_minus_' + str(window_interval * 2)
        frame_3 = 'glucose_minus_' + str(window_interval * 3)

        frame_shift_1 = int(window_interval / 5)
        frame_shift_2 = int((window_interval * 2) / 5)
        frame_shift_3 = int((window_interval * 3) / 5)
        # print(frame_shift_1, frame_shift_2, frame_shift_3)

        df[frame_1] = df[g_col].shift(+frame_shift_1)
        df[frame_2] = df[g_col].shift(+frame_shift_2)
        df[frame_3] = df[g_col].shift(+frame_shift_3)
        pred_prog.progress(10, progress_text)
        # drop na values
        df = df.dropna()
        # print(df)


        lin_model = LinearRegression()

        rf_model = RandomForestRegressor(n_estimators=100, max_features=3, random_state=1)
        pred_prog.progress(15, progress_text)
        # organize and reshape data
        x1, x2, x3, y = df[frame_1], df[frame_2], df[frame_3], df[g_col]
        x1, x2, x3, y = np.array(x1), np.array(x2), np.array(x3), np.array(y)
        x1, x2, x3, y = x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1), y.reshape(-1, 1)
        final_x = np.concatenate((x1, x2, x3), axis=1)
        pred_prog.progress(25, progress_text)
        # split 70/30 into train and test sets
        X_train_size = int(len(final_x) * 0.7)
        set_index = len(final_x) - X_train_size
        # print(set_index)
        X_train, X_test, y_train, y_test = final_x[:-set_index], final_x[-set_index:], y[:-set_index], y[-set_index:]
        
        pred_prog.progress(30, progress_text)
        # fit models
        rf_model.fit(X_train, y_train.ravel())  # random forest
        lin_model.fit(X_train, y_train)  # linear regression
        pred_prog.progress(40, progress_text)
        # make Random Forest Regressor prediction
        pred = rf_model.predict(X_test)

        pred_prog.progress(60, progress_text)
        # make Linear Regression prediction
        lin_pred = lin_model.predict(X_test)
        # combine Aggregate predictions
        pred = pred.reshape(-1, 1)
        pred_prog.progress(80, progress_text)
        aggregate_pred_org = np.mean(np.array([lin_pred, pred]), axis=0)
        pred_prog.progress(100, progress_text)
        future_preds = []
        for i in range(steps):
            # Create features from the recent_15_mins_data
            frame_1 = recent_15_mins_data[g_col].iloc[-3]  # The data point from 15 minutes ago
            frame_2 = recent_15_mins_data[g_col].iloc[-2]  # The data point from 10 minutes ago
            frame_3 = recent_15_mins_data[g_col].iloc[-1]  # The most recent data point (from 5 minutes ago)
            
            # Create a feature array for prediction
            feature_array = np.array([[frame_1, frame_2, frame_3]])

            # Make predictions with the trained models
            lin_pred = lin_model.predict(feature_array).reshape(-1, 1)
            rf_pred = rf_model.predict(feature_array).reshape(-1, 1)

            # Calculate the aggregate prediction
            aggregate_pred = np.mean(np.array([lin_pred, rf_pred]), axis=0)[0, 0]

            # Save this prediction
            future_preds.append(aggregate_pred)
            
            # Update the recent_15_mins_data DataFrame with the new prediction
            new_row = pd.DataFrame({g_col: [aggregate_pred], time_col: [recent_15_mins_data[time_col].iloc[-1] + pd.Timedelta(minutes=5)]})
            recent_15_mins_data = pd.concat([recent_15_mins_data, new_row]).reset_index(drop=True)
            recent_15_mins_data = recent_15_mins_data.tail(3)  # Keep only the most recent 3 readings for the next iteration
        return np.array(future_preds), y_test, aggregate_pred_org
    except Exception as e:
        print("doing analytics")
        print(e)
        return None