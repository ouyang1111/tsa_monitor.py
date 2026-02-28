import requests
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from prophet import Prophet
import holidays

WECHAT_WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=6a94540b-22f3-4875-9553-b633b7fda8f4"

# =========================
# 1️⃣ 获取 TSA 历史数据
# =========================
def get_tsa_history():
    import pandas as pd
    import requests

    url = "https://www.tsa.gov/travel/passenger-volumes"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()  # 如果还有问题，会报错
    tables = pd.read_html(r.text)
    df = tables[0]

    df.columns = ["date","current_year","last_year"]
    df["date"] = pd.to_datetime(df["date"])
    df["current_year"] = df["current_year"].str.replace(",","").astype(int)
    df["last_year"] = df["last_year"].str.replace(",","").astype(int)
    df = df.sort_values("date")
    return df

# =========================
# 2️⃣ 特征工程
# =========================
def prepare_data(df):
    df["weekday"] = df["date"].dt.weekday
    df["ma7"] = df["current_year"].rolling(7).mean()
    df["ma30"] = df["current_year"].rolling(30).mean()
    df["trend"] = df["current_year"].diff()
    df["yoy_abs"] = df["current_year"] - df["last_year"]
    df["yoy_pct"] = df["yoy_abs"] / df["last_year"]
    df["month"] = df["date"].dt.month
    us_holidays = holidays.US()
    df["holiday"] = df["date"].apply(lambda x: 1 if x in us_holidays else 0)
    df = df.fillna(0)
    return df

# =========================
# 3️⃣ 线性模型训练
# =========================
def train_linear(df):
    X = df[["weekday","ma7","ma30","trend","yoy_abs","yoy_pct","month","holiday"]]
    y = df["current_year"]
    model = LinearRegression()
    model.fit(X,y)
    residual = y - model.predict(X)
    std = np.std(residual)
    return model,std

# =========================
# 4️⃣ XGBoost 模型训练
# =========================
def train_xgb(df):
    X = df[["weekday","ma7","ma30","trend","yoy_abs","yoy_pct","month","holiday"]]
    y = df["current_year"]
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
    model.fit(X,y)
    return model

# =========================
# 5️⃣ Prophet 模型训练
# =========================
def train_prophet(df):
    prophet_df = df[["date","current_year"]].rename(columns={"date":"ds","current_year":"y"})
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    us_holidays = holidays.US()
    holiday_df = pd.DataFrame({
        'holiday':'US_holiday',
        'ds':[d for d in df["date"] if d in us_holidays],
        'lower_window':0,'upper_window':1
    })
    m.add_country_holidays(country_name='US')
    m.fit(prophet_df)
    return m

# =========================
# 6️⃣ 预测次日
# =========================
def predict_next_day(df, linear_model, linear_std, xgb_model, prophet_model):
    last_row = df.iloc[-1]
    tomorrow = last_row["date"] + datetime.timedelta(days=1)

    weekday = tomorrow.weekday()
    ma7 = df["current_year"].tail(7).mean()
    ma30 = df["current_year"].tail(30).mean()
    trend = df["current_year"].iloc[-1] - df["current_year"].iloc[-2]

    last_year_row = df[df["date"] == (tomorrow - datetime.timedelta(days=365))]
    if not last_year_row.empty:
        yoy_abs = df["current_year"].iloc[-1] - last_year_row["current_year"].values[0]
        yoy_pct = yoy_abs / last_year_row["current_year"].values[0]
    else:
        yoy_abs = 0
        yoy_pct = 0

    month = tomorrow.month
    us_holidays = holidays.US()
    holiday = 1 if tomorrow in us_holidays else 0

    X_new = np.array([[weekday, ma7, ma30, trend, yoy_abs, yoy_pct, month, holiday]])

    pred_linear = linear_model.predict(X_new)[0]
    pred_xgb = xgb_model.predict(X_new)[0]

    # Prophet预测
    future = pd.DataFrame({"ds":[tomorrow]})
    pred_prophet = prophet_model.predict(future)["yhat"].values[0]

    # 多模型融合（加权平均）
    pred_fusion = (pred_linear + pred_xgb + pred_prophet)/3

    # 区间估计
    lower = pred_fusion - 1.96*linear_std
    upper = pred_fusion + 1.96*linear_std

    return pred_linear, pred_xgb, pred_prophet, pred_fusion, lower, upper, ma7

# =========================
# 7️⃣ 交易信号和仓位
# =========================
def trading_signal(pred, ma7, winrate):
    diff = (pred - ma7)/ma7
    # 简化仓位控制
    kelly = winrate - (1-winrate)
    size = max(0, min(kelly * abs(diff), 0.3))
    if diff > 0.02:
        signal = "做多"
    elif diff < -0.02:
        signal = "做空"
    else:
        signal = "观望"
    return signal, round(size,3)

# =========================
# 8️⃣ 滚动回测胜率
# =========================
def rolling_backtest(df, window=30):
    wins = 0
    total = 0
    for i in range(window, len(df)-1):
        train_df = df.iloc[i-window:i]
        test_df = df.iloc[i:i+1]

        linear_model, _ = train_linear(train_df)
        X_test = test_df[["weekday","ma7","ma30","trend","yoy_abs","yoy_pct","month","holiday"]]
        pred = linear_model.predict(X_test)[0]
        actual = test_df["current_year"].values[0]
        prev = df.iloc[i-1]["current_year"]

        if (pred-prev)*(actual-prev) > 0:
            wins += 1
        total += 1
    return round(wins/total,3) if total>0 else 0

# =========================
# 9️⃣ 主程序
# =========================
def main():
    df = get_tsa_history()
    df = prepare_data(df)

    linear_model, linear_std = train_linear(df)
    xgb_model = train_xgb(df)
    prophet_model = train_prophet(df)

    winrate = rolling_backtest(df)

    pred_linear, pred_xgb, pred_prophet, pred_fusion, lower, upper, ma7 = predict_next_day(
        df, linear_model, linear_std, xgb_model, prophet_model
    )

    signal, size = trading_signal(pred_fusion, ma7, winrate)

    msg = f"""
📊 TSA量化预测系统

模型预测:
- 线性回归: {int(pred_linear)}
- XGBoost: {int(pred_xgb)}
- Prophet: {int(pred_prophet)}
- 融合预测: {int(pred_fusion)}

预测区间: {int(lower)} - {int(upper)}
7日均值: {int(ma7)}
滚动胜率: {winrate}

交易信号: {signal}
建议仓位: {size*100}%
"""

    requests.post(WECHAT_WEBHOOK,json={"msgtype":"text","text":{"content":msg}})

if __name__ == "__main__":
    main()
