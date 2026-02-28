import pandas as pd
import numpy as np
import datetime
import requests
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from prophet import Prophet
import holidays

# ============ 填你的企业微信 webhook =============
WECHAT_WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=6a94540b-22f3-4875-9553-b633b7fda8f4"

# =========================
# 1️⃣ 获取 TSA 历史数据（读取CSV）
# =========================
def get_tsa_history():
    import pandas as pd
    # 指定编码为 'utf-8-sig' 或 'latin1' 避免解码错误
    df = pd.read_csv("data/tsa_history.csv", encoding="utf-8-sig")
    # 如果utf-8-sig报错, 改成 encoding="cp1252"
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["last_year"] = df["current_year"].shift(365)
    df = df.fillna(0)
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
    df["yoy_pct"] = df["yoy_abs"] / (df["last_year"].replace(0,1))
    df["month"] = df["date"].dt.month
    us_holidays = holidays.US()
    df["holiday"] = df["date"].apply(lambda x: 1 if x in us_holidays else 0)
    df = df.fillna(0)
    return df

# =========================
# 3️⃣ 线性回归训练
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
# 4️⃣ XGBoost训练
# =========================
def train_xgb(df):
    X = df[["weekday","ma7","ma30","trend","yoy_abs","yoy_pct","month","holiday"]]
    y = df["current_year"]
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
    model.fit(X,y)
    return model

# =========================
# 5️⃣ Prophet训练
# =========================
def train_prophet(df):
    prophet_df = df[["date","current_year"]].rename(columns={"date":"ds","current_year":"y"})
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
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

    # 去年同比
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

    future = pd.DataFrame({"ds":[tomorrow]})
    pred_prophet = prophet_model.predict(future)["yhat"].values[0]

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
    kelly = max(0, min(winrate - (1-winrate), 0.3))  # 简化仓位
    if diff > 0.02:
        signal = "做多"
    elif diff < -0.02:
        signal = "做空"
    else:
        signal = "观望"
    size = round(kelly*100,1)
    return signal, size

# =========================
# 8️⃣ 滚动回测胜率
# =========================
def rolling_backtest(df, window=30):
    wins = 0
    total = 0
    for i in range(window, len(df)-1):
        train_df = df.iloc[i-window:i]
        test_df = df.iloc[i:i+1]
        linear_model,_ = train_linear(train_df)
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

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"""
📊 TSA量化预测系统 ({now})

模型预测:
- 线性回归: {int(pred_linear)}
- XGBoost: {int(pred_xgb)}
- Prophet: {int(pred_prophet)}
- 融合预测: {int(pred_fusion)}

预测区间: {int(lower)} - {int(upper)}
7日均值: {int(ma7)}
滚动胜率: {winrate}

交易信号: {signal}
建议仓位: {size}%
"""

    requests.post(WECHAT_WEBHOOK,json={"msgtype":"text","text":{"content":msg}})

if __name__ == "__main__":
    main()
