import requests
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import holidays

WECHAT_WEBHOOK = "填你的webhook"

# =========================
# 1️⃣ 获取TSA历史数据
# =========================

def get_tsa_history():
    url = "https://www.tsa.gov/travel/passenger-volumes"
    tables = pd.read_html(url)
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

    # 同比差值
    df["yoy_abs"] = df["current_year"] - df["last_year"]

    # 同比百分比
    df["yoy_pct"] = (df["current_year"] - df["last_year"]) / df["last_year"]

    df["month"] = df["date"].dt.month

    us_holidays = holidays.US()
    df["holiday"] = df["date"].apply(lambda x: 1 if x in us_holidays else 0)

    df = df.fillna(0)
    return df

# =========================
# 3️⃣ 训练模型
# =========================

def train_model(df):
    X = df[["weekday","ma7","ma30","trend","yoy_abs","yoy_pct","month","holiday"]]
    y = df["current_year"]

    model = LinearRegression()
    model.fit(X,y)

    residual = y - model.predict(X)
    std = np.std(residual)

    return model,std

# =========================
# 4️⃣ 滚动回测胜率
# =========================

def rolling_backtest(df,window=30):
    wins = 0
    total = 0

    for i in range(window,len(df)-1):
        train_df = df.iloc[i-window:i]
        test_df = df.iloc[i:i+1]

        model,_ = train_model(train_df)

        X_test = test_df[["weekday","ma7","ma30","trend","yoy_abs","yoy_pct","month","holiday"]]
        pred = model.predict(X_test)[0]
        actual = test_df["current_year"].values[0]

        # 方向预测是否正确
        prev = df.iloc[i-1]["current_year"]

        if (pred-prev)*(actual-prev) > 0:
            wins += 1

        total += 1

    if total == 0:
        return 0

    return round(wins/total,3)

# =========================
# 5️⃣ 次日预测
# =========================

def predict_next_day(df,model,std):
    last_row = df.iloc[-1]
    tomorrow = last_row["date"] + datetime.timedelta(days=1)

    weekday = tomorrow.weekday()
    ma7 = df["current_year"].tail(7).mean()
    ma30 = df["current_year"].tail(30).mean()
    trend = df["current_year"].iloc[-1] - df["current_year"].iloc[-2]

    # 去年同日
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

    X_new = np.array([[weekday,ma7,ma30,trend,yoy_abs,yoy_pct,month,holiday]])

    pred = model.predict(X_new)[0]
    lower = pred - 1.96*std
    upper = pred + 1.96*std

    return pred,lower,upper,ma7

# =========================
# 6️⃣ 仓位管理模型
# =========================

def position_size(pred,ma7,winrate):
    edge = abs(pred-ma7)/ma7

    # Kelly简化版本
    kelly = winrate - (1-winrate)

    size = kelly * edge

    # 限制仓位
    size = max(0,min(size,0.3))

    return round(size,3)

# =========================
# 7️⃣ 市场价格偏离判断
# =========================

def market_deviation(pred,market_price):
    deviation = (market_price - pred)/pred

    if deviation > 0.03:
        return "市场高估"
    elif deviation < -0.03:
        return "市场低估"
    else:
        return "价格合理"

# =========================
# 8️⃣ 主程序
# =========================

def main():

    df = get_tsa_history()
    df = prepare_data(df)

    model,std = train_model(df)

    winrate = rolling_backtest(df)

    pred,low,up,ma7 = predict_next_day(df,model,std)

    # 这里你可以替换为真实市场价格
    market_price = ma7  # 暂时假设市场价格=7日均线

    deviation_status = market_deviation(pred,market_price)

    size = position_size(pred,ma7,winrate)

    if pred > ma7:
        signal = "做多"
    elif pred < ma7:
        signal = "做空"
    else:
        signal = "观望"

    msg = f"""
📊 TSA量化预测系统

预测人数: {int(pred)}
预测区间: {int(low)} - {int(up)}

7日均值: {int(ma7)}
滚动胜率: {winrate}

市场状态: {deviation_status}
建议方向: {signal}
建议仓位: {size*100}%

"""

    requests.post(WECHAT_WEBHOOK,json={
        "msgtype":"text",
        "text":{"content":msg}
    })

if __name__ == "__main__":
    main()
