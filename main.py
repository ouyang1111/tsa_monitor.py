import requests
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import holidays

WECHAT_WEBHOOK = "填你的webhook"

# ======================
# 1️⃣ 抓 TSA 官方历史数据
# ======================

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

# ======================
# 2️⃣ 构建特征
# ======================

def prepare_data(df):
    df["weekday"] = df["date"].dt.weekday
    df["ma7"] = df["current_year"].rolling(7).mean()
    df["ma30"] = df["current_year"].rolling(30).mean()
    df["trend"] = df["current_year"].diff()
    df["yoy"] = df["current_year"] - df["last_year"]
    df["season"] = df["date"].dt.month
    
    us_holidays = holidays.US()
    df["holiday"] = df["date"].apply(lambda x: 1 if x in us_holidays else 0)
    
    df = df.fillna(0)
    return df

# ======================
# 3️⃣ 训练模型
# ======================

def train_model(df):
    X = df[["weekday","ma7","ma30","trend","yoy","season","holiday"]]
    y = df["current_year"]
    
    model = LinearRegression()
    model.fit(X,y)
    
    residual = y - model.predict(X)
    std = np.std(residual)
    
    return model,std

# ======================
# 4️⃣ 预测次日
# ======================

def predict_next_day():
    df = get_tsa_history()
    df = prepare_data(df)
    
    model,std = train_model(df)
    
    last_row = df.iloc[-1]
    tomorrow = last_row["date"] + datetime.timedelta(days=1)
    
    weekday = tomorrow.weekday()
    ma7 = df["current_year"].tail(7).mean()
    ma30 = df["current_year"].tail(30).mean()
    trend = df["current_year"].iloc[-1] - df["current_year"].iloc[-2]
    
    # 去年同日
    last_year_same_day = df[df["date"] == (tomorrow - datetime.timedelta(days=365))]
    if not last_year_same_day.empty:
        yoy = df["current_year"].iloc[-1] - last_year_same_day["current_year"].values[0]
    else:
        yoy = 0
    
    season = tomorrow.month
    us_holidays = holidays.US()
    holiday = 1 if tomorrow in us_holidays else 0
    
    X_new = np.array([[weekday,ma7,ma30,trend,yoy,season,holiday]])
    
    pred = model.predict(X_new)[0]
    lower = pred - 1.96*std
    upper = pred + 1.96*std
    
    return pred,lower,upper,ma7

# ======================
# 5️⃣ 交易信号
# ======================

def trading_signal(pred,ma7):
    diff = (pred - ma7)/ma7
    
    if diff > 0.02:
        return "做多"
    elif diff < -0.02:
        return "做空"
    else:
        return "观望"

# ======================
# 6️⃣ 企业微信发送
# ======================

def send(msg):
    data = {
        "msgtype":"text",
        "text":{"content":msg}
    }
    requests.post(WECHAT_WEBHOOK,json=data)

# ======================
# 主程序
# ======================

def main():
    pred,low,up,ma7 = predict_next_day()
    signal = trading_signal(pred,ma7)
    
    msg = f"""
📊 TSA 次日预测系统

预测人数: {int(pred)}
预测区间: {int(low)} - {int(up)}

7日均值: {int(ma7)}
交易信号: {signal}
"""
    
    send(msg)

if __name__ == "__main__":
    main()
