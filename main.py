import requests
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import holidays

# ======================
# 填你的企业微信Webhook
# ======================

WECHAT_WEBHOOK = "填你的webhook地址"

TOP_AIRPORTS = ["ATL","LAX","ORD","DFW","DEN","JFK","LAS","SEA","MCO","CLT"]

# ======================
# 1️⃣ 抓OpenSky数据
# ======================

def get_opensky_data():
    url = "https://opensky-network.org/api/states/all"
    r = requests.get(url)
    data = r.json()
    
    total = len(data["states"])
    us_count = 0
    
    for s in data["states"]:
        if s[2] and "US" in s[2]:
            us_count += 1
            
    ratio = us_count / total if total else 0
    
    return total, ratio

# ======================
# 2️⃣ 抓天气
# ======================

def get_weather_score():
    score = 0
    
    for airport in TOP_AIRPORTS:
        url = f"https://aviationweather.gov/api/data/metar?ids={airport}&format=json"
        r = requests.get(url)
        data = r.json()
        
        if not data:
            continue
        
        raw = data[0].get("rawOb","")
        
        if "RA" in raw:
            score -= 1
        if "SN" in raw:
            score -= 2
        if "TS" in raw:
            score -= 1
            
    return score

# ======================
# 3️⃣ 节假日
# ======================

def is_holiday(date):
    us_holidays = holidays.US()
    return 1 if date in us_holidays else 0

# ======================
# 4️⃣ 模拟历史数据（你以后可替换真实TSA）
# ======================

def load_data():
    df = pd.DataFrame({
        "tsa":[2200000,2300000,2100000,2400000,2350000,2500000,2450000,2550000,2600000,2500000],
        "flight":[30000,32000,28000,35000,34000,36000,35500,37000,38000,36000],
        "weather":[0,-2,-1,0,-3,0,-1,-2,0,-1],
        "holiday":[0,0,1,0,0,1,0,0,0,0],
        "weekday":[1,2,3,4,5,6,7,1,2,3]
    })
    
    df["ma7"] = df["tsa"].rolling(7).mean()
    df["trend"] = df["tsa"].diff()
    df["season"] = df.index % 12
    
    df = df.fillna(0)
    
    return df

# ======================
# 5️⃣ 训练模型
# ======================

def train_model():
    df = load_data()
    
    X = df[["flight","weather","holiday","weekday","ma7","trend","season"]]
    y = df["tsa"]
    
    model = LinearRegression()
    model.fit(X,y)
    
    residual = y - model.predict(X)
    std = np.std(residual)
    
    return model,std,df

# ======================
# 6️⃣ 预测次日
# ======================

def predict_next_day():
    model,std,df = train_model()
    
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    
    flights,_ = get_opensky_data()
    weather = get_weather_score()
    holiday = is_holiday(tomorrow)
    weekday = tomorrow.weekday()+1
    
    ma7 = df["tsa"].tail(7).mean()
    trend = df["tsa"].iloc[-1] - df["tsa"].iloc[-2]
    season = tomorrow.month
    
    X_new = np.array([[flights,weather,holiday,weekday,ma7,trend,season]])
    
    pred = model.predict(X_new)[0]
    lower = pred - 1.96*std
    upper = pred + 1.96*std
    
    return pred,lower,upper,ma7

# ======================
# 7️⃣ 交易信号
# ======================

def trading_signal(pred,ma7):
    diff = (pred - ma7)/ma7
    
    if diff > 0.03:
        return "做多（高于趋势）"
    elif diff < -0.03:
        return "做空（低于趋势）"
    else:
        return "观望"

# ======================
# 8️⃣ 回测
# ======================

def backtest():
    model,std,df = train_model()
    
    X = df[["flight","weather","holiday","weekday","ma7","trend","season"]]
    preds = model.predict(X)
    
    error = np.mean(abs(preds - df["tsa"]))
    
    return int(error)

# ======================
# 9️⃣ 企业微信
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
    error = backtest()
    
    msg = f"""
📊 TSA交易系统

预测次日人数: {int(pred)}
区间: {int(low)} - {int(up)}

7日均值: {int(ma7)}

交易信号: {signal}

模型平均误差: {error}
"""
    send(msg)

if __name__ == "__main__":
    main()
