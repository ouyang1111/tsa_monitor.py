import requests
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import holidays

# ======================
# 填写你的API
# ======================

WEATHER_API_KEY = "填你的天气API"
AVIATION_API_KEY = "填你的航班API"
WECHAT_WEBHOOK = "填你的企业微信Webhook"

# ======================
# 1. 获取天气
# ======================

def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q=New York&appid={WEATHER_API_KEY}&units=metric"
    r = requests.get(url).json()
    weather_score = 0
    
    desc = r["weather"][0]["description"]
    wind = r["wind"]["speed"]
    
    if "rain" in desc:
        weather_score -= 1
    if "snow" in desc:
        weather_score -= 2
    if wind > 10:
        weather_score -= 1
        
    return weather_score, desc

# ======================
# 2. 获取航班信息
# ======================

def get_flight_data():
    url = f"http://api.aviationstack.com/v1/flights?access_key={AVIATION_API_KEY}"
    r = requests.get(url).json()
    
    total = 0
    delay = 0
    international = 0
    
    for f in r["data"]:
        total += 1
        
        if f["departure"]["delay"]:
            delay += 1
            
        if f["flight"]["iata"]:
            if len(f["flight"]["iata"]) > 4:
                international += 1
    
    delay_rate = delay / total if total else 0
    intl_ratio = international / total if total else 0
    
    return delay_rate, intl_ratio

# ======================
# 3. 节假日判断
# ======================

def is_holiday():
    us_holidays = holidays.US()
    today = datetime.date.today()
    return 1 if today in us_holidays else 0

# ======================
# 4. 模拟历史TSA数据
# ======================

def load_data():
    data = pd.DataFrame({
        "tsa":[2200000,2300000,2100000,2400000,2350000,2500000],
        "delay":[0.1,0.15,0.2,0.05,0.08,0.12],
        "weather":[0,-1,-2,0,0,-1],
        "intl":[0.2,0.25,0.3,0.18,0.22,0.27],
        "holiday":[0,0,1,0,0,1]
    })
    return data

# ======================
# 5. 训练模型
# ======================

def train_model():
    data = load_data()
    X = data[["delay","weather","intl","holiday"]]
    y = data["tsa"]
    
    model = LinearRegression()
    model.fit(X,y)
    
    residual = y - model.predict(X)
    std = np.std(residual)
    
    return model,std

# ======================
# 6. 预测
# ======================

def predict():
    delay, intl = get_flight_data()
    weather, desc = get_weather()
    holiday = is_holiday()
    
    model,std = train_model()
    
    X_new = np.array([[delay,weather,intl,holiday]])
    pred = model.predict(X_new)[0]
    
    lower = pred - 1.96*std
    upper = pred + 1.96*std
    
    return pred,lower,upper,desc,delay,intl,holiday

# ======================
# 7. 企业微信推送
# ======================

def send(msg):
    data = {
        "msgtype":"text",
        "text":{"content":msg}
    }
    requests.post(WECHAT_WEBHOOK,json=data)

# ======================
# 8. 主程序
# ======================

def main():
    pred,low,up,desc,delay,intl,holiday = predict()
    
    msg = f"""
📊 TSA预测报告 {datetime.date.today()}

天气: {desc}
延误率: {round(delay*100,2)}%
国际航班比例: {round(intl*100,2)}%
是否节假日: {holiday}

预测TSA: {int(pred)}
区间: {int(low)} - {int(up)}
"""
    send(msg)

if __name__=="__main__":
    main()
