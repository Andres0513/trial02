import requests

url = "https://day-aheediction-sijbvximde.cn-hangzhou.fcapp.run"
data = {"input_data": "测试数据"}


res = requests.post(url, json=data)
print("状态码：", res.status_code)
print("原始返回：", res.text)
# 直接解析JSON
result = res.json()
print("解析后的msg：", result["msg"])