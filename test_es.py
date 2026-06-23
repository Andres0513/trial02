import requests
import json

url = "http://116.62.87.215:39200/zj_bidding_ret/_search"
auth = ("andres", "andres")

query = {
    "size": 10000,
    "query": {
        "bool": {
            "must": [
                {
                    "range": {
                        "applied_time": {
                            "from": "2026-06-01T00:00:00.000+08:00",
                            "to": "2026-06-03T23:59:59.999+08:00",
                            "include_lower": True,
                            "include_upper": True,
                            "time_zone": "+08:00",
                            "boost": 1.0
                        }
                    }
                }
            ],
            "adjust_pure_negative": True,
            "boost": 1.0
        }
    },
    "sort": [
        {
            "applied_time": {
                "order": "asc"
            }
        }
    ],
    "track_total_hits": 2147483647
}

headers = {"Content-Type": "application/json"}
response = requests.get(url, auth=auth, headers=headers, data=json.dumps(query))

if response.status_code == 200:
    data = response.json()
    for hit in data['hits']['hits']:
        print(hit['_source'])
