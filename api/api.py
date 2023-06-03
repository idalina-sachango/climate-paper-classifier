import requests
import json
import pandas as pd
import time
import time
import sys

js = []

st = time.process_time()
while len(js) < 5000:
    for num in range(50):
        r = requests.get(f"https://www.osti.gov/api/v1/records?q=54&page={num}")

        records = r.json()
        kept = []
        ost_id = set()
        for re in records:
            if "subjects" in re.keys():
                if "54 environmental sciences" in [x.lower() for x in re["subjects"]]:
                    if re["osti_id"] not in ost_id:
                        kept.append(re)
                        ost_id.add(re["osti_id"])

        js.extend(kept)
    time.sleep(10)
    print(len(js))

df = pd.DataFrame(js)
df = pd.DataFrame.from_dict(js)
df = pd.DataFrame.from_records(js)
df.to_csv(f"{sys.argv[1]}.csv")


# get execution time
et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')