import gzip,json; f=gzip.open('data/runs.json.gz','rt'); data=json.load(f); r=data[0]; [print(k,':',v) for k,v in r.items() if not isinstance(v,dict)]
