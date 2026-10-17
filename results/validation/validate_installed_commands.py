"""Exercise the installed wheel away from the source checkout."""
import importlib.metadata
import json,os,subprocess,sys,time
from pathlib import Path
import pandas as pd
import numpy as np
import src
from src.config import BASE_DIR
root=Path.cwd();tools=Path(sys.executable).parent
assert str(Path(src.__file__).resolve()).startswith('/workspace/scratch/eb86e47ab5e3/audit-wheel-env/')
assert BASE_DIR==root
results=[]
def command(name,args):
    start=time.perf_counter()
    run=subprocess.run([str(tools/name),*args],capture_output=True,text=True,timeout=240)
    (root/(name+'.log')).write_text(run.stdout+'\n'+run.stderr)
    results.append(dict(command=name,args=args,returncode=run.returncode,seconds=round(time.perf_counter()-start,3)))
    (root/'cli-validation.json').write_text(json.dumps(dict(installed_source=src.__file__,workspace=str(BASE_DIR),version=src.__version__,checks=results),indent=2))
    if run.returncode:raise RuntimeError(name+' failed: '+(run.stdout+run.stderr)[-4000:])
    print(name,'PASS',flush=True)
for name in ['fraud-demo','fraud-pipeline','fraud-study','fraud-audit','fraud-latency-audit','fraud-stats-rigor','fraud-latency','fraud-bandit']:
    command(name,['--help'])
command('fraud-demo',['--output-dir','example','--weeks','8','--transactions-per-week','100'])
path=root/'example/synthetic_paysim.csv'
raw=pd.read_csv(path)
ieee=pd.DataFrame(dict(TransactionID=np.arange(10000,10000+len(raw)),TransactionDT=(raw.step-1)*3600,
                       TransactionAmt=raw.amount,isFraud=raw.isFraud,ProductCD='W',
                       card1=1000+np.arange(len(raw))%19,card2=200+np.arange(len(raw))%3,
                       addr1=300+np.arange(len(raw))%5,C1=np.arange(len(raw))%7,
                       P_emaildomain=np.where(np.arange(len(raw))%2,'a.example','b.example')))
ieee.to_csv('synthetic_ieee.csv',index=False)
pd.DataFrame(dict(TransactionID=ieee.TransactionID,DeviceType=np.where(np.arange(len(raw))%2,'mobile','desktop'))).to_csv('synthetic_identity.csv',index=False)
command('fraud-pipeline',['--dataset','synthetic_ieee.csv','--identity','synthetic_identity.csv','--all-features','--history-features','--explain','--synthetic','--output-dir','feature_run'])
command('fraud-audit',['--dataset','synthetic_ieee.csv','--mode','both','--synthetic','--output-dir','cohort_run'])
command('fraud-latency-audit',['--dataset',str(path),'--output-dir','delay_run','--delays','0','2','--synthetic'])
command('fraud-study',['--dataset',str(path),'--output-dir','study','--seeds','42','101','--train-windows','2','5','--test-slices','2','--windows','2','--budgets','1','--delays','2','--synthetic'])
command('fraud-stats-rigor',['--benchmark','paysim','--paysim-dataset',str(path),'--output-dir','statistics','--seeds','42','101','--n-bootstrap','20','--synthetic'])
# Latency measurements require real-data readiness. A missing model must fail clearly.
(root/'features.json').write_text('{}')
run=subprocess.run([str(tools/'fraud-latency'),'--features','features.json','--requests','2'],capture_output=True,text=True)
assert run.returncode!=0 and 'Load a real-data model package' in run.stderr
print('fraud-latency readiness rejection PASS',flush=True)
print('COMPLETED',flush=True)
