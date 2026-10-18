"""Generate a declared client-shift classification task without external data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def client_data(seed, client, split, samples=128, features=20):
    """Stable causal label mechanism and client-specific spurious measurements."""
    if samples < 4 or features < 2:
        raise ValueError("Use at least four rows and two features per client")
    streams=np.random.SeedSequence([seed,client]).spawn(3)
    signal=np.random.default_rng(streams[0]);nuisance=np.random.default_rng(streams[1])
    environment=np.random.default_rng(streams[2])
    shift=environment.normal(0,.5,2)
    causal=signal.normal(size=(samples,2))+shift
    label=(1.5*causal[:,0]-causal[:,1]+signal.normal(0,.25,samples)>0).astype(int)
    correlation=environment.uniform(.6,1.2)*(1 if split=='train' else -1)
    other=nuisance.normal(size=(samples,features-2))
    if features>2:
        other+=correlation*(2*label[:,None]-1)
        other+=environment.normal(0,.5,features-2)
    values=np.column_stack([causal,other])
    result=pd.DataFrame(values,columns=[f'x{j:02d}' for j in range(features)])
    result.insert(0,'label',label);result.insert(0,'split',split)
    result.insert(0,'client',f'client_{client:03d}')
    result['malicious']=False;result['extreme']=bool(np.linalg.norm(shift)>.9)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='data/generated_clients.csv')
    parser.add_argument('--train-clients',type=int,default=12)
    parser.add_argument('--test-clients',type=int,default=4)
    parser.add_argument('--samples-per-client',type=int,default=128)
    parser.add_argument('--features',type=int,default=20)
    parser.add_argument('--seed',type=int,default=13)
    args=parser.parse_args()
    if args.train_clients<3 or args.test_clients<1:
        parser.error('Need at least three training clients and one test client')
    rows=[client_data(args.seed,i,'train' if i<args.train_clients else 'test',
                      args.samples_per_client,args.features)
          for i in range(args.train_clients+args.test_clients)]
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    pd.concat(rows,ignore_index=True).to_csv(target,index=False)
    setting={'source':'analytic_client_shift','seed':args.seed,
             'train_clients':args.train_clients,'test_clients':args.test_clients,
             'samples_per_client':args.samples_per_client,'features':args.features,
             'causal_features':['x00','x01'],
             'label_mechanism':'1.5*x00-x01+independent N(0,0.25^2)>0',
             'nuisance_mechanism':'class-correlated Gaussian measurements with test correlation reversal',
             'roles':'whole clients are disjoint; no test data select parameters',
             'measurements':'generated task inputs, not measured client records'}
    target.with_suffix('.json').write_text(json.dumps(setting,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
