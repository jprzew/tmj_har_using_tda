import pandas as pd
import pickle
from celery.bin.result import result

DIAGRAMS1 = '../temp/first_diagrams_backup/diagrams.pkl'
DIAGRAMS2 = '../temp/second_diagrams_backup/diagrams.pkl'


diagrams1 = pd.read_pickle(DIAGRAMS1)
diagrams2 = pd.read_pickle(DIAGRAMS2)

result = {}
for key in diagrams1.keys():
    df1 = diagrams1[key]
    df2 = diagrams2[key]

    df = df1.combine_first(df2)
    result[key] = df

with open('../temp/merged_diagrams.pkl', 'wb') as f:
    pickle.dump(result, f)
