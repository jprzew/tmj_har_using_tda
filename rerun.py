import os
import pandas as pd
from tqdm import tqdm

RESULT_CSV = 'failure_log3.csv'

df = pd.read_csv(RESULT_CSV)
failed_df = df.query('State == "Failed"')

for index, row in tqdm(failed_df.iterrows(), total=failed_df.shape[0], desc='Rerunning failed experiments'):
    print(f'Rerunning experiment: {row["Experiment"]}')
    os.system(f'dvc exp apply {row["Experiment"]}')
    os.system('dvc exp run')