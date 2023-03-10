import pandas as pd 
import os 
import ast 

for ds in os.listdir('data'):
    for t in os.listdir(f'data/{ds}'):
        if t.endswith('.csv'):
            print(f'data/{ds}/{t}')
            df = pd.read_csv(f'data/{ds}/{t}', sep='\t')
            df['s1_len'] = df['sentence1'].apply(lambda x: len(x.split()))
            l1 = len(df[df['s1_len'] <= 3])
            l2 = 0
            if 'sentence2' in df.columns:
                df['s2_len'] = df['sentence2'].apply(lambda x: len(x.split()))
                l2 = len(df[df['s2_len'] <= 3])
            print('s1', l1, len(df), round(100*l1/len(df), 2), '%')
            print('s2', l2, len(df), round(100*l2/len(df), 2), '%')
        elif t.endswith('.json'):
            print(f'data/{ds}/{t}')
            s1 = 0
            s2 = 0
            c = 0
            with open(f'data/{ds}/{t}', 'r') as fd:
                for line in fd.readlines():
                    data = ast.literal_eval(line)
                    if 'sentence1' in data and len(data['sentence1']) <= 3:
                        s1 += 1
                    if 'sentence2' in data and len(data['sentence2']) <= 3:
                        s2 += 1 
                    c += 1 
            print('s1', s1, c, round(s1/c*100, 2), '%')
            print('s2', s2, c, round(s2/c*100, 2), '%')
