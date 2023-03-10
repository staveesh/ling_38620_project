import json
from operator import le
import pandas as pd 
import numpy as np
import pathlib as pl 
import re
import string
import os
import ast
import json

input_path = 'data'
output_path = 'manipulated_data'

pl.Path(output_path).mkdir(parents=True, exist_ok=True)

def tokenize(sentence):
    """
    Returns words and punctuation symbols together. Question mark and full stop positions are also returned for preservation.
    """
    retain_pos = []
    stripped = []

    for x in re.split("(\W+)", sentence):
        stripped.append(x.strip())

    symbol_list = [item for item in stripped if len(item) > 0]
    for idx, symbol in enumerate(symbol_list):
        if symbol == '?' or symbol == '.':
            retain_pos.append(idx)

    return symbol_list, retain_pos

def shuffle_type_1(sentence):
    """
    Shuffles a sentence randomly. Retains the positions of question marks and full stops.
    """
    orig, pos = tokenize(sentence)
    words = np.delete(orig, pos)
    if len(words) <= 1 and len(pos) <= 1:
        return sentence
    word_indexes = list(range(len(words)))
    shuffled = list(range(len(words)))
    while True:
        np.random.shuffle(shuffled)
        done = True
        for i in range(len(word_indexes)):
            if shuffled[i] == word_indexes[i]:
                done = False
                break
        if done:
            break
    # Assemble into a new sentence
    i = 0
    ret_idx = 0
    res = ''
    while i < len(orig):
        if ret_idx < len(pos) and i == pos[ret_idx]:
            res += orig[pos[ret_idx]]
            ret_idx += 1
            i += 1
        if i - ret_idx < len(words):
            res += words[shuffled[i-ret_idx]]
        if i < len(shuffled) - 1:
            res += ' '
        i += 1
    return res

def shuffle_type_2(sentence):
    """
    Shuffles each bigram in a sentence. Retains the positions of question marks and full stops.
    """
    orig, pos = tokenize(sentence)
    words = np.delete(orig, pos)
    if len(words) <= 1 and len(pos) <= 1:
        return sentence
    bigram_pairs = []
    for i in range(0, len(words), 2):
        if i == len(words)-1:
            bigram_pairs.append((words[i], ''))
        else:
            bigram_pairs.append((words[i], words[i+1]))

    word_indexes = list(range(len(bigram_pairs)))
    shuffled = list(range(len(bigram_pairs)))

    if len(shuffled) == 1:
        return sentence

    while True:
        np.random.shuffle(shuffled)
        done = True
        for i in range(len(word_indexes)):
            if shuffled[i] == word_indexes[i]:
                done = False
                break
        if done:
            break
    # Assemble into a new sentence
    i = 0
    j = 0
    ret_idx = 0
    res = ''
    while i < len(shuffled):
        res += bigram_pairs[shuffled[i]][0]
        j += 1 
        if ret_idx < len(pos) and pos[ret_idx] == j:
            res += orig[pos[ret_idx]]
            ret_idx += 1
            j += 1
        res += ' '
        res += bigram_pairs[shuffled[i]][1]
        res += ' '
        j += 1
        i += 1
    return res


def shuffle_type_3(sentence):
    """
    Swaps words in each bigram in a sentence. No changes in positions of bigrams. Retains the positions of question marks and full stops.
    """
    orig, pos = tokenize(sentence)
    words = np.delete(orig, pos)
    if len(words) <= 1 and len(pos) <= 1:
        return sentence
    bigram_pairs = []
    for i in range(0, len(words), 2):
        if i == len(words)-1:
            bigram_pairs.append(('', words[i]))
        else:
            bigram_pairs.append((words[i+1], words[i]))

    shuffled = list(range(len(bigram_pairs)))

    if len(shuffled) == 1:
        return sentence

    # Assemble into a new sentence
    i = 0
    j = 0
    ret_idx = 0
    res = ''
    while i < len(shuffled):
        res += bigram_pairs[shuffled[i]][0]
        j += 1 
        if ret_idx < len(pos) and pos[ret_idx] == j:
            res += orig[pos[ret_idx]]
            ret_idx += 1
            j += 1
        res += ' '
        res += bigram_pairs[shuffled[i]][1]
        res += ' '
        j += 1
        i += 1
    return res

def shuffle_type_4(sentence):
    """
    Shuffles each trigram in a sentence. Retains the positions of question marks and full stops.
    """
    orig, pos = tokenize(sentence)
    words = np.delete(orig, pos)
    if len(words) <= 1 and len(pos) <= 1:
        return sentence
    bigram_pairs = []
    for i in range(0, len(words), 3):
        if i == len(words)-1:
            bigram_pairs.append((words[i], '', ''))
        elif i == len(words)-2:
            bigram_pairs.append((words[i], words[i+1], ''))
        else:
            bigram_pairs.append((words[i], words[i+1], words[i+2]))

    word_indexes = list(range(len(bigram_pairs)))
    shuffled = list(range(len(bigram_pairs)))

    if len(shuffled) == 1:
        return sentence

    while True:
        np.random.shuffle(shuffled)
        done = True
        for i in range(len(word_indexes)):
            if shuffled[i] == word_indexes[i]:
                done = False
                break
        if done:
            break
    # Assemble into a new sentence
    i = 0
    j = 0
    ret_idx = 0
    res = ''
    while i < len(shuffled):
        res += bigram_pairs[shuffled[i]][0]
        j += 1 
        if ret_idx < len(pos) and pos[ret_idx] == j:
            res += orig[pos[ret_idx]]
            ret_idx += 1
            j += 1
        res += ' '
        res += bigram_pairs[shuffled[i]][1]
        j += 1 
        if ret_idx < len(pos) and pos[ret_idx] == j:
            res += orig[pos[ret_idx]]
            ret_idx += 1
            j += 1 
        res += ' '
        res += bigram_pairs[shuffled[i]][2]
        j += 1 
        if ret_idx < len(pos) and pos[ret_idx] == j:
            res += orig[pos[ret_idx]]
            ret_idx += 1
            j += 1
        res += ' '
        i += 1
    return res

def shuffle_type_5(sentence):
    """
    Shuffles each trigram in a sentence. No changes in the positions of trigrams. Retains the positions of question marks and full stops.
    """
    orig, pos = tokenize(sentence)
    words = np.delete(orig, pos)
    if len(words) <= 1 and len(pos) <= 1:
        return sentence
    bigram_pairs = []
    for i in range(0, len(words), 3):
        if i == len(words)-1:
            bigram_pairs.append((words[i], '', ''))
        elif i == len(words)-2:
            bigram_pairs.append((words[i], words[i+1], ''))
        else:
            bigram_pairs.append((words[i], words[i+1], words[i+2]))
        l = list(bigram_pairs[-1])
        np.random.shuffle(l)
        bigram_pairs[-1] = tuple(l)

    shuffled = list(range(len(bigram_pairs)))
    
    if len(shuffled) == 1:
        return sentence

    # Assemble into a new sentence
    i = 0
    j = 0
    ret_idx = 0
    res = ''
    while i < len(shuffled):
        res += bigram_pairs[shuffled[i]][0]
        j += 1 
        if ret_idx < len(pos) and pos[ret_idx] == j:
            res += orig[pos[ret_idx]]
            ret_idx += 1
            j += 1
        res += ' '
        res += bigram_pairs[shuffled[i]][1]
        j += 1 
        if ret_idx < len(pos) and pos[ret_idx] == j:
            res += orig[pos[ret_idx]]
            ret_idx += 1
            j += 1 
        res += ' '
        res += bigram_pairs[shuffled[i]][2]
        j += 1 
        if ret_idx < len(pos) and pos[ret_idx] == j:
            res += orig[pos[ret_idx]]
            ret_idx += 1
            j += 1
        res += ' '
        i += 1
    return res

def shuffle_type_6(sentence):
    """
    Shuffles a sentence such that each bigram pair is out of order in the result. Retains the positions of question marks and full stops.
    """
    orig, pos = tokenize(sentence)
    words = np.delete(orig, pos)
    word_indexes = list(range(len(words)))
    # Start from mid, arrange alternately for length >= 5. 
    shuffled = [-1 for _ in words]
    if len(word_indexes) == 4:
        shuffled = [word_indexes[2], word_indexes[0], word_indexes[3], word_indexes[1]]
    else:
        last = None
        if len(words) % 2 != 0:
            n = len(word_indexes)
            last = word_indexes[n//2]
            word_indexes = word_indexes[:n//2] + word_indexes[n//2+1:]
        s = 0
        e = len(word_indexes)//2
        for i in range(len(word_indexes)):
            if i % 2 == 0:
                shuffled[i] = word_indexes[e]
                e += 1 
            else:
                shuffled[i] = word_indexes[s]
                s += 1
        if last is not None:
            shuffled[-1] = last
    # Assemble into a new sentence
    i = 0
    ret_idx = 0
    res = ''
    while i < len(orig):
        if ret_idx < len(pos) and i == pos[ret_idx]:
            res += orig[pos[ret_idx]]
            ret_idx += 1
            i += 1
        if i - ret_idx < len(words):
            res += words[shuffled[i-ret_idx]]
        if i < len(shuffled) - 1:
            res += ' '
        i += 1
    return res

def shuffle_type_7(sentence):
    """
    Shuffles a sentence such that global context is preserved but local is disturbed. Retains the positions of question marks and full stops.
    """
    orig, pos = tokenize(sentence)
    words = np.delete(orig, pos)
    word_indexes = list(range(len(words)))
    # Start from mid, arrange alternately for length >= 5. 
    shuffled = [-1 for _ in words]
    n = len(word_indexes)

    if len(word_indexes) == 4:
        shuffled = [word_indexes[2], word_indexes[0], word_indexes[3], word_indexes[1]] 
    else:
        last = None

        if len(words) % 4 == 1:
           last = [word_indexes[-1]]
           word_indexes = word_indexes[:-1]
        elif len(words) % 4 == 2:
            last = [word_indexes[-1], word_indexes[-2]]
            word_indexes = word_indexes[:-2]
        elif len(words) % 4 == 3:
            last = [word_indexes[-2], word_indexes[-3], word_indexes[-1]]
            word_indexes = word_indexes[:-3]

        for i in range(0, len(word_indexes)-3, 4):
            shuffled[i] = word_indexes[i+2]
            shuffled[i+1] = word_indexes[i]
            shuffled[i+2] = word_indexes[i+3]
            shuffled[i+3] = word_indexes[i+1]

        if last is not None:
            shuffled += last
    # Assemble into a new sentence
    i = 0
    ret_idx = 0
    res = ''
    while i < len(orig):
        if ret_idx < len(pos) and i == pos[ret_idx]:
            res += orig[pos[ret_idx]]
            ret_idx += 1
            i += 1
        if i - ret_idx < len(words):
            res += words[shuffled[i-ret_idx]]
        if i < len(shuffled) - 1:
            res += ' '
        i += 1
    return res

    
def jsonreader(f):
    res = []
    with open(f, 'r') as fd:
        for line in fd.readlines():
            d = ast.literal_eval(line)
            res.append(d)
    return res

def csvreader(f):
    df = pd.read_csv(f, sep='\t')
    df = df.dropna()
    return df

def process_input(shuffle_type, modify_sentence_1, modify_sentence_2):
    
    lim = 0
    if shuffle_type == 1:
        shuffler_fn = shuffle_type_1
    elif shuffle_type == 2:
        shuffler_fn = shuffle_type_2 
    elif shuffle_type == 3:
        shuffler_fn = shuffle_type_3 
    elif shuffle_type == 4:
        shuffler_fn = shuffle_type_4
    elif shuffle_type == 5:
        shuffler_fn = shuffle_type_5
    elif shuffle_type == 6:
        shuffler_fn = shuffle_type_6 
        lim = 3
    elif shuffle_type == 7:
        shuffler_fn = shuffle_type_7 
        lim = 3  
        
    if modify_sentence_1 and not modify_sentence_2:  
        mod_str = '_sen1'
    elif not modify_sentence_1 and modify_sentence_2:
        mod_str = '_sen2'

    for fol in os.listdir(input_path):
        pl.Path(f'{output_path}/{fol}/type_{shuffle_type}').mkdir(parents=True, exist_ok=True)
        print(f'Folder: {fol}...')
        for inp_file in os.listdir(f'{input_path}/{fol}'):
            if inp_file.endswith('.json'):

                new_file = inp_file[:-5]+'_type_'+str(shuffle_type)+ mod_str +'.json'
                with open(f'{output_path}/{fol}/type_{shuffle_type}/{new_file}', 'w') as fd:
                    corpus = jsonreader(f'{input_path}/{fol}/{inp_file}')
                    for line in corpus:
                        new_dict = dict(line)
                        if modify_sentence_1 and 'sentence1' in line:
                            if len(line['sentence1']) > lim:
                                manipulation = shuffler_fn(line['sentence1'])
                                new_dict['sentence1'] = manipulation
                        if modify_sentence_2 and 'sentence2' in line:
                            if len(line['sentence2']) > lim:
                                manipulation = shuffler_fn(line['sentence2'])
                                new_dict['sentence2'] = manipulation
                        if len(new_dict) > 0:
                            fd.write(str(json.dumps(new_dict))+'\n')
                print(f'Done with {inp_file}...')

            elif inp_file.endswith('.csv'):
                new_file = inp_file[:-4]+'_type_'+str(shuffle_type)+ mod_str +'.json'
                df = csvreader(f'{input_path}/{fol}/{inp_file}')
                sen1 = 'sentence1' in df.columns
                sen2 = 'sentence2' in df.columns 
                for idx, row in df.iterrows():
                    if sen1 and modify_sentence_1 and len(row['sentence1']) > lim:
                        df.at[idx, 'sentence1'] = shuffler_fn(row['sentence1'])
                    elif sen2 and modify_sentence_2 and len(row['sentence2']) > lim:
                        df.at[idx, 'sentence2'] = shuffler_fn(row['sentence2'])
                new_file = inp_file[:-4]+'_type_'+str(shuffle_type)+ mod_str +'.csv'
                cols = ['sentence1', 'sentence2', 'label'] if 'sentence2' in df.columns else ['sentence1', 'label']
                df[cols].to_csv(f'{output_path}/{fol}/type_{shuffle_type}/{new_file}', sep='\t', index=False)
                """
                new_file = inp_file[:-4]+'_type_'+str(shuffle_type)+ mod_str +'.json'
                with open(f'{output_path}/{fol}/type_{shuffle_type}/{new_file}', 'w') as fd:
                    fd.write(df.to_json(orient='records', lines=True))
                """
                print(f'Done with {inp_file}...')

if __name__ == "__main__":
    """
    print('Type 1')
    process_input(1, True, False)
    process_input(1, False, True)
    print('Type 2')
    process_input(2, True, False)
    process_input(2, False, True)
    print('Type 3')
    process_input(3, True, False)
    process_input(3, False, True)
    print('Type 4')   
    process_input(4, True, False)
    process_input(4, False, True)
    print('Type 5')
    process_input(5, True, False)
    process_input(5, False, True)
    print('Type 6')
    """
    process_input(7, True, False)
    # process_input(7, False, True)
