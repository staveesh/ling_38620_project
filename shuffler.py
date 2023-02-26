import json
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
    if len(words) == 1 and len(pos) <= 1:
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
    if len(words) == 1 and len(pos) <= 1:
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
    if len(words) == 1 and len(pos) <= 1:
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

def jsonreader(f):
    res = []
    with open(f, 'r') as fd:
        for line in fd.readlines():
            d = ast.literal_eval(line)
            res.append(d)
    return res

def csvreader(f):
    return pd.read_csv(f, sep='\t')

def process_input(shuffle_type, modify_sentence_1, modify_sentence_2):

    if shuffle_type == 1:
        shuffler_fn = shuffle_type_1
    elif shuffle_type == 2:
        shuffler_fn = shuffle_type_2 
    elif shuffle_type == 3:
        shuffler_fn = shuffle_type_3 
    
    if modify_sentence_1 and not modify_sentence_2:
        mod_str = '_sen1'
    elif not modify_sentence_1 and modify_sentence_2:
        mod_str = '_sen2'

    for fol in os.listdir(input_path):
        pl.Path(f'{output_path}/{fol}').mkdir(parents=True, exist_ok=True)
        print(f'Folder: {fol}...')
        for inp_file in os.listdir(f'{input_path}/{fol}'):
            if inp_file.endswith('.json'):

                new_file = inp_file[:-5]+'_type_'+str(shuffle_type)+ mod_str +'.json'
                with open(f'{output_path}/{fol}/{new_file}', 'w') as fd:
                    corpus = jsonreader(f'{input_path}/{fol}/{inp_file}')
                    for line in corpus:
                        new_dict = dict(line)
                        if modify_sentence_1 and 'sentence1' in line:
                            manipulation = shuffler_fn(line['sentence1'])
                            new_dict['sentence1'] = manipulation
                        if modify_sentence_2 and 'sentence2' in line:
                            manipulation = shuffler_fn(line['sentence2'])
                            new_dict['sentence2'] = manipulation
                        fd.write(str(json.dumps(new_dict))+'\n')
                print(f'Done with {inp_file}...')

            elif inp_file.endswith('.csv'):
                new_file = inp_file[:-4]+'_type_'+str(shuffle_type)+ mod_str +'.json'
                df = csvreader(f'{input_path}/{fol}/{inp_file}')
                sen1 = 'sentence1' in df.columns
                sen2 = 'sentence2' in df.columns 
                for idx, row in df.iterrows():
                    if sen1 and modify_sentence_1:
                        df.at[idx, 'sentence1'] = shuffler_fn(row['sentence1'])
                    elif sen2 and modify_sentence_2:
                        df.at[idx, 'sentence2'] = shuffler_fn(row['sentence2'])
                new_file = inp_file[:-4]+'_type_'+str(shuffle_type)+ mod_str +'.csv'
                df.to_csv(f'{output_path}/{fol}/{new_file}', sep='\t')
                new_file = inp_file[:-4]+'_type_'+str(shuffle_type)+ mod_str +'.json'
                with open(f'{output_path}/{fol}/{new_file}', 'w') as fd:
                    fd.write(df.to_json(orient='records', lines=True))

if __name__ == "__main__":
    process_input(1, True, False)
    process_input(1, False, True)
    process_input(2, True, False)
    process_input(2, False, True)
    process_input(3, True, False)
    process_input(3, False, True)
