import requests
import os
import pandas as pd

# 该文件与preprocess_knowledge.py 不同在于，其使用了Oxford 官方提供的无限制的访问API，但是unlimited plan 每进行一次hit，花费约是2分钱RMB，
# 若访问量过多则会很伤钱的
# 以下id和key已过期，停止使用，请自行申请无限量账密并替换
app_id = '0b1d83d8'
app_key = '0c70a826f51ef761a94e9e3bb006a532'

path = '/api/v2/search/en-gb'
entry_path = '/api/v2/entries/en-gb/'
urlOxfordApiProxy = 'https://od-api.oxforddictionaries.com'
methodRequestGet = 'GET'

def request_word_explanation(query_word):
    '''
    查询牛津字典对于词汇的解释
    :param query_word: 查询单词
    :return: 返回描述文本 {'status': False} or {'status': True, 'result': final_dict_knowledge}
    '''
    target_headers = {
  "Accept": "application/json",
  "app_id": app_id,
  "app_key": app_key
}
    r = requests.get(urlOxfordApiProxy+path, headers=target_headers,params={'q':query_word})
    if r.status_code != 200:
        substrs = query_word.split()
        if len(substrs) < 2:
            return {'status': False}
        result = []
        for str_ in substrs:
            res = request_word_explanation(str_)
            if res['status']:
                result.append(res['result'])
        return {'status': True, 'result': ' [SEP] '.join(result)}


    res = eval(r.text)  # matchString = 原词汇, id,word,label =fuzzy 匹配的词汇 , 优先使用word
    if 'results' not in res or len(res['results']) == 0:
        substrs = query_word.split()
        if len(substrs) < 2:
            return {'status': False}
        result = []
        for str_ in substrs:
            res = request_word_explanation(str_)
            if res['status']:
                result.append(res['result'])
        return {'status': True, 'result': ' [SEP] '.join(result)}
    infos = res['results'][0]
    keyRetry = infos['word'] or infos['id'] or infos['label']  # 用于查牛津词典的名词解释的中间单词
    r = requests.get(urlOxfordApiProxy+entry_path+keyRetry,headers={"app_id": app_id,"app_key": app_key})
    if r.status_code != 200:
        return {'status': False}
    res = eval(r.text)
    try:
        descriptions = []
        entries = res['results'][0]['lexicalEntries'][0]['entries']
        for entry in entries:  # 一个entry下所有解释
            description = ' '.join([' '.join(describe['definitions']) for describe in entry['senses']])  # 一个senses下所有解释
            descriptions.append(description)
    except KeyError:
        return {'status': False}
    final_dict_knowledge = ' '.join(descriptions)
    if final_dict_knowledge:
        return {'status': True, 'result': final_dict_knowledge}
    else:
        return {'status': False}




knowledge_file = './dictionary_knowledge.txt'
if not os.path.exists(knowledge_file):
    open(knowledge_file,'wt',encoding='utf-8').close()
knows = dict()
with open(knowledge_file,'rt',encoding='utf-8') as inp:
    line = inp.readline()
    while line:
        try:
            x,y = line.split('\t')
            knows[x] = y.strip()
        except:
            pass
        line = inp.readline()

for filename in os.listdir('.'):
    filepath = './' + filename
    if os.path.isdir(filepath):
        outdir = filepath + '/output_know'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for file in ['dev','test', 'train']:
            count = 0
            in_filepath = f'{filepath}/{file}.tsv'
            if not os.path.exists(in_filepath):
                print(f'无{in_filepath}.')
                continue
            out_filepath = f'{outdir}/{file}.tsv'
            data = []
            print(f'正在处理{in_filepath}...')
            with open(in_filepath,'rt',encoding='utf-8') as input, open(out_filepath, 'wt', encoding='utf-8') as outp, open(knowledge_file,'at',encoding='utf-8') as addp:
                line = input.readline()
                while line:
                    line = input.readline()
                    if not line:
                        break
                    try:
                        [sentence, aspect, polarity] = line.split('\t')
                    except:
                        [sentence, ter] = line.split('\t')
                        polarity = ter[-2:].strip()
                        aspect = ter[:-2].strip()
                    polarity = polarity.strip()
                    if aspect in knows:
                        add_know = knows[aspect]
                        count +=1
                    else:
                        response = request_word_explanation(aspect)
                        if not response['status']:
                            add_know = 'not found'
                        else:
                            add_know = response['result'][:600]
                        addp.write(aspect+'\t'+add_know)
                        addp.write('\n')
                        knows[aspect] = add_know
                        print(aspect,add_know)
                    sentence = sentence + ' [SEP] ' + add_know
                    data.append([sentence,aspect,polarity])
                df = pd.DataFrame(data, columns=['review', 'aspect', 'sentiment'], dtype=int)
                df.to_csv(out_filepath, sep='\t', index=False)
                print("处理数",count)