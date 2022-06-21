import pandas as pd



def process(file,outfile):
    source_data = {'sentence1': [], 'sentence2': [], 'label': []}
    with open(file,encoding='utf-8',mode='rt') as inp:
        line = inp.readline().strip()
        while line:
            samples = line.split('\t')
            if len(samples) == 3:
                source_data['sentence1'].append( samples[0])
                source_data['sentence2'] .append( samples[1])
                source_data['label'].append(samples[2])
            line = inp.readline().strip()
    df = pd.DataFrame(source_data)
    df.to_csv(outfile)



if __name__ == '__main__':
    process('./train.txt', './LCQMC_train.csv')
    process('./test.txt', './LCQMC_dev.csv')