def read_conll(in_file, lowercase=False, max_example=None):
    examples = []
    with open(in_file) as f:
        word, stem, pos, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    stem.append(sp[2].lower() if lowercase else sp[2])
                    pos.append(sp[3])
                    label.append(sp[7])
            elif len(word) > 0:
                examples.append({'word': word, 'stem': stem, 'pos': pos,  'label': label})
                word, stem, pos, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': word, 'stem': stem, 'pos': pos,  'label': label})
    return examples 
