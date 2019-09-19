import json,numpy as np
class Preprocess(object):
    def __init__(self,data_set_path='data/train_3.txt',
                 dict_csk_path='preprocess/dict_dedup',
                 stopword_path='preprocess/stopwords',
                 split_ratio=0.1):
        self.data_set_path=data_set_path
        self.dict_csk_path=dict_csk_path
        self.stopword_path=stopword_path
        self.split_ratio=split_ratio
        self.init_load()
    def init_load(self):
        """
        resource的 keys 是'csk_entities', 'dict_csk_entities', 'dict_csk',
        'vocab_dict', 'csk_triples', 'dict_csk_triples'
        :return:
        """

        with open(self.dict_csk_path,'r') as f:
            self.dict_csk=json.load(f)
        with open(self.stopword_path,'r') as f:
            self.stopword=json.load(f)
        self.csk_triples = []
        self.csk_entities = []
        self.dict_csk_triples = {}
        self.dict_csk_entities = {}
        for entity in self.dict_csk:
            for triple in self.dict_csk[entity]:
                self.csk_triples.append(triple)
                hrt=triple.split(', ')
                self.csk_entities+=[hrt[0],hrt[-1]]
        self.csk_triples=list(set(self.csk_triples))
        self.csk_entities=list(set(self.csk_entities))
        for idx,triple in enumerate(self.csk_triples):
            self.dict_csk_triples[triple]=idx
        for idx,entity in enumerate(self.csk_entities):
            self.dict_csk_entities[entity]=idx
    def mapping_between_csk_dialogue(self,post,response):
        try:
            list_of_entities = []
            post_triple_dict = {}
            post_triples = []
            response_triples = []
            response_triple_dict = {}
            match_triples = []
            list_of_triples = []
            match_index = []
            for word in post:
                if self.stopword.get(word, -1) == -1:
                    if self.dict_csk.get(word, -1) != -1:
                        if word not in post_triple_dict:
                            post_triple_dict[word] = len(list_of_triples) + 1
                            entity = []
                            for idx, triple in enumerate(self.dict_csk[word]):
                                hrt = triple.split(', ')
                                if hrt[0] == word:
                                    entity.append(hrt[2])
                                    if hrt[2] in response:
                                        match_triples.append(triple)
                                        response_triple_dict[hrt[2]] = [len(list_of_triples) + 1, idx, triple]
                                elif hrt[2] == word:
                                    entity.append(hrt[0])
                                    if hrt[0] in response:
                                        match_triples.append(triple)
                                        response_triple_dict[hrt[0]] = [len(list_of_triples) + 1, idx, triple]
                            list_of_entities.append(entity)
                            list_of_triples.append(self.dict_csk[word])

            for word in post:
                if word in post_triple_dict:
                    post_triples.append(post_triple_dict[word])
                else:
                    post_triples.append(0)

            for word in response:
                if word in response_triple_dict:
                    response_triples.append(response_triple_dict[word][2])
                    match_index.append(response_triple_dict[word][:2])
                else:
                    response_triples.append(-1)
                    match_index.append([-1, -1])

            match_triples = list(set(match_triples))
            return list_of_entities, list_of_triples, match_triples, post_triples, response_triples, match_index
        except Exception as e:
            print(e)
            return 'An error occured. Discard this training example'
    def generate_file(self):
        if self.data_set_path.endswith('train_3.txt'):
            """
            preprocess our own dataset
            """
            with open(self.data_set_path,'r',encoding='utf-8') as f:
                samples=np.array(f.readlines())
                np.random.shuffle(samples)
                val_num=int(len(samples)*self.split_ratio)
                trainset=[]
                valset=[]
                testset=[]
                count=0
                vocab={}
                for idx,sample in enumerate(samples):
                    temp=sample.split('\t')
                    d={}
                    context=temp[0].split()
                    response=temp[1].split()
                    entities, triples, match_triples, post_triples, response_triples, match_index=self.mapping_between_csk_dialogue(context,response)
                    if len(match_triples)==0:
                        continue
                    count+=1
                    d['post'] = context
                    d['response'] = response
                    for word in context+response:
                        if word in vocab:
                            vocab[word]+=1
                        else:
                            vocab[word]=1
                    d['match_triples'] = [self.dict_csk_triples[m] for m in match_triples]
                    d['all_triples'] = [[self.dict_csk_triples[m] for m in tri] for tri in triples]
                    d['all_entities'] = [[self.dict_csk_entities[m] for m in ent] for ent in entities]
                    d['post_triples'] = post_triples
                    d['response_triples'] = [-1 if m == -1 else self.dict_csk_triples[m] for m in response_triples]
                    d['match_index'] = match_index
                    if count<=100:
                        testset.append(json.dumps(d))
                    elif count<=100+val_num:
                        valset.append(json.dumps(d))
                    else:
                        trainset.append(json.dumps(d))
                print('valnum {}'.format(val_num))
                print("match number>0 {}".format(count))
                print("vocab num {}".format(len(vocab)))
                print('train number:{}, val num:{} test number:{}'.format(len(trainset),len(valset),len(testset)))
                source={'dict_csk':self.dict_csk,'csk_entities':self.csk_entities,'csk_triples':self.csk_triples,
                        'dict_csk_entities':self.dict_csk_entities,'dict_csk_triples':self.dict_csk_triples,
                        'vocab_dict':vocab}
                with open('data/source.json','w') as f:
                    json.dump(source,f)
                with open('data/trainset.txt','w') as f:
                    f.write('\n'.join(trainset))
                with open('data/valset.txt','w') as f:
                    f.write('\n'.join(valset))
                with open('data/testset.txt','w') as f:
                    f.write('\n'.join(testset))

        else:
            raise ValueError("We haven't implemented the preprocess function of this dataset!")

p=Preprocess()
p.generate_file()

