import os
import pickle


class Data:

    def __init__(self, data_dir, reverse=False, tail_entity=None):
        self.train_data = self.load_data(data_dir, "train", reverse)
        print('size of train data', len(self.train_data))
        self.valid_data = self.load_data(data_dir, "valid", reverse, tail_entity)
        print('size of valid data', len(self.valid_data))
        self.test_data = self.load_data(data_dir, "test", reverse, tail_entity)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        print('train_relations')
        print(self.train_relations)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        print('test_relations')
        print(self.test_relations)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]
        print('relations')
        print(self.relations)
        
        with open('relations_str.txt', 'w') as filehandle:
            for listitem in self.relations:
                filehandle.write('%s\n' % listitem)
        #print(self.train_relations)

    def load_data(self, data_dir, data_type, reverse=False, tail_entity=None):
        with open('labeled_row_to_malware_mapping.pickle', 'rb') as handle:
            labeled_row_to_malware_mapping = pickle.load(handle)
        
        fp = os.path.join(data_dir, data_type+ ".txt")
        with open(fp, "r") as f:
            lines = f.read().strip().split("\n")  
            data = []
            for line in lines:
                if tail_entity is None:
                    data.append(line.split())
                else:
                    _, tail = labeled_row_to_malware_mapping[line]
                    if tail == tail_entity:
                        data.append(line.split())
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        #print('@@@@@@@@',(entities[0]))
        # len 2102 for MT3k
        with open('entities_str.txt', 'w') as filehandle:
            for listitem in entities:
                filehandle.write('%s\n' % listitem)
        return entities
