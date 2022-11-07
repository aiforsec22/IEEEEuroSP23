from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import sys
import csv
from tqdm import tqdm
from find_ent_from_ID import get_data

np.set_printoptions(threshold=sys.maxsize)


class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.1):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}

        #print("learning rate = ", self.learning_rate, "\nentity dim = ", self.ent_vec_dim, "\nrelation dim = ",self.rel_vec_dim)
        #print("batch size = ", self.batch_size)

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate(self, model, data, target_relation_id, print_metrics_flag):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        # print("#"*20, "\n", test_data_idxs, "\n", "#"*20)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        #print("Number of data points: %d" % len(test_data_idxs))
        flag = False
        best_rank=1000 
        best_idx_j = -1 # idx of the triple which has the best rank???
        best_sorted_idx = torch.empty(2500) # sorted array of predictions
        best_sorted_score = torch.empty(2500) # sorted array of scoring values
        best_triple = []
        best_triple_list = []
        target_rel_all_ranks = []
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            # print(type(data_batch), data_batch.size())
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            # print(e1_idx, "@"*5, r_idx, "@"*5, e2_idx)
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            
            predictions = model.forward(e1_idx, r_idx)
            
            #print("#"*10, "Predcitions type", predictions.size())

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                # print("filt = ", filt)
                # .item() gives us the value at this position, just reading the tensor
                target_value = predictions[j, e2_idx[j]].item()

                # print("target value", target_value)
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value  # why assign a value which was previously there?

            '''
            x = torch.randn(3, 4)
            >>> x
            tensor([[ 0.8610, -2.1084,  0.5696,  0.8934],
                    [-1.6797, -1.2572, -1.7736, -0.0443],
                    [-0.3985,  1.7761,  0.5967,  0.2563]])
            >>> sorted, indices = torch.sort(x, 1, descending=True)
            >>> sorted
            tensor([[ 0.8934,  0.8610,  0.5696, -2.1084],
                    [-0.0443, -1.2572, -1.6797, -1.7736],
                    [ 1.7761,  0.5967,  0.2563, -0.3985]])
            >>> indices
            tensor([[3, 0, 2, 1],
                    [3, 1, 0, 2],
                    [1, 2, 3, 0]])
            '''
            # dim 1 means sorting the values in each row according to items in their column
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            # converting torch tensor t numpy ndarray

            sort_idxs = sort_idxs.cpu().numpy()
            # if flag == False:

            sort_values = sort_values.cpu().numpy()
            #print(np.shape(sort_values), np.shape(sort_idxs), '------')
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                #if j==0 and i==0:
                    #print('######',sort_idxs[j],' and ,', e2_idx[j], ' and rank = ', rank)
                    
                ranks.append(rank + 1)
                #print(ranks)
                #if r_idx[j].item()==target_relation_id:  
                if rank<=3:
                    target_rel_all_ranks.append(rank)                  
                    if rank < best_rank:
                        best_rank = rank
                        best_idx_j = j
                        best_sorted_idx = sort_idxs[best_idx_j]
                        best_sorted_score = sort_values[best_idx_j]
                        best_triple = [e1_idx[best_idx_j], e2_idx[best_idx_j], r_idx[best_idx_j]]
                    # best_triple_list.append([e1_idx[best_idx_j], e2_idx[best_idx_j], r_idx[best_idx_j]])
                for hits_level in range(10):  
                    
                          
                    if rank <= hits_level:
                        if flag == False:
                            #                            print('sort_idxs', sort_idxs)
                            # print("4th row - ", sort_idxs[4])
                            #print("looking for", e2_idx)
                            #print("np where", np.where(sort_idxs[j] == e2_idx[j].item()))
                            #print("found at ", rank)
                            flag = True
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
        if print_metrics_flag == True:
            print('\n\nHits @10: {0}'.format(np.mean(hits[9])))
            print('Hits @3: {0}'.format(np.mean(hits[2])))
            print('Hits @1: {0}'.format(np.mean(hits[0])))
            print('Mean rank: {0}'.format(np.mean(ranks)))
            print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
        return best_idx_j, best_rank, best_sorted_idx, best_sorted_score, best_triple, target_rel_all_ranks, best_triple_list







    def evaluate_final(self, model, data, target_relation_id):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        # print("#"*20, "\n", test_data_idxs, "\n", "#"*20)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        #print("Number of data points: %d" % len(test_data_idxs))
        flag = False
        best_rank=1000 
        best_idx_j = -1 # idx of the triple which has the best rank???
        best_sorted_idx = torch.empty(2500) # sorted array of predictions
        best_sorted_score = torch.empty(2500) # sorted array of scoring values
        best_triple = []
        best_triple_list = []
        target_rel_all_ranks = []
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            # print(type(data_batch), data_batch.size())
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            # print(e1_idx, "@"*5, r_idx, "@"*5, e2_idx)
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            
            predictions = model.forward(e1_idx, r_idx)
            
            #print("#"*10, "Predcitions type", predictions.size())

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                # print("filt = ", filt)
                # .item() gives us the value at this position, just reading the tensor
                target_value = predictions[j, e2_idx[j]].item()

                # print("target value", target_value)
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value  # why assign a value which was previously there?

            # dim 1 means sorting the values in each row according to items in their column
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            # converting torch tensor t numpy ndarray

            sort_idxs = sort_idxs.cpu().numpy()
            # if flag == False:

            sort_values = sort_values.cpu().numpy()
            #print(np.shape(sort_values), np.shape(sort_idxs), '------')
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                #if j==0 and i==0:
                    #print('######',sort_idxs[j],' and ,', e2_idx[j], ' and rank = ', rank)
                    
                ranks.append(rank + 1)
                #print(ranks)
                #if r_idx[j].item()==target_relation_id:  
                if rank<3:


                    # print here........
                    head = e1_idx[j].item()
                    tail = e2_idx[j].item()
                    get_data(head, tail, rank, sort_idxs[j], sort_values[j], self.entity_idxs_inv)



                    target_rel_all_ranks.append(rank)                  
                    if rank < best_rank:
                        best_rank = rank
                        best_idx_j = j
                        best_sorted_idx = sort_idxs[best_idx_j]
                        best_sorted_score = sort_values[best_idx_j]
                        best_triple = [e1_idx[best_idx_j], e2_idx[best_idx_j], r_idx[best_idx_j]]
                    best_triple_list.append([e1_idx[best_idx_j], e2_idx[best_idx_j], r_idx[best_idx_j]])
                for hits_level in range(10):  
                    
                          
                    if rank <= hits_level:
                        if flag == False:
                            #                            print('sort_idxs', sort_idxs)
                            # print("4th row - ", sort_idxs[4])
                            #print("looking for", e2_idx)
                            #print("np where", np.where(sort_idxs[j] == e2_idx[j].item()))
                            #print("found at ", rank)
                            flag = True
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)




    def train_and_eval(self, rel_name):
        #print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.entity_idxs_inv = {i: d.entities[i] for i in range(len(d.entities))}
        #print("&&&&&&&&&", type(self.entity_idxs), len(self.entity_idxs))
        
        '''w1 = csv.writer(open("/content/gdrive/MyDrive/Colab Notebooks/TuckER-master/entity_to_id_dict.csv", "w"))
        for key, val in self.entity_idxs.items():
            w1.writerow([key, val])'''
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}
        #print(self.relation_idxs)
        target_relation_id = self.relation_idxs[rel_name]
        print("Relation ", rel_name)
        
        '''w2 = csv.writer(open("/content/gdrive/MyDrive/Colab Notebooks/TuckER-master/relation_to_id_dict.csv", "w"))
        for key, val in self.relation_idxs.items():
            w2.writerow([key, val])'''
        
        train_data_idxs = self.get_data_idxs(d.train_data)
        #print("Number of training data points: %d" % len(train_data_idxs))

        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        min_rank = 1000
        for it in tqdm(range(1, self.num_iterations + 1)):
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0])
                r_idx = torch.tensor(data_batch[:, 1])
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                print("---------------------------------------->\n------------------------>", type(e1_idx), torch.Size(e1_idx))
                
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            #print(it, '-->')
            #print(time.time() - start_train)
            #print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                #print("Validation:")
                best_idx_j_valid, best_rank_valid, best_sorted_idx_valid, best_sorted_score_valid, best_triple_valid, target_rel_all_ranks_valid, _ = self.evaluate(model, d.valid_data, target_relation_id, False)
                if not it % 2:
                    #print("Test:")
                    start_test = time.time()
                    best_idx_j_test, best_rank_test, best_sorted_idx_test, best_sorted_score_test, best_triple_test, target_rel_all_ranks_test, _ = self.evaluate(model, d.test_data, target_relation_id, False)
                    if best_rank_test <= 3:    
                        head = best_triple_test[0].item()
                        tail = best_triple_test[1].item()
                        get_data(head, tail, best_rank_test, best_sorted_idx_test, best_sorted_score_test, self.entity_idxs_inv)
                    
                    #print(time.time() - start_test)
                    #if min(target_rel_all_ranks) < min_rank:
                     #   min_rank = min(target_rel_all_ranks)
                if it == self.num_iterations:

                    print('FINAL EVALUATION-----------')
                    self.evaluate_final(model, d.test_data, target_relation_id)


                    best_idx_j_test, best_rank_test, best_sorted_idx_test, best_sorted_score_test, best_triple_test, target_rel_all_ranks_test, best_triple_list = self.evaluate(model, d.test_data, target_relation_id, True)
                    #print('best_idx_j_test',best_idx_j_test, 'best_rank_test', best_rank_test, best_sort_test[:best_rank_test+10], best_triple_test)
                    #print(type(best_triple_test[0]), best_triple_test[0].item())
                    return best_idx_j_test, best_rank_test, best_sorted_idx_test, best_sorted_score_test, best_triple_test, target_rel_all_ranks_test, best_triple_list


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                        help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                        help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                        help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                        help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                        help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                        help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                        help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                        help="Amount of label smoothing.")
    parser.add_argument("--target_rel", type=str, default="uses", nargs="?",
                        help="Which relation to chose for prediction analysis?")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)

    
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir, reverse=False)
    
    #arr = ['attributedTo', 'attributedTo_reverse', 'authored', 'authored_reverse', 'belongsTo', 'belongsTo_reverse', 'connectsTo', 'connectsTo_reverse', 'exploits', 'exploits_reverse', 'hasAttachment', 'hasAttachment_reverse', 'hasAttackLocation', 'hasAttackLocation_reverse', 'hasAttackTime', 'hasAttackTime_reverse', 'hasAuthor', 'hasAuthor_reverse', 'hasCharacteristics', 'hasCharacteristics_reverse', 'hasFamily', 'hasFamily_reverse', 'hasMember', 'hasMember_reverse', 'hasProduct', 'hasProduct_reverse', 'hasType', 'hasType_reverse', 'hasVulnerability', 'hasVulnerability_reverse', 'indicates', 'indicates_reverse', 'involvesMalware', 'involvesMalware_reverse', 'mitigates', 'mitigates_reverse', 'targets', 'targets_reverse', 'usesAddress', 'usesAddress_reverse', 'usesDropper', 'usesDropper_reverse', 'usesMalware', 'usesMalware_reverse']
    #for i in range(2): 
        # this is an inefficient way to do this, because training and evaluating multiple times
        # ideal would be to store best of each relation in a data structure and retrieve values
    #target_rel = arr[i]
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr,
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1,
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing)
    # instead of args.target_rel, changed it to  variable for running code for all possible relations
    best_idx_j_test, best_rank_test, best_sorted_idx, best_sorted_score, best_triple_test, target_rel_all_ranks, best_triple_list = experiment.train_and_eval(args.target_rel)

    head = best_triple_test[0].item()
    tail = best_triple_test[1].item()

    # for i in range(min(500, len(best_triple_list))):
    #         head = best_triple_list[i][0].item()
    #         tail = best_triple_list[i][1].item()
    #         get_data(head, tail, best_rank_test, best_sorted_idx, best_sorted_score, experiment.entity_idxs_inv)

# ['attributedTo', 'attributedTo_reverse', 'authored', 'authored_reverse', 'belongsTo', 'belongsTo_reverse', 
# 'connectsTo', 'connectsTo_reverse', 
# 'exploits', ----> 122
#'exploits_reverse', 'hasAttachment', 'hasAttachment_reverse', 
# 'hasAttackLocation', 'hasAttackLocation_reverse', 
#'hasAttackTime', ------> 186 'hasAttackTime_reverse', 'hasAuthor', 'hasAuthor_reverse',
#  'hasCharacteristics', 'hasCharacteristics_reverse', 
# 'hasFamily', -----> no result found 'hasFamily_reverse', 
#'hasMember', -----> no result found, 'hasMember_reverse', 
# 'hasProduct', 'hasProduct_reverse', 'hasType', 'hasType_reverse', 'hasVulnerability', 'hasVulnerability_reverse', 
# 'indicates', 'indicates_reverse', 'involvesMalware', 'involvesMalware_reverse', 'mitigates', 'mitigates_reverse', 
# 'targets', 'targets_reverse', 'usesAddress', 'usesAddress_reverse', 
#'usesDropper',---> no result found 'usesDropper_reverse', 
# 'usesMalware', 'usesMalware_reverse']


