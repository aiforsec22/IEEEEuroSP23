from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import sys

np.set_printoptions(threshold=sys.maxsize)
    
class Experiment:

    def __init__(self, learning_rate, ent_vec_dim, rel_vec_dim, 
                 num_iterations, batch_size, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., outfile='result.txt'):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.outfile = outfile
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}

        print("learning rate = ", self.learning_rate, "\nentity dim = ", self.ent_vec_dim, "\nrelation dim = ", self.rel_vec_dim, "\n")
        print("batch size = ", self.batch_size)
        with open(self.outfile, 'a') as fout:
            line = "learning rate = "+ str(self.learning_rate)+"\nentity dim = ", str(self.ent_vec_dim)+"\nrelation dim = "+ str(self.rel_vec_dim)+"\nbatch size = "+ str(self.batch_size) +"\n"
            fout.writelines(line)


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
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    
    def evaluate(self, model, data, current_iteration, test_flag):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        #print("#"*20, "\n", test_data_idxs, "\n", "#"*20)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        #print("Number of data points: %d" % len(test_data_idxs))
        flag = False
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            #print(type(data_batch), data_batch.size())
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            #print(e1_idx, "@"*5, r_idx, "@"*5, e2_idx)
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
                target_value = predictions[j,e2_idx[j]].item()

                #print("target value", target_value)
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value # why assign a value which was previously there?


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
            #if flag == False:
                #with open('/content/gdrive/MyDrive/Colab Notebooks/TuckER-master/sorted_idxs.txt', 'w') as f:
                    #f.write(np.array2string(sort_idxs))
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]

                ranks.append(rank+1)

                for hits_level in range(10):


                    if rank <= hits_level:
                        if flag == False:
#                            print('sort_idxs', sort_idxs)
                            #print("4th row - ", sort_idxs[4])
                            #print("looking for", e2_idx)
                            #print("np where", np.where(sort_idxs[j]==e2_idx[j].item()))
                            #print("found at ", rank)
                            flag = True
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
        
        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))
        with open(self.outfile, 'a') as fout:
            line = 'Hits @10: {0}'.format(np.mean(hits[9]))+'\n'+'Hits @3: {0}'.format(np.mean(hits[2]))+ '\n'+ 'Hits @1: {0}'.format(np.mean(hits[0]))+ '\n'+ 'Mean rank: {0}'.format(np.mean(ranks))+'\n'+ 'Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks)))+'\n'
            fout.writelines(line)


    def train_and_eval(self):
        print("Training the TuckER model...")
        with open(self.outfile, 'a') as fout:
            fout.writelines("Training the TuckER model...\n")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        with open(self.outfile, 'a') as fout:
            line = "Number of training data points: " + str(len(train_data_idxs))+'\n'
            fout.writelines(line)

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
        with open(self.outfile, 'a') as fout:
            fout.writelines("Starting training...\n")
        for it in range(1, self.num_iterations+1):
            print("iteration: ", it, "/", self.num_iterations)
            with open(self.outfile, 'a') as fout:
                line = "iteration: " + str(it) + "/" + str(self.num_iterations) + "\n"
                fout.writelines(line)
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])  
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    print("-----------------line 196, main.py", type(e1_idx))
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            
            print(time.time()-start_train)    
            print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                print("Validation:")
                with open(self.outfile, 'a') as fout:
                    fout.writelines("Validation:\n")
                self.evaluate(model, d.valid_data, it, False)
                if not it%2:
                    print("Test:")
                    with open(self.outfile, 'a') as fout:
                        fout.writelines("Test:\n")
                    start_test = time.time()
                    self.evaluate(model, d.test_data, it, True)
                    #print(time.time()-start_test)
           

        

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

    args = parser.parse_args()
    dataset = args.dataset
    #data_dir = "data/%s/" % dataset
    data_dir = "/content/gdrive/My Drive/Colab Notebooks/TuckER-master/data/%s/"%dataset
    print(data_dir)
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True)
    
    batch_size_list = [128, 256]
    num_iter_list = [400, 500]
    learning_rate_list = [0.001, 0.0005]
    ent_embd_list = [200, 50]
    rel_embd_list = [200, 50]
    for batch_id in batch_size_list:
        for iter_id in num_iter_list:
            for ent_id in ent_embd_list:
                for rel_id in rel_embd_list:
                    for lr_id in learning_rate_list:
                        outfile_id = data_dir + 'trial_results/batch'+str(batch_id)+'iter'+str(iter_id)+'lr'+str(lr_id)+'ent'+str(ent_id)+'rel'+str(rel_id)+'.txt'
                        print(outfile_id)
                        experiment = Experiment(num_iterations=iter_id, batch_size=batch_id, learning_rate=lr_id, 
                                        decay_rate=args.dr, ent_vec_dim=ent_id, rel_vec_dim=rel_id, cuda=args.cuda,
                                        input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                                        hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing, outfile=outfile_id)
                        experiment.train_and_eval()
                        print("Finished printing to", outfile_id)
                        with open(outfile_id, 'a') as fout:
                            fout.writelines("========================")


    '''experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing)
'''
            

                

