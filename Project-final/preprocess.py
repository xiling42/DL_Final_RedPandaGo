import numpy as np
import tensorflow as tf
import numpy as np

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
FRENCH_WINDOW_SIZE = 14
ENGLISH_WINDOW_SIZE = 14
##########DO NOT CHANGE#####################


def main():
    #train_labels = np.load('./reshape_labels_0_10000.npy')
    #train_data = np.load('./reshape_data_0_10000.npy')
    #print('label shape', train_labels.shape)
    train_data = np.load('./reshape_data_0_60000.npy')
    train_data_new = convert_to_ACGT(train_data[:70000])
    k = 4
    print("train_data_new: ",train_data_new)

    train_data_kmer = k_mers_block(k,train_data_new)
    print("train_data_kmer shape: ",train_data_kmer.shape) #(10000, 997, 4)

    id_dict = build_kmer_dict(k)
    print("id_dict: ",id_dict)
    train_data_id = convert_to_id(train_data_kmer,id_dict) ##(10000, 997)
    print(train_data_id)
    train_data_id = np.array(train_data_id)
    start = 0
    end = 60000
    data_name = "./data_id_"+str()+"_"+str(start)+"_"+str(end)

    np.save(data_name, train_data_id)



def getAllKLength(set, k): 
    
    n = len(set)  
    all_kmer = getAllKLengthRec(set, "", n, k) 
    all_kmer_np = np.array(all_kmer)
    return all_kmer_np.flatten()

def getAllKLengthRec(set, prefix, n, k): 
    # Base case: k is 0, 
    # print prefix 
    if (k == 0) :  
        return prefix
  
    # One by one add all characters  
    # from set and recursively  
    # call for k equals to k-1 
    all_list = []
    for i in range(n): 
        # Next character of input added 
        newPrefix = prefix + set[i] 
        # k is decreased, because  
        # we have added a new character 
        all_list.append(getAllKLengthRec(set, newPrefix, n, k - 1))
    return all_list

def convert_to_id(data,id_dict):

    #all_kmer = getAllKLength(["0","1","2","3","4"], k)
    #print(a)
    #convert each k-mer to an id can be treat as converting a 5-base number to decimal number
    
    #baseList = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    #id_dict = build_id_dict(k)

    #ten_digit = baseList[:10]
    #n_digit = baseList[:n]
    data_id = []
    for line in data:
        new_line = []
        for block in line:
            
            block_str = map(str, block.numpy())
            #print("block_str: ",block_str)
            block_str_concat = ''.join(block_str)
            #print("block_str_concat: ",block_str_concat)
            #break
            id = id_dict[block_str_concat]
            new_line.append(id)
        #break
        data_id.append(new_line)
    return data_id
    

def k_mers_block(k, data):
        outputs = []
        for line in data:
            #print(line)
            outputs.append(np.array([line[i:i+k] for i in range(0,len(line)-k+1)]))
        #print(outputs)
        return tf.convert_to_tensor(outputs)


def convert_to_ACGT(data):
    transformer = np.array([1,2,3,4])
    data = np.matmul(transformer,data)
    return data

def build_kmer_dict(k):
    all_kmer = getAllKLength(["0","1","2","3","4"], k)
    length = len(all_kmer)
    id_dict = {}

    for (kmer, i) in zip(all_kmer,range(length)):
        id_dict[kmer] = i
    
    return id_dict



if __name__ == '__main__':
	main()
