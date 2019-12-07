import tensorflow as tf
import numpy as np
import transformer_func as transformer
#import multi_head_func as transformer

tf.keras.backend.set_floatx('float64')

class CNN_Attention(tf.keras.Model):

    def __init__(self):
     
        super(CNN_Attention, self).__init__()


        self.k = 4
        self.embedding_size = 20
        self.batch_size = 100
        self.conv_kernel_size = 8
        self.pool_kernel_size = 4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.dna_length = 997
        
        self.nkernels = [320,480,960]
        sequence_length =  1000
        n_genomic_features  = 919
        
        self.num_E = np.power(5,self.k)

        self.E =  tf.Variable(tf.random.normal(shape=[self.num_E,self.embedding_size], stddev=.1, dtype=tf.float32))
        

        #Dense layer (output 919)  
        # remove data_format='channels_first',

        self.pos_encoder = transformer.Position_Encoding_Layer(self.dna_length,self.embedding_size)
        self.encoder = transformer.Transformer_Block(self.embedding_size,is_decoder=False,multi_headed=True)

        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=self.nkernels[0],kernel_size = self.conv_kernel_size,  padding='valid',activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool1D(pool_size=self.pool_kernel_size,strides=self.pool_kernel_size, padding='valid')

        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=self.nkernels[1], kernel_size = self.conv_kernel_size, padding='valid',activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool1D(pool_size=self.pool_kernel_size, strides=self.pool_kernel_size, padding='valid')
        
        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=self.nkernels[2], kernel_size = self.conv_kernel_size, padding='valid',activation='relu')

        self.dense1 = tf.keras.layers.Dense(n_genomic_features, activation='relu')

        self.dense2 = tf.keras.layers.Dense(n_genomic_features, activation='sigmoid')   

         
        
        #params = self.parameters()
        

    def call(self, inputs):
        

        #inputs = tf.transpose(inputs, perm=[0, 2, 1]) 
        print("inputs.shape: ",inputs.shape)

        #k-mer
        #input_kmer = self.k_mers_block(self.k,inputs)
        encoder_input_embeddings = tf.nn.embedding_lookup(self.E,inputs)
        pos_encoder_output = self.pos_encoder(encoder_input_embeddings)
        encoder_output = self.encoder(pos_encoder_output)

        #print("input kmer shape: ",input_kmer.shape)
        out = self.cnn_layer1(encoder_output)
        #print(out)
        #print("out.shape cnn1: ",out.shape)
        #out = tf.nn.max_pool(out,ksize=self.pool_kernel_size, strides=self.pool_kernel_size, padding ='VALID')
        out = self.maxpool1(out)
        #print('max output size: ', out.shape)
        out = tf.nn.dropout(out,0.2)
        #print("out.shape1: ",out.shape)
        #print(out)
        
        out = self.cnn_layer2(out)
        #print("out.shape cnn2: ",out.shape)
        out = self.maxpool2(out)
        #out = tf.nn.max_pool(out,ksize=self.pool_kernel_size, strides=self.pool_kernel_size, padding ='VALID')
        out = tf.nn.dropout(out,0.2)
        #print("out.shape2: ",out.shape)
        
        out = self.cnn_layer3(out)
        out = tf.nn.dropout(out,0.5)
        
        #print("out.shape3: ",out.shape)
        #out = tf.keras.layers.Flatten(out)
        #out.set_shape([out.shape[0], self.nkernels[2] * self.n_channels])
        out = tf.reshape(out,[out.shape[0], -1])
        #reshape_out = tf.reshape(out,[-1, self.nkernels[2] * self.n_channels])
        #print("reshape size: ", out.shape)
        out = self.dense1(out)
        #predict = tf.keras.backend.sigmoid(out)
        print("dense1_out: ",out)
        predict = self.dense2(out)
        print("dense2_out: ",out)
        #print('final output size: ', predict.shape)
        return predict
    
    def loss(self, logits, labels):
        #print('loss label shape', labels.shape)
        tf.keras.losses.BinaryCrossentropy()
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))
        return loss
    
    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(logits, labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    


def train(model, train_inputs, train_labels):
    loss = tf.keras.losses.BinaryCrossentropy()
    for i in range(int(train_inputs.shape[0]/model.batch_size)):
        train_x_batch = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        train_y_batch = train_labels[i*model.batch_size:(i+1)*model.batch_size]
        train_y_batch = train_y_batch.astype(float)
        
        with tf.GradientTape() as tape:
            logits = model.call(train_x_batch.astype(float))
            l = loss(y_pred = logits, y_true = tf.convert_to_tensor(train_y_batch))
        gradients = tape.gradient(l, model.trainable_variables)
        print("loss: ", i, ": ", l)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
def test(model, test_inputs, test_labels):
    total_accuracy = 0
    for i in range(int(test_inputs.shape[0]/model.batch_size)):
        test_x_batch = test_inputs[i*model.batch_size:(i+1)*model.batch_size]
        test_y_batch = test_labels[i*model.batch_size:(i+1)*model.batch_size]
        test_y_batch = test_y_batch.astype(float)
        
        logits = model.call(test_x_batch.astype(float))
        accuracy = model.accuracy(logits, test_y_batch)
        total_accuracy += accuracy
        print("accuracy loop ", i, ": ", accuracy)
    return total_accuracy / (int(test_inputs.shape[0]/model.batch_size))

def baseconvert(number, fromdigits, todigits):
    if str(number)[0] == '-':
        number = str(number)[1:]
        neg = 1
    else:
        neg = 0
    x = 0
    for digit in str(number):
        x = x*len(fromdigits) + fromdigits.index(digit)
    res = ""
    while x > 0:
        digit = x % len(todigits)
        res = todigits[digit] + res
        x //= len(todigits)
    if neg:
        res = "-"+res
    return res
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

def main():
    # train label and data
    train_labels = np.load('./reshape_labels_0_10000.npy')
    train_data = np.load('./data_id_0_10000.npy')
    print('label shape', train_labels.shape)
    
    #========Preprocess=========
    '''
    train_data_new = convert_to_ACGT(train_data)
    k = 4
    print("train_data_new: ",train_data_new)

    train_data_kmer = k_mers_block(k,train_data_new)
    print("train_data_kmer shape: ",train_data_kmer.shape) #(10000, 997, 4)

    id_dict = build_kmer_dict(k)
    print("id_dict: ",id_dict)
    train_data_id = convert_to_id(train_data_kmer,id_dict) ##(10000, 997)
    print(train_data_id)

    #vocab = buil_vocab(k)
    '''
    
    #print(len(train_data_new))
    #print(len(train_data_new[0]))
    #print("finish preprocessing")
    
    model = CNN_Attention(k)

    
    for i in range(1):
        print("-----------epoch "+str(i+1)+"--------------")
        #train(model, train_data, train_labels)
        train(model, train_data, train_labels)
    #test(model, )
   # print('final accuracy {}'.format(test(model, test_inputs, test_labels)))
    
    #test_input_part = test_inputs[:10]
    #test_label_part = test_labels[:10]
    #visualize_results(test_input_part, model.call(test_input_part), test_label_part, CAT, DOG)
    return
    


if __name__ == '__main__':
    main()



