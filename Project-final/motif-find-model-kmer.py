
import tensorflow as tf
import numpy as np
import transformer_func as transformer
from sklearn.metrics import roc_auc_score

tf.keras.backend.set_floatx('float32')

def main():
    # train label and data
    train_labels = np.load('./reshape_labels_0_60000.npy')
    train_data = np.load('./data_id_0_60000.npy')
   # print('label shape', train_labels.shape)
    
    
   # print("finish preprocessing")
    
    model = CNN_Attention()

    
    for i in range(5):
        print("-----------epoch "+str(i+1)+"--------------")
        #train(model, train_data, train_labels)
        train(model, train_data, train_labels)
    #test(model, )
   # print('final accuracy {}'.format(test(model, test_inputs, test_labels)))
    
    test_labels = np.load('./reshape_test_labels_10000_280000.npy')
   
    test_data = np.load('./testdata_id_10000_80000.npy')
    print("sks", test_data.shape)
    test_labels = test_labels[:10000]
    test_data = test_data[:10000]
    #print(test_labels.shape)
    #print(test_labels)
    AUROC = test(model,test_data,test_labels)
    print("Test AUROC score:",AUROC)

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
        n_genomic_features  = 690
        
        self.num_E = np.power(5,self.k)

        self.E =  tf.Variable(tf.random.normal(shape=[self.num_E,self.embedding_size], stddev=.1, dtype=tf.float32))
        

        #Dense layer (output 919)  
        # remove data_format='channels_first',

        self.pos_encoder = transformer.Position_Encoding_Layer(self.dna_length,self.embedding_size)
        #self.encoder = transformer.Transformer_Block(self.embedding_size,is_decoder=False,multi_headed=True)
        self.encoder = transformer.Transformer_Block(self.embedding_size,is_decoder=False,multi_headed=False)
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
        #print("inputs.shape: ",inputs.shape)

        #k-mer
        #input_kmer = self.k_mers_block(self.k,inputs)
        #print(self.E.shape)
        #print(inputs.shape)
        encoder_input_embeddings = tf.nn.embedding_lookup(self.E,inputs)
        #print("encoder_input_embeddings: ",encoder_input_embeddings.shape)
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
        #print("dense1_out: ",out)
        predict = self.dense2(out)
        #print("dense2_out: ",out)
        #print('final output size: ', predict.shape)
        return predict
    
    def loss(self, logits, labels):
        #print('loss label shape', labels.shape)
        tf.keras.losses.BinaryCrossentropy()
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))
        return loss
    
    def accuracy(self, logits, labels):
       # print(labels.shape)
       # print(tf.reduce_sum(labels))
        score = roc_auc_score(labels, logits)
        #return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return score
    


def train(model, train_inputs, train_labels):
    loss = tf.keras.losses.BinaryCrossentropy()
    cur_loss = []
    for i in range(int(train_inputs.shape[0]/model.batch_size)):
        train_x_batch = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        train_y_batch = train_labels[i*model.batch_size:(i+1)*model.batch_size,125:815]
        train_y_batch = train_y_batch.astype(float)
        
        with tf.GradientTape() as tape:
            logits = model.call(train_x_batch)
            l = loss(y_pred = logits, y_true = tf.convert_to_tensor(train_y_batch))
        gradients = tape.gradient(l, model.trainable_variables)
        #print("loss: ", i, ": ", l)
        cur_loss.append(l)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
       
    cur_loss = tf.convert_to_tensor(cur_loss)
    print("mean train loss:",tf.reduce_mean(cur_loss))
        
def test(model, test_inputs, test_labels):
    total_accuracy = 0
    #print(test_inputs.shape)
    #AUROC_scores = []
    cur_logits = []
    for i in range(int(test_inputs.shape[0]/model.batch_size)):
        test_x_batch = test_inputs[i*model.batch_size:(i+1)*model.batch_size]
        #test_y_batch = test_labels[i*model.batch_size:(i+1)*model.batch_size,125:815]
        #test_y_batch = test_y_batch.astype(float)
        
        logits = model.call(test_x_batch)
        cur_logits.append(logits)
        #flatten_logits = tf.reshape(logits, [-1])
        #flatten_y = tf.reshape(test_y_batch, [-1])
        #accuracy = model.accuracy(logits, test_y_batch)
       # print("i=",i)
       # flatten_y = tf.reshape(test_y_batch, [-1])print(flatten_y.shape)
       # print(flatten_y)
        #score = model.accuracy(flatten_logits,flatten_y)
        #print("score:")
        #AUROC_scores.append(score)
        #total_accuracy += accuracy
        #print("Test batch ", i, "AUROC score: ", score)
    #return total_accuracy / (int(test_inputs.shape[0]/model.batch_size))
    flatten_logits = tf.reshape(cur_logits, [-1])
    flatten_y = tf.reshape(test_labels[:,125:815], [-1])
    score = model.accuracy(flatten_logits,flatten_y)
    return score





if __name__ == '__main__':
    main()



