
import tensorflow as tf
import numpy as np
import transformer_func as transformer
from sklearn.metrics import roc_auc_score

tf.keras.backend.set_floatx('float32')

def main():
    # train label and data
    train_labels = np.load('./reshape_labels_0_60000.npy')
    train_data = np.load('./data_id_0_60000.npy')
  
    
    model = CNN_Attention()

    
    for i in range(5):
        print("-----------epoch "+str(i+1)+"--------------")
        
        train(model, train_data, train_labels)
   
    
    test_labels = np.load('./reshape_test_labels_10000_280000.npy')
    test_data = np.load('./testdata_id_10000_80000.npy')

    test_labels = test_labels[:10000]
    test_data = test_data[:10000]
   
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

         
        

    def call(self, inputs):
        


        #k-mer
       
        encoder_input_embeddings = tf.nn.embedding_lookup(self.E,inputs)
        pos_encoder_output = self.pos_encoder(encoder_input_embeddings)
        encoder_output = self.encoder(pos_encoder_output)
        
        out = self.cnn_layer1(encoder_output)
        out = self.maxpool1(out)
        out = tf.nn.dropout(out,0.2)
  
        
        out = self.cnn_layer2(out)
        out = self.maxpool2(out)
        out = tf.nn.dropout(out,0.2)

        
        out = self.cnn_layer3(out)
        out = tf.nn.dropout(out,0.5)
        
       
        
        out = tf.reshape(out,[out.shape[0], -1])
        out = self.dense1(out)
        
        predict = self.dense2(out)
       
        return predict
    
    def loss(self, logits, labels):
       
        tf.keras.losses.BinaryCrossentropy()
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))
        return loss
    
    def accuracy(self, logits, labels):
     
        score = roc_auc_score(labels, logits)
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

        cur_loss.append(l)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
       
    cur_loss = tf.convert_to_tensor(cur_loss)
    print("mean train loss:",tf.reduce_mean(cur_loss))
        
def test(model, test_inputs, test_labels):
  
    
    cur_logits = []
    for i in range(int(test_inputs.shape[0]/model.batch_size)):
        test_x_batch = test_inputs[i*model.batch_size:(i+1)*model.batch_size]
        logits = model.call(test_x_batch)
        cur_logits.append(logits)
        
    flatten_logits = tf.reshape(cur_logits, [-1])
    flatten_y = tf.reshape(test_labels[:,125:815], [-1])
    score = model.accuracy(flatten_logits,flatten_y)
    return score





if __name__ == '__main__':
    main()



