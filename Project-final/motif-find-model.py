import tensorflow as tf
import numpy as np
import transformer_func as transformer
from sklearn.metrics import roc_auc_score


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
        
       
        
        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=self.nkernels[0],kernel_size = self.conv_kernel_size,  padding='valid',activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool1D(pool_size=self.pool_kernel_size,strides=self.pool_kernel_size, padding='valid')
        
        # if use transformer after first conv, uncomment one of the following lines regarding to muti-head or single-head option.
        
        #self.encoder = transformer.Transformer_Block(self.nkernels[0], is_decoder=False, multi_headed=True)
        #self.encoder = transformer.Transformer_Block(self.nkernels[0], is_decoder=False, multi_headed=False)

        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=self.nkernels[1], kernel_size = self.conv_kernel_size, padding='valid',activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool1D(pool_size=self.pool_kernel_size, strides=self.pool_kernel_size, padding='valid')
        
        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=self.nkernels[2], kernel_size = self.conv_kernel_size, padding='valid',activation='relu')

        # if use transformer after the thrid conv, uncomment one of the following lines regarding to muti-head or single-head option.
        
        #self.encoder = transformer.Transformer_Block(self.nkernels[2], is_decoder=False, multi_headed=True)
        #self.encoder = transformer.Transformer_Block(self.nkernels[2], is_decoder=False, multi_headed=False)

        self.dense1 = tf.keras.layers.Dense(n_genomic_features, activation='relu')

        self.dense2 = tf.keras.layers.Dense(n_genomic_features, activation='sigmoid')   

         
        

    def call(self, inputs):
        

        inputs = tf.transpose(inputs, perm=[0, 2, 1]) 
        

        out = self.cnn_layer1(inputs)
        out = self.maxpool1(out)
        out = tf.nn.dropout(out,0.2)
        
        # if use transformer after first conv, uncomment the following line
        
        #out = self.encoder.call(out)
        
        out = self.cnn_layer2(out)
        out = self.maxpool2(out)
        out = tf.nn.dropout(out,0.2)
        
        
        out = self.cnn_layer3(out)
        out = tf.nn.dropout(out,0.5)
        
        # if use transformer after first conv, uncomment the following line
        #out = self.encoder.call(out)
        
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
        #return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return score
    
    


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
    # prediction results are collected across all batches and the AUROC score will be calculated after running the whole test dataset
    print(test_inputs.shape)
    all_pred = []
    
    for i in range(int(test_inputs.shape[0]/model.batch_size)):
        test_x_batch = test_inputs[i*model.batch_size:(i+1)*model.batch_size]
        test_y_batch = test_labels[i*model.batch_size:(i+1)*model.batch_size]
        test_y_batch = test_y_batch.astype(float)
        
        logits = model.call(test_x_batch.astype(float))
        all_pred.append(logits)
       
    all_pred = tf.reshape(all_pred,[-1])
    flatten_y = tf.reshape(test_labels,[-1])
    score =  model.accuracy(all_pred,flatten_y)
    return score


    

def main():

    train_labels = np.load('./reshape_labels_0_60000.npy')
    train_data = np.load('./reshape_data_0_60000.npy')
    print('label shape', train_labels.shape)
    print('train data shape:',train_data.shape)
   
    model = CNN_Attention()

    
    for i in range(5):
        print("-----------epoch "+str(i+1)+"--------------")
        
        train(model, train_data, train_labels)
    
    
    #==============TEST================
    test_labels = np.load('reshape_test_labels_10000_280000.npy')
    test_data = np.load('./reshape_test_data_10000_280000.npy')
    test_labels = test_labels[:10000]
    test_data = test_data[:10000]
    
    #test_accu = test(model,test_data,test_labels)
    AUROC = test(model,test_data,test_labels)
    print("Test AUROC score:",AUROC)
    
if __name__ == '__main__':
    main()



