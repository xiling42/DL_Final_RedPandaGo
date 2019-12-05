import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float64')

class CNN_baseline(tf.keras.Model):

    def __init__(self):
     
        super(CNN_baseline, self).__init__()
    
        self.batch_size = 100
        self.conv_kernel_size = 8
        self.pool_kernel_size = 4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.nkernels = [320,480,960]
        sequence_length =  1000
        n_genomic_features  = 919
        
        
        # remove data_format='channels_first',
        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=self.nkernels[0],kernel_size = self.conv_kernel_size,  padding='valid',activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool1D(pool_size=self.pool_kernel_size,strides=self.pool_kernel_size, padding='valid')

        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=self.nkernels[1], kernel_size = self.conv_kernel_size, padding='valid',activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool1D(pool_size=self.pool_kernel_size, strides=self.pool_kernel_size, padding='valid')
        
        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=self.nkernels[2], kernel_size = self.conv_kernel_size, padding='valid',activation='relu')

        self.dense1 = tf.keras.layers.Dense(n_genomic_features, activation='relu')
        self.dense2 = tf.keras.layers.Dense(n_genomic_features, activation='sigmoid')      
        
        #params = self.parameters()
        

    def call(self, inputs):
        inputs = tf.transpose(inputs, perm=[0, 2, 1]) 
        print("inputs.shape: ",inputs.shape)
        out = self.cnn_layer1(inputs)
        #print(out)
        print("out.shape cnn1: ",out.shape)
        #out = tf.nn.max_pool(out,ksize=self.pool_kernel_size, strides=self.pool_kernel_size, padding ='VALID')
        out = self.maxpool1(out)
        print('max output size: ', out.shape)
        out = tf.nn.dropout(out,0.2)
        print("out.shape1: ",out.shape)
        #print(out)
        
        out = self.cnn_layer2(out)
        print("out.shape cnn2: ",out.shape)
        out = self.maxpool2(out)
        #out = tf.nn.max_pool(out,ksize=self.pool_kernel_size, strides=self.pool_kernel_size, padding ='VALID')
        out = tf.nn.dropout(out,0.2)
        print("out.shape2: ",out.shape)
        
        out = self.cnn_layer3(out)
        out = tf.nn.dropout(out,0.5)
        
        print("out.shape3: ",out.shape)
        #out = tf.keras.layers.Flatten(out)
        #out.set_shape([out.shape[0], self.nkernels[2] * self.n_channels])
        out = tf.reshape(out,[out.shape[0], -1])
        #reshape_out = tf.reshape(out,[-1, self.nkernels[2] * self.n_channels])
        print("reshape size: ", out.shape)
        out = self.dense1(out)
        #predict = tf.keras.backend.sigmoid(out)
        predict = self.dense2(out)
        print('final output size: ', predict.shape)
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

def main():
    # train label and data
    train_labels = np.load('./reshape_labels_0_10000.npy')
    train_data = np.load('./reshape_data_0_10000.npy')
    print('label shape', train_labels.shape)
    
    model = CNN_baseline()
    for i in range(10):
        print("-----------epoch "+str(i+1)+"--------------")
        train(model, train_data, train_labels)
    #test(model, )
   # print('final accuracy {}'.format(test(model, test_inputs, test_labels)))
    
    #test_input_part = test_inputs[:10]
    #test_label_part = test_labels[:10]
    #visualize_results(test_input_part, model.call(test_input_part), test_label_part, CAT, DOG)
    return



if __name__ == '__main__':
    main()



