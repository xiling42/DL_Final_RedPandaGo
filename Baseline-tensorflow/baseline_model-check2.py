import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float64')

class CNN_baseline(tf.keras.Model):

    def __init__(self):
     
        super(CNN_baseline, self).__init__()
    
        self.batch_size = 100
        self.conv_kernel_size = 8
        self.pool_kernel_size = 4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        
        nkernels = [320,480,560]
        sequence_length =  1000
        n_genomic_features  = 919
        
        
        # remove data_format='channels_first',
        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=nkernels[0], kernel_size = self.conv_kernel_size,  padding='valid',activation='relu')
        #self.maxpool1 = tf.keras.layers.MaxPool1D(pool_size=self.pool_kernel_size, strides=self.pool_kernel_size, padding='valid')
        #self.dropout1 = tf.nn.dropout(0.2)
        
        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=nkernels[1], kernel_size = self.conv_kernel_size, padding='valid',activation='relu')
        #self.maxpool2 = tf.keras.layers.MaxPool1D(pool_size=self.pool_kernel_size, strides=self.pool_kernel_size, padding='valid')
        #self.maxpool2 = tf.nn.max_pool(ksize=pool_kernel_size, strides=pool_kernel_size)
        #self.dropout2 = tf.nn.dropout(0.2)
        
        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=nkernels[2], kernel_size = self.conv_kernel_size, padding='valid',activation='relu')
        #self.maxpool3 = tf.keras.layers.MaxPool1D(pool_size=self.pool_kernel_size, strides=self.pool_kernel_size, padding='valid')
        #self.maxpool3 = tf.nn.max_pool(ksize=pool_kernel_size, strides=pool_kernel_size)
        #self.dropout3 = tf.nn.dropout(0.5)
        
        reduce_by = self.conv_kernel_size - 1
        pool_kernel_size = float(self.pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        print("reduce_by: ",reduce_by)
        print("self.n_channels: ",self.n_channels)
        
        self.dense1 = tf.keras.layers.Dense(nkernels[2] * self.n_channels, activation='relu')
        self.dense2 = tf.keras.layers.Dense(n_genomic_features, activation='sigmoid')      
        
        #params = self.parameters()
        

    def call(self, inputs):
        inputs = tf.transpose(inputs, perm=[0, 2, 1]) 
        print("inputs.shape: ",inputs.shape)
        out = self.cnn_layer1(inputs)
        
        print("out.shape cnn1: ",out.shape)
        out = tf.nn.max_pool(out,ksize=self.pool_kernel_size, strides=self.pool_kernel_size, padding ='VALID')
        #out = self.maxpool1(out)
        print('max output size: ', out.shape)
        out = tf.nn.dropout(out,0.2)
        print("out.shape1: ",out.shape)
        
        out = self.cnn_layer2(out)
        print("out.shape cnn2: ",out.shape)
        #out = self.maxpool1(out)
        out = tf.nn.max_pool(out,ksize=self.pool_kernel_size, strides=self.pool_kernel_size, padding ='VALID')
        out = tf.nn.dropout(out,0.2)
        print("out.shape2: ",out.shape)
        
        out = self.cnn_layer3(out)
        out = tf.nn.dropout(out,0.5)
        
        print("out.shape3: ",out.shape)
        reshape_out = tf.reshape(out,[-1, 560 * self.n_channels])
        print("reshape size: ", reshape_out.shape)
        out = self.dense1(reshape_out)
        predict = self.dense2(out)
        print('final output size: ', predict.shape)
        return predict
    
#    def loss(self, logits, labels):
#        return tf.keras.backend.binary_crossentropy(labels, logits, from_logits=True)

def train(model, train_inputs, train_labels):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    for i in range(int(train_inputs.shape[0]/model.batch_size)):
        train_x_batch = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        train_y_batch = train_labels[i*model.batch_size:(i+1)*model.batch_size]
        train_y_batch = train_y_batch.astype(float)
        
        with tf.GradientTape() as tape:
            logits = model.call(train_x_batch.astype(float))
            tape.watch(model.trainable_variables)
            #l = model.loss(logits, train_y_batch)
            l = loss(logits, tf.convert_to_tensor(train_y_batch))
        gradients = tape.gradient(l, model.trainable_variables)
        print("loss: ",l)
        #self.optimizer.minimize(cost)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        

def main():
    # train label and data
    labels = np.load('./reshape_labels_0_10000.npy')
    data = np.load('./reshape_data_0_10000.npy')
    
    model = CNN_baseline()
    for i in range(1):
        print("-----------epoch "+str(i+1)+"--------------")
        train(model, data, labels)
   # print('final accuracy {}'.format(test(model, test_inputs, test_labels)))
    
    #test_input_part = test_inputs[:10]
    #test_label_part = test_labels[:10]
    #visualize_results(test_input_part, model.call(test_input_part), test_label_part, CAT, DOG)
    return



if __name__ == '__main__':
    main()



