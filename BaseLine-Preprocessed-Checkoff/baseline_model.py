import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float64')

class CNN_baseline(tf.keras.Model):
    def __init__(self):
        super(CNN_baseline, self).__init__()
        
        self.batch_size = 100
        self.conv_kernel_size = 8
        self.pool_kernel_size = 4
        
        nkernels = [320,480,960]
        sequence_length =  1000
        n_genomic_features  = 919
        
        
        
        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=320,kernel_size = self.conv_kernel_size, data_format='channels_first',padding='same',activation='relu')
        #self.maxpool1 = tf.nn.max_pool(ksize=pool_kernel_size, strides=pool_kernel_size)
        #self.dropout1 = tf.nn.dropout(0.2)
          
        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=480, kernel_size = self.conv_kernel_size, data_format='channels_first',padding='same',activation='relu')
        #self.maxpool2 = tf.nn.max_pool(ksize=pool_kernel_size, strides=pool_kernel_size)
        #self.dropout2 = tf.nn.dropout(0.2)
        
        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=960, kernel_size = self.conv_kernel_size, data_format='channels_first', padding='same',activation='relu')
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
        
        self.dense1 =tf.keras.layers.Dense(960 * self.n_channels, activation='relu')
        self.dense2 =tf.keras.layers.Dense(n_genomic_features, activation=tf.nn.sigmoid)      
        
        #params = self.parameters()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)

    def call(self, inputs):
        print("inputs.shape: ",inputs.shape)
        out = self.cnn_layer1(inputs)
        out = tf.nn.max_pool(out,ksize=self.pool_kernel_size, strides=self.pool_kernel_size, padding ='SAME')
        out = tf.nn.dropout(out,0.2)
        print("out.shape: ",out.shape)
        
        out = self.cnn_layer2(out)
        out = tf.nn.max_pool(out,ksize=self.pool_kernel_size, strides=self.pool_kernel_size, padding ='SAME')
        out = tf.nn.dropout(out,0.2)
        print("out.shape: ",out.shape)
        
        out = self.cnn_layer3(out)
        out = tf.nn.max_pool(out,ksize=self.pool_kernel_size, strides=self.pool_kernel_size, padding ='SAME')
        out = tf.nn.dropout(out,0.5)
        
        print("out.shape: ",out.shape)
        reshape_out = tf.reshape(out,[-1, 960 * self.n_channels])
        out = self.dense1(out)
        predict = self.dense2(out)
        return predict

def train(model, train_inputs, train_labels):
    loss = tf.keras.losses.BinaryCrossentropy()
    for i in range(int(train_inputs.shape[0]/model.batch_size)):
        train_x_batch = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        train_y_batch = train_labels[i*model.batch_size:(i+1)*model.batch_size]
        
        #train_x_batch = torch.tensor(train_x_batch)
        #train_y_batch = torch.tensor(train_y_batch)
        train_y_batch = train_y_batch.astype(float)
        logits = model.call(train_x_batch.astype(float))
        
        
        with tf.GradientTape() as tape:
            l = loss(logits,train_y_batch)
        print("loss: ",l)
        model.optimizer.apply_gradients([(gradients, parameters)])

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



