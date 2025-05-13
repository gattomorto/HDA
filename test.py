import unittest
import v4
import numpy as np
import tensorflow as tf
import conv
import time
import random


class Regrow(unittest.TestCase):
    @staticmethod
    def create_indices(num_free,layer_idx,model):
        '''
        riempie W_indices[layer_idx] di indici casuali tale che ne rimangano liberi num_free

        :param num_free:
        :param layer_idx:
        :param model:
        :return: void
        '''
        rows, cols = model.W_shapes[layer_idx]
        total = rows * cols
        nnz = total- num_free
        rng = np.random.default_rng(0)
        chosen = rng.choice(total, size=nnz, replace=False)
        indices = np.stack(np.unravel_index(chosen, model.W_shapes[layer_idx]), axis=1)
        model.W_indices[layer_idx] = tf.constant(indices, dtype=tf.int64)

    def test_A(self):

        # nesun parametro è importante tranne num_hidden
        m = v4.FFNsSparse3(input_dim=100,hidden_dim=100,output_dim=100,num_hidden_layers=3,sparsity=0)
        #momenta deve contenere i momenti di ogni variabile per ogni layer
        #quindi la prima lista deve essere di dimensione 14*14
        #qui è stato fatto in modo tale che la media dei momenti per lay1 sia 6, poi 1,2,1
        momenta = [[6,6,7,5],[1,2,0],[2],[1,1,0,0,2,2]]

        m.W_shapes[0] = [14,14] #196 pesi
        m.W_shapes[1] = [20,20]
        m.W_shapes[2] = [10,10]
        m.W_shapes[3] = [12,12]

        self.create_indices(100,0,m)
        self.create_indices(300,1,m)
        self.create_indices(10,2,m)
        self.create_indices(25,3,m)

        nnz_before_regrowth = v4.get_total_nonzero_weights(m)

        v4.regrow(m,momenta,200,[0,1,2,3])

        l0_size=len(m.W_indices[0])
        self.assertEqual(l0_size,14*14-0)#layer 0 deve essere pieno , quindi di dimenensione 196

        l1_size=len(m.W_indices[1])
        self.assertEqual(l1_size,20*20-235)#layer 1 deve avere 235 posizioni libere

        l2_size=len(m.W_indices[2])
        self.assertEqual(l2_size,10*10-0)#layer 2 deve essere saturo (0 posizioni libere)

        l3_size=len(m.W_indices[3])
        self.assertEqual(l3_size,12*12-0)#layer 3 deve essere saturo (0 posizioni libere)

        nnz_after_regrowth = v4.get_total_nonzero_weights(m)
        self.assertEqual(nnz_before_regrowth+200,nnz_after_regrowth)#devono essere stati distribuiti esattamente 200 pesi

    def test_B(self):
        '''
        # si testa cosa succede se allocazione = spazio libero
        '''
        # nesun parametro è importante tranne num_hidden
        m = v4.FFNsSparse3(input_dim=100,hidden_dim=100,output_dim=100,num_hidden_layers=3,sparsity=0)

        momenta = [6,1,2,1]

        m.W_shapes[0] = [14,14] #196 pesi
        m.W_shapes[1] = [20,20]
        m.W_shapes[2] = [10,10]
        m.W_shapes[3] = [12,12]

        self.create_indices(120,0,m)
        self.create_indices(300,1,m)
        self.create_indices(10,2,m)
        self.create_indices(25,3,m)

        nnz_before_regrowth = v4.get_total_nonzero_weights(m)

        v4.regrow(m,momenta,200,[0,1,2,3])

        l0_size=len(m.W_indices[0])
        self.assertEqual(l0_size,14*14-0)#layer 0 deve essere pieno , quindi di dimenensione 196

        l1_size=len(m.W_indices[1])
        self.assertEqual(l1_size,20*20-255)#layer 1 deve avere 255 posizioni libere

        l2_size=len(m.W_indices[2])
        self.assertEqual(l2_size,10*10-0)#layer 2 deve essere saturo (0 posizioni libere)

        l3_size=len(m.W_indices[3])
        self.assertEqual(l3_size,12*12-0)#layer 3 deve essere saturo (0 posizioni libere)

        nnz_after_regrowth = v4.get_total_nonzero_weights(m)
        self.assertEqual(nnz_before_regrowth + 200, nnz_after_regrowth)  # devono essere stati distribuiti esattamente 200 pesi

    def test_C(self):
        '''
        regrow 1 peso
        '''
        # nesun parametro è importante tranne num_hidden
        m = v4.FFNsSparse3(input_dim=100,hidden_dim=100,output_dim=100,num_hidden_layers=3,sparsity=0)

        momenta = [6,1,2,1]

        m.W_shapes[0] = [14,14] #196 pesi
        m.W_shapes[1] = [20,20]
        m.W_shapes[2] = [10,10]
        m.W_shapes[3] = [12,12]

        self.create_indices(120,0,m)
        self.create_indices(300,1,m)
        self.create_indices(10,2,m)
        self.create_indices(25,3,m)

        nnz_before_regrowth = v4.get_total_nonzero_weights(m)

        v4.regrow(m,momenta,1,[0,1,2,3])

        l0_size=len(m.W_indices[0])
        self.assertEqual(l0_size,14*14-119)# l'unico peso deve nascere in l0 (119 liberi)

        l1_size=len(m.W_indices[1])
        self.assertEqual(l1_size,20*20-300)#layer 1 invariato

        l2_size=len(m.W_indices[2])
        self.assertEqual(l2_size,10*10-10)#layer 2 invariato

        l3_size=len(m.W_indices[3])
        self.assertEqual(l3_size,12*12-25)#layer 3 invariato

        nnz_after_regrowth = v4.get_total_nonzero_weights(m)
        self.assertEqual(nnz_before_regrowth + 1, nnz_after_regrowth)

    def test_D(self):
        '''
        regrow 1 peso con momenti uguali
        devo ancora capire come gestire
        '''
        # nesun parametro è importante tranne num_hidden=1
        m = v4.FFNsSparse3(input_dim=100,hidden_dim=100,output_dim=100,num_hidden_layers=1,sparsity=0)

        momenta = [1,1]

        m.W_shapes[0] = [14,14] #196 pesi
        m.W_shapes[1] = [20,20]

        self.create_indices(120,0,m)
        self.create_indices(300,1,m)

        debdt = v4.regrow(m,momenta,1,[0,1])
        self.assertEqual(debdt,1)

    def test_E(self):
        '''
        vedo cosa succede se si cercano di allocare piu pesi di quanti sono disponibili
        '''
        # nessun parametro è importante tranne num_hidden=1
        m = v4.FFNsSparse3(input_dim=100,hidden_dim=100,output_dim=100,num_hidden_layers=1,sparsity=0)

        momenta = [2,1]

        m.W_shapes[0] = [2,2]#4 pesi
        m.W_shapes[1] = [2,2]#4 pesi

        self.create_indices(2,0,m)
        self.create_indices(1,1,m)

        with self.assertRaises(Exception) as context:
            v4.regrow(m, momenta, 100, [0, 1])
        self.assertIn("layers empty, probably to_grow > available space", str(context.exception))

    #TODO
    def test_F(self):
        pass
        #testare quando momentum sono a zero

    def test_G(self):
        '''
        test regrow_layer
        nota che non W1 è inizializzato ma non si usa
        '''
        # nessun parametro è importante tranne num_hidden=1
        m = v4.FFNsSparse3(input_dim=100,hidden_dim=100,output_dim=100,num_hidden_layers=1,
                           sparsity=0)

        m.W_shapes[0] = [2,2]#4 pesi
        m.W_shapes[1] = [2,2]#4 pesi

        indices0 = np.array([[0,1],[1,0]])
        values0 = np.array([    -1,
                                   3    ],dtype=np.float32)

        indices1 = np.array([[0,1]])
        values1 = np.array([    -1,
                                         ],dtype=np.float32)

        m.W_indices[0] = tf.constant(indices0, dtype=tf.int64)
        m.W_values[0] = tf.Variable(values0, name=f"W{0}_values", trainable=True)

        m.W_indices[1] = tf.constant(indices1, dtype=tf.int64)
        m.W_values[1] = tf.Variable(values1, name=f"W{1}_values", trainable=True)

        regrown = v4.regrow_layer(0,m,200)
        self.assertEqual(regrown,2)#perche ci sono solo due posti liberi

        # in questo blocco si testa che indices0 corrispondano ancora agli stessi valori values0
        new_indices0 =m.W_indices[0].numpy()
        position_index_01 = np.where((new_indices0 == [0,1]).all(axis=1))[0]
        value01 = m.W_values[0].numpy()[position_index_01]
        self.assertEqual(-1,value01)
        position_index_10 = np.where((new_indices0 == [1,0]).all(axis=1))[0]
        value01 = m.W_values[0].numpy()[position_index_10]
        self.assertEqual(3,value01)

class Prune(unittest.TestCase):
    def test_A(self):
        '''
        test funzionamento base di prune_layer
        '''
        m = v4.FFNsSparse3(input_dim=2,hidden_dim=2,output_dim=2,num_hidden_layers=0,sparsity=0)
        values_before_pruning = np.array([0.1,-0.3,
                                          0.01,0.4])
        m.W_values[0] = tf.Variable(values_before_pruning, name=f"W{0}_values", trainable=True)
        num_pruned = v4.prune_layer(0,m)

        values_after_pruning = m.W_values[0].numpy()
        # devono rimanere i due piu grandi in valore assoluto
        self.assertTrue( np.array_equal(values_after_pruning,np.array([-0.3,0.4])))

        indices_after_pruning = m.W_indices[0].numpy()
        self.assertTrue( np.array_equal(indices_after_pruning,np.array([[0,1],[1,1]])))

    def test_B(self):
        '''
        prune_layer in caso di tutti i pesi uguali
        '''
        m = v4.FFNsSparse3(input_dim=2,hidden_dim=2,output_dim=2,num_hidden_layers=0,sparsity=0)
        values_before_pruning = np.array([-0.3, -0.3,
                                         - 0.3, -0.3])
        m.W_values[0] = tf.Variable(values_before_pruning, name=f"W{0}_values", trainable=True)
        num_pruned = v4.prune_layer(0,m)

        values_after_pruning = m.W_values[0].numpy()
        self.assertTrue(np.array_equal(values_after_pruning,np.array([-0.3,-0.3])))

    def test_C(self):
        '''
        prune_layer se gli indici non hanno un ordine particolare
        '''
        m = v4.FFNsSparse3(input_dim=2,hidden_dim=2,output_dim=2,num_hidden_layers=0,sparsity=0)
        values_before_pruning = np.array([0.1,-0.3,
                                          0.01,0.4])
        m.W_values[0] = tf.Variable(values_before_pruning, name=f"W{0}_values", trainable=True)
        m.W_indices[0] = np.array([[1,1],[0,0],
                                   [0,1],[1,0]])
        num_pruned = v4.prune_layer(0,m)

        values_after_pruning = m.W_values[0].numpy()
        self.assertTrue( np.array_equal(values_after_pruning,np.array([-0.3,0.4])))

        indices_after_pruning = m.W_indices[0].numpy()
        self.assertTrue( np.array_equal(indices_after_pruning,np.array([[0,0],[1,0]])))


class Conv(unittest.TestCase):

    # C_in = 1, C_out = 1, Q denso
    def test_conv(self):

        '''
        #test hard coded
        X = tf.constant([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=tf.float32)
        Q = tf.constant([[1, -1],
                         [3, 2]], dtype=tf.float32)
        K = 2

        custom_out = prove.conv(X, Q)

        X_tf = tf.reshape(X, [1, X.shape[0], X.shape[1], 1])  # [batch, height, width, channels]
        Q_tf = tf.reshape(Q, [K, K, 1, 1])  # [filter_height, filter_width, in_channels, out_channels]

        tf_out = tf.nn.conv2d(X_tf, Q_tf, strides=1, padding="VALID")
        tf_out = tf.squeeze(tf_out)  # Remove batch and channel dims

        self.assertTrue(tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-5), "Sparse conv output does not match tf.nn.conv2d")

        '''

        num_tests = 100
        max_size = 100
        min_size = 50
        max_k = 5
        min_k = 2

        for _ in range(num_tests):
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(H, W).astype(np.float32))
            Q = tf.constant(np.random.randn(K, K).astype(np.float32))
            custom_out = prove.conv(X, Q)

            X_tf = tf.reshape(X, [1, H, W, 1])  # [batch, height, width, channels]
            Q_tf = tf.reshape(Q, [K, K, 1, 1])  # [filter_height, filter_width, in_channels, out_channels]
            tf_out = tf.nn.conv2d(X_tf, Q_tf, strides=1, padding="VALID")
            tf_out = tf.squeeze(tf_out)  # Remove batch and channel dims

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-5)

            if not close:
                print("Input X:", X.numpy())
                print("Filter Q:", Q.numpy())
                print("Custom output:", custom_out.numpy())
                print("TF output:", tf_out.numpy())

            self.assertTrue(close, "Sparse conv output does not match tf.nn.conv2d for random case")

    # C_in = x, C_out = 1, Q denso
    def test_conv3(self):
        # Test multiple random cases
        num_tests = 30
        max_size = 50
        min_size = 10
        min_k = 2
        max_k = min_size
        num_channels = 100

        for _ in range(num_tests):
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(1, H, W,num_channels ).astype(np.float32))  # [H, W, 2]
            Q = tf.constant(np.random.randn(K, K, num_channels, 1).astype(np.float32))

            # Your custom convolution implementation (needs to handle 2 input channels)
            custom_out = prove.conv3(X, Q)

            # TensorFlow's implementation
            tf_out = tf.nn.conv2d(X, Q, strides=1, padding="VALID")  # [1, H', W', 1]
            tf_out = tf.squeeze(tf_out)  # Remove batch and channel dims -> [H', W']

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-2)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    # C_in = x, C_out = y, Q denso
    def test_conv4(self):
        # Test multiple random cases
        num_tests = 10
        max_size = 50
        min_size = 10
        min_k = 2
        max_k = min_size
        num_channels_in = 10
        num_channels_out = 10

        for _ in range(num_tests):
            # Generate random input with 2 channels
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(1, H, W,num_channels_in ).astype(np.float32))
            Q = tf.constant(np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32))

            custom_out = prove.conv4(X, Q)

            tf_out = tf.nn.conv2d(X, Q, strides=1, padding="VALID")  # [1, H', W', 1]
            #tf_out = tf.squeeze(tf_out)  # Remove batch and channel dims -> [H', W']

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    # C_in = x, C_out = y, N = z, Q denso
    def test_conv5(self):

        '''
        test hardcoded
        channel_01 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # Shape [3, 3]
        channel_11 = tf.constant([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])  # Shape [3, 3]
        channel_21 = tf.constant([[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]])  # Shape [3, 3]
        X1 = tf.stack([channel_01, channel_11, channel_21], axis=-1)
        channel_02 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # Shape [3, 3]
        channel_12 = tf.constant([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])  # Shape [3, 3]
        channel_22 = tf.constant([[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]])  # Shape [3, 3]
        X2 = tf.stack([channel_02, channel_12, channel_22], axis=-1)

        X = tf.stack([X1, X2], axis=0)

        Q = tf.constant([
            [  # First row of kernel
                [[1.0, 100.0], [5.0, 105.0], [9.0, 109.0]],  # input channels × output channels
                [[2.0, 102.0], [6.0, 106.0], [11.0, 111.0]]
            ],
            [  # Second row of kernel
                [[3.0, 103.0], [7.0, 107.0], [13.0, 113.0]],
                [[4.0, 104.0], [8.0, 108.0], [15.0, 115.0]]
            ]
        ], dtype=tf.float32)  # Shape [2, 2, 3, 2]

        custom_out = conv5(X, Q)  # Should return [H', W', 1]

        #X_tf = tf.reshape(X, [1, 3, 3, 3])  # [batch, height, width, in_channels]
        tf_out = tf.nn.conv2d(X, Q, strides=1, padding="VALID")
        #tf_out = tf.squeeze(tf_out)  # Remove batch and channel dims -> [H', W']

        # Compare results
        close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-5)
        print(close)

        '''
        # Test multiple random cases
        num_tests = 50
        max_size = 100
        min_size = 50
        min_k = 2
        max_k = min_size
        num_channels_in = 10
        num_channels_out = 10
        N_max = 10

        for _ in range(num_tests):
            N = np.random.randint(N_max)+1
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in ).astype(np.float32))
            Q = tf.constant(np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32))

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q, strides=1, padding="VALID")  # [1, H', W', 1]
            print("fine tf")

            print("inizio cus")
            custom_out = prove.conv5(X, Q)
            print("fine cus")



            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)
############################################################
    # C_in = 1, C_out = 1, Q sparso
    def test_conv_sparse(self):
        '''

        K = 2
        Q_dense = tf.constant([[-1,   0],
                               [-0. , 2]])
        Q_sparse = tf.sparse.from_dense(Q_dense)
        X = tf.constant([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=tf.float32)

        c_mio = conv_sparse(X,Q_sparse)

        X_tf = tf.reshape(X, [1, X.shape[0], X.shape[1], 1])  # [batch, height, width, channels]
        Q_tf = tf.reshape(Q_dense, [K, K, 1, 1])  # [filter_height, filter_width, in_channels, out_channels]

        tf_out = tf.nn.conv2d(X_tf, Q_tf, strides=1, padding="VALID")
        tf_out = tf.squeeze(tf_out)  # Remove batch and channel dims
        print(tf.reduce_all(tf.abs(c_mio - tf_out) < 1e-5).numpy())

        '''
        # Test multiple random cases
        num_tests = 10
        max_size = 10
        min_size = 3
        max_k = 3
        min_k = 3

        for _ in range(num_tests):
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            #X = tf.constant(np.random.randn(H, W).astype(np.float32))
            X=tf.random.uniform([H,W])

            sparsity = 0.2
            Q_dense = np.random.randn(K, K).astype(np.float32)
            mask = np.random.rand(K, K) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sparse = tf.sparse.from_dense(Q_dense)

            X_tf = tf.reshape(X, [1, H, W, 1])  # [batch, height, width, channels]
            Q_tf = tf.reshape(Q_dense, [K, K, 1, 1])  # [filter_height, filter_width, in_channels, out_channels]
            print("inizio tf")
            tf_out = tf.nn.conv2d(X_tf, Q_tf, strides=1, padding="VALID")
            print("fine tf")

            tf_out = tf.squeeze(tf_out)  # Remove batch and channel dimensions

            print("inizio cus")
            try:
                custom_out = prove.conv_sparse(X, Q_sparse)
            except:
                pass
            print("fine cus")


            # Check if outputs are close
            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-5)

            if not close:
                print("Input X:", X.numpy())
                print("Dense Filter Q:", Q_dense)
                print("Custom output:", custom_out.numpy())
                print("TF output:", tf_out.numpy())

            assert close, "Sparse conv output does not match tf.nn.conv2d for random case"

    # C_in = x, C_out = y, N = z, Q sparso
    def test_conv_sparse2(self):
        '''

        channel_01 = tf.constant([[ 1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # Shape [3, 3]
        channel_11 = tf.constant([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])  # Shape [3, 3]
        channel_21 = tf.constant([[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]])  # Shape [3, 3]
        X1 = tf.stack([channel_01, channel_11, channel_21], axis=-1)
        channel_02 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # Shape [3, 3]
        channel_12 = tf.constant([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])  # Shape [3, 3]
        channel_22 = tf.constant([[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]])  # Shape [3, 3]
        X2 = tf.stack([channel_02, channel_12, channel_22], axis=-1)

        X = tf.stack([X1, X2], axis=0)

        Q = tf.constant([
            [  # First row of kernel
                [[1.0, 100.0], [5.0, 105.0], [9.0, 109.0]],
                [[2.0, 102.0], [6.0, 106.0], [11.0, 111.0]]
            ],
            [  # Second row of kernel
                [[3.0, 103.0], [7.0, 107.0], [13.0, 113.0]],
                [[4.0, 104.0], [8.0, 108.0], [15.0, 115.0]]
            ]
        ], dtype=tf.float32)  # Shape [2, 2, 3, 2]

        Q_sp = tf.sparse.from_dense(Q)

        custom_out = conv_sparse2(X, Q_sp)  # Should return [H', W', 1]

        #X_tf = tf.reshape(X, [1, 3, 3, 3])  # [batch, height, width, in_channels]
        tf_out = tf.nn.conv2d(X, Q, strides=1, padding="VALID")
        #tf_out = tf.squeeze(tf_out)  # Remove batch and channel dims -> [H', W']

        # Compare results
        close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-5)
        print(close)
        '''
        num_tests = 10
        max_size = 50
        min_size = 10
        min_k = 2
        max_k = 2
        num_channels_in = 10
        num_channels_out = 10
        N_max = 10

        for _ in range(num_tests):
            N = np.random.randint(N_max)+1
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in ).astype(np.float32))

            sparsity = 1
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=1, padding="VALID")  # [1, H', W', 1]
            print("fine tf")

            print("inizio custom")
            custom_out = prove.conv_sparse2(X, Q_sp)
            print("fine custom")


            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)
#############################################################
    # C_in = 1, C_out = 1, Q sparso, fast
    # vettorizza sparse_to_sparse
    def test_conv_sparse_fast(self):
        num_tests = 30
        max_size = 1000
        min_size = 1000
        max_k = 3
        min_k = 3

        for _ in range(num_tests):
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            # X = tf.constant(np.random.randn(H, W).astype(np.float32))
            X = tf.random.uniform([H, W])

            sparsity = 0.9
            Q_dense = np.random.randn(K, K).astype(np.float32)
            mask = np.random.rand(K, K) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sparse = tf.sparse.from_dense(Q_dense)

            X_tf = tf.reshape(X, [1, H, W, 1])  # [batch, height, width, channels]
            Q_tf = tf.reshape(Q_dense, [K, K, 1, 1])  # [filter_height, filter_width, in_channels, out_channels]
            print("inizio tf")
            tf_out = tf.nn.conv2d(X_tf, Q_tf, strides=1, padding="VALID")
            print("fine tf")

            tf_out = tf.squeeze(tf_out)  # Remove batch and channel dimensions

            print("inizio cus")
            try:
                custom_out = prove.conv_sparse_fast(X, Q_sparse)
            except:
                pass
            print("fine cus")

            # Check if outputs are close
            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-5)

            if not close:
                print("Input X:", X.numpy())
                print("Dense Filter Q:", Q_dense)
                print("Custom output:", custom_out.numpy())
                print("TF output:", tf_out.numpy())

            assert close, "Sparse conv output does not match tf.nn.conv2d for random case"

    # c_in = x, c_out = y, N = z, Q sparse fast
    # usa sparse_to_sparse2 ma all'interno chiama sparse_to_sparse_fast
    def test_conv_sparse_fast2(self):
        num_tests = 10
        max_size = 10
        min_size = 10
        min_k = 2
        max_k = min_size
        num_channels_in = 10
        num_channels_out = 10
        N_max = 10

        for _ in range(num_tests):
            N = np.random.randint(N_max) + 1
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))

            sparsity = 0.99
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=1, padding="VALID")  # [1, H', W', 1]
            print("fine tf")

            print("inizio custom")
            custom_out = prove.conv_sparse_fast2(X, Q_sp)
            print("fine custom")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    # c_in = x, c_out = y, N = z, Q sparse, fast
    # vettorizza sparse_to_sparse2 e all'interno chiama sparse_to_sparse_fast
    def test_conv_sparse_fast3(self):
        num_tests = 1000
        max_size = 300
        min_size = 300
        min_k = 3
        max_k = 3
        num_channels_in = 128#10
        num_channels_out = 256
        N_max = 128

        for _ in range(num_tests):
            N = np.random.randint(N_max) + 1
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))

            sparsity = 0.99
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=1, padding="VALID")  # [1, H', W', 1]
            print("fine tf")

            print("inizio custom")
            try:
                custom_out = prove.conv_sparse_fast3(X, Q_sp)
            except:
                exit()
            print("fine custom")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    # c_in = x, c_out = y, N = z, Q sparse, fast
    # sparse_to_sparse_fast3 fused con sparse_to_sparse_fast
    def test_conv_sparse_fast4(self):
        num_tests = 1000
        max_size = 128
        min_size = 128
        min_k = 3
        max_k = 3
        num_channels_in = 10
        num_channels_out = 10
        N_max = 5
        N_min = 2

        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))

            sparsity = 0.9
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=1, padding="VALID")  # [1, H', W', 1]
            print("fine tf")

            print("inizio custom")

            custom_out = prove.conv_sparse_fast4(X, Q_sp)

            print("fine custom")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    # questa proprio non ha for
    def test_conv_sparse_fast5(self):
        num_tests = 1000
        max_size = 128
        min_size = 128
        min_k = 3
        max_k = 3
        num_channels_in = 256
        num_channels_out = 128
        N_max = 50
        N_min = 40

        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))

            sparsity = 0.99
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=1, padding="VALID")  # [1, H', W', 1]
            print("fine tf")

            print("inizio custom")
            custom_out = prove.conv_sparse_fast5(X, Q_sp)
            print("fine custom")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    # un altra versione potenzialmente piu veloce ma non sembra
    def test_conv_sparse_fast6(self):
        num_tests = 1000
        max_size = 128
        min_size = 128
        min_k = 2
        max_k = 2
        num_channels_in = 10
        num_channels_out = 20
        N_max = 50
        N_min = 10

        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))

            sparsity = 0.9
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=1, padding="VALID")  # [1, H', W', 1]
            print("fine tf")

            print("inizio custom")
            custom_out = prove.conv_sparse_fast6(X, Q_sp)
            print("fine custom")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    # versione piu veloce di tutte quelle precedenti
    def test_conv_sparse_fast7(self):
        num_tests = 1000
        max_size = 128
        min_size = 128
        min_k = 2
        max_k = 2
        num_channels_in = 128
        num_channels_out = 20
        N_max = 256
        N_min = 128

        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))

            sparsity = 0.99
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=1, padding="VALID")  # [1, H', W', 1]
            print("fine tf")

            print("inizio custom")
            custom_out = prove.conv_sparse_fast7(X, Q_sp)
            print("fine custom")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    # versione piu veloce di tutte quelle precedenti
    def test_conv_sparse_fast8(self):
        num_tests = 1000
        max_size = 128
        min_size = 128
        min_k = 2
        max_k = 2
        num_channels_in = 4
        num_channels_out = 5
        N_max = 5
        N_min = 2

        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))

            sparsity = 0.9
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=1, padding="VALID")  # [1, H', W', 1]
            print("fine tf")

            print("inizio custom")
            custom_out = prove.conv_sparse_fast8(X, Q_sp)
            print("fine custom")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    ################################################################
    # faccio dense matmul invece di sparse matmul
    #
    def test_conv_sparse_fast_D8(self):
        num_tests = 1000
        max_size = 64
        min_size = 64
        min_k = 3
        max_k = 3
        num_channels_in = 10
        num_channels_out = 5
        N_max = 16
        N_min = 15

        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))

            sparsity = 0.9
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)



            print("inizio custom")
            custom_out = prove.conv_sparse_fast_D8(X, Q_sp)
            print("fine custom")

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=1, padding="VALID")  # [1, H', W', 1]
            print("fine tf")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    ################################################################

    # conv_sparse_fast8 & padding = same
    def test_conv_sparse_fast8_padding(self):
        num_tests = 10000
        max_size = 244
        min_size = 50
        min_k = 2
        max_k = 7
        num_channels_in = 1
        num_channels_out = 10
        N_max = 29
        N_min = 2

        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))

            sparsity = 0.9
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=1, padding="SAME")  # [1, H', W', 1]
            print("fine tf")

            print("inizio custom")
            custom_out = conv.conv_sparse_fast8_padding(X, Q_sp)
            print("fine custom")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-3)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    # conv_sparse_fast8 & padding un parametro
    def test_conv_sparse_fast8_padding_v2(self):
        num_tests = 10000
        max_size = 244
        min_size = 50
        min_k = 2
        max_k = 7
        num_channels_in = 1
        num_channels_out = 10
        N_max = 29
        N_min = 2

        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))
            padding = "VALID" if random.randint(0, 1) == 0 else "SAME"

            sparsity = 0.91
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)

            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=1, padding=padding)  # [1, H', W', 1]
            print("fine tf")

            print("inizio custom")
            custom_out = conv.conv_sparse_fast8_padding_v2(X, Q_sp, padding=padding)
            print("fine custom")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-5)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)
    ################################################################

    # versione con padding & stride
    def test_conv_sparse_fast8_padding_v2_stride(self):
        SEED = 2
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        num_tests = 10000
        max_size = 244
        min_size = 50
        min_k = 2
        max_k = 7
        num_channels_in = 1
        num_channels_out = 10
        N_max = 29
        N_min = 2
        stride_min = 1
        stride_max = 7

        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))
            padding = "VALID" if random.randint(0, 1) == 0 else "SAME"
            stride = np.random.randint(stride_min, stride_max + 1)
            #padding= "SAME"
            #stride = 2

            print(padding)
            print(stride)
            print(K)
            print()



            sparsity = 0
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)

            #print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=stride, padding=padding)
            #print("fine tf")

            #print("inizio custom")
            custom_out = conv.conv_sparse_fast8_padding_v2_stride(X, Q_sp, padding=padding, stride=stride)
            #print("fine custom")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-4)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)


    ################################################################

    def test_time(self):
        num_tests = 10
        max_size = 32
        min_size = 32
        min_k = 3
        max_k = 3
        num_channels_in = 15
        num_channels_out = 10
        N_max = 10
        N_min = 10

        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            H = np.random.randint(min_size, max_size + 1)
            W = np.random.randint(min_size, max_size + 1)
            K = np.random.randint(min_k, min(max_k, H, W) + 1)
            X = tf.constant(np.random.randn(N, H, W, num_channels_in).astype(np.float32))

            sparsity = 0.9
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)
            Q_sp = tf.sparse.from_dense(Q_dense)



            start_time = time.perf_counter()
            custom_out7 = conv.conv_sparse_fast_D8(X, Q_sp)
            print(time.perf_counter() - start_time, "seconds")

            start_time = time.perf_counter()
            custom_out8 = conv.conv_sparse_fast8(X, Q_sp)
            print(time.perf_counter() - start_time, "seconds")



            print()










if __name__ == '__main__':
    unittest.main()
