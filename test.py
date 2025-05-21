import unittest
import psutil

import funzioni
import v4
import numpy as np
import tensorflow as tf
import conv
import time
import random
import utils
import gc


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

    def test_conv2d(self):
        SEED = 0
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        num_tests = 10000
        max_size = 244
        min_size = 50
        num_channels_in = 1
        num_channels_out = 10
        N_max = 29
        N_min = 2
        stride_min = 1
        stride_max = 5
        max_K = 7

        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            HW = np.random.randint(min_size, max_size + 1)
            X = tf.constant(np.random.randn(N, HW, HW, num_channels_in).astype(np.float32))
            padding = "VALID" if random.randint(0, 1) == 0 else "SAME"
            stride = np.random.randint(stride_min, stride_max + 1)
            #K = 1 if random.randint(0, 1) == 0 else 3
            K = np.random.randint(1, min(max_K, HW) + 1)  # Ensure kernel isn't larger than either dimension


            sparsity = 0.9
            '''Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)'''

            Q_sp = funzioni.SparseTensor((K, K, num_channels_in, num_channels_out), sparsity=sparsity)
            Q_dense = Q_sp.to_tf_dense()


            print("inizio tf")
            tf_out = tf.nn.conv2d(X, Q_dense, strides=stride, padding=padding)
            print("fine tf")

            print("inizio mio")
            #custom_out = conv.classic_conv2d_4(X, Q_sp, padding=padding, stride=stride)
            custom_out = conv.sparse_to_dense_conv2d(X, Q_sp, stride=stride, padding=padding)
            print("fine mio")

            close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-4)

            if not close:
                print(tf.reduce_max(tf.abs(custom_out - tf_out)))

            self.assertTrue(close)

    def test_time(self):
        SEED = 2
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        num_tests = 100
        max_size = 244
        min_size = 64

        num_channels_in = 128
        num_channels_out = 64
        N_max = 128
        N_min = 64
        stride_min = 1
        stride_max = 2
        max_K = 3


        for _ in range(num_tests):
            N = np.random.randint(N_min,N_max+1)
            HW = np.random.randint(min_size, max_size + 1)
            #W = np.random.randint(min_size, max_size + 1)
            #K = np.random.randint(min_k, min(max_k, H, W) + 1)  # Ensure kernel isn't larger than either dimension
            X = tf.constant(np.random.randn(N, HW, HW, num_channels_in).astype(np.float32))
            padding = "VALID" if random.randint(0, 1) == 0 else "SAME"
            stride = np.random.randint(stride_min, stride_max + 1)
            K = 1 if random.randint(0, 1) == 0 else 3
            #K = np.random.randint(1, min(max_K, HW) + 1)  # Ensure kernel isn't larger than either dimension


            sparsity = 0.9
            Q_dense = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity  # Random mask for sparsity
            Q_dense = tf.convert_to_tensor(Q_dense * mask)  # Apply sparsity
            Q_sp = tf.sparse.from_dense(Q_dense)
            Q_dense = tf.convert_to_tensor(Q_dense)

            Q_sp = funzioni.SparseTensor((K, K, num_channels_in, num_channels_out), sparsity=sparsity)
            Q_dense = Q_sp.to_tf_dense()


            start_time = time.perf_counter()
            _ = tf.nn.conv2d(X, Q_dense, strides=stride, padding=padding)
            print(time.perf_counter() - start_time, "seconds")

            start_time = time.perf_counter()
            #Q_d = tf.sparse.to_dense(Q_sp)
            #_ = tf.nn.conv2d(X, Q_d, strides=stride, padding=padding)
            #_ = conv.conv_sparse_fast8_padding_v2_stride(X, Q_sp, stride=stride, padding=padding)
            _ = conv.sparse_to_dense_conv2d(X,Q_sp,stride=stride,padding=padding)
            print(time.perf_counter() - start_time, "seconds")

            print()

    def test_mem(self):
        SEED = 3
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        num_tests = 10000
        max_size = 64
        min_size = 64
        min_k = 1
        max_k = 3
        num_channels_in = 64
        num_channels_out = 128
        N_max = 32
        N_min = 8
        stride_min = 1
        stride_max = 2

        for _ in range(num_tests):
            N = np.random.randint(N_min, N_max + 1)
            HW = np.random.randint(min_size, max_size + 1)
            K = 1 if random.randint(0, 1) == 0 else 3


            # Prepare input tensors
            X = tf.constant(np.random.randn(N, HW, HW, num_channels_in).astype(np.float32))
            padding = "VALID" if random.randint(0, 1) == 0 else "SAME"
            stride = np.random.randint(stride_min, stride_max + 1)

            sparsity = 0.9
            Q_dense_np = np.random.randn(K, K, num_channels_in, num_channels_out).astype(np.float32)
            mask = np.random.rand(K, K, num_channels_in, num_channels_out) > sparsity
            Q_dense = tf.convert_to_tensor(Q_dense_np * mask)
            Q_sp = tf.sparse.from_dense(Q_dense)
            #Q_dense = tf.convert_to_tensor(Q_dense)


            #_ = conv.direct_sparse_conv2d_6(X, Q_sp, padding=padding, stride=stride)
            _ = tf.nn.conv2d(X, Q_dense, strides=stride, padding=padding)

            #Q_d = tf.sparse.to_dense(Q_sp)
            #_ = tf.nn.conv2d(X, Q_d, strides=stride, padding=padding)




            print(utils.mem_usage())#peak memory


class Varie(unittest.TestCase):
    def test_random_sparse_indices3(self):
        '''
        si testa che gli indici siano in row major
        '''
        test_shapes = [
            [10, 10, 10],  # Original case
            [5, 5, 5, 5],  # 4D tensor
            [5, 53, 5, 53],  # 4D tensor
            [1, 1, 1, 1],  # 4D tensor
            [4, 500, 3, 4],  # 4D tensor
            [20, 20],  # 2D tensor
            [100,2],  # 1D tensor
            [8, 15, 3],  # Irregular 3D
            [2, 2, 2, 2, 2],  # 5D tensor
        ]

        density = 0.5  # You can also make this parameterized if needed

        for shape in test_shapes:
            # Run the original test logic for each shape
            indices, nnz = v4.random_sparse_indices3(shape, density)
            indices_np = indices.numpy()

            flat = np.ravel_multi_index(indices_np.T, shape, order='C')
            self.assertTrue(np.all(flat[:-1] <= flat[1:]))




if __name__ == '__main__':
    unittest.main()
