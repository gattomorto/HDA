import unittest
import psutil
import v4
import numpy as np
import tensorflow as tf
import conv
import time
import random
import utils
import gc
import models

#non eserguire
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

            Q_sp = models.SparseTensor((K, K, num_channels_in, num_channels_out), sparsity=sparsity)
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

            Q_sp = models.SparseTensor((K, K, num_channels_in, num_channels_out), sparsity=sparsity)
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
    def test_prune_and_regrow_stats0(self):
        tensor = models.SparseTensor(tf.constant([[0.1, -0.3],
                                                  [0.01,  0.4]]))

        tensor.prune(rho=0.5)

        self.assertEqual(tensor.num_pruned,2)
        self.assertEqual(tensor.num_active_weights(),2)
        self.assertEqual(tensor.num_inactive_weights(),2)
        self.assertFalse(tensor.is_saturated())
        self.assertEqual(tensor.num_weights(),4)
        tensor.reset_prune_and_regrow_stats()
        self.assertEqual(tensor.num_pruned,0)

    def test_prune_and_regrow_stats1(self):
        '''
        '''
        # 3x2x4 = 24
        tensor =  models.SparseTensor(tf.constant([
            [
                [3.1, 1.2, -2.5, 7.8],
                [4.2, 9.1, 1.3, -0.7]
            ],
            [
                [6.7, -3.3, 2.3, 8.8],
                [0.5, 2.0, 5.5, -1.1]
            ],
            [
                [7.7, -4.5, 3.4, 1.0],
                [2.2, -6.6, 3.0, 4.4]
            ]
        ], dtype=tf.float32))

        tensor.prune(rho=0.1)
        tensor.rowmajor_reorder()
        tensor_check =  models.SparseTensor(tf.constant([
            [
                [3.1, 1.2, -2.5, 7.8],
                [4.2, 9.1, 1.3, 0]
            ],
            [
                [6.7, -3.3, 2.3, 8.8],
                [0, 2.0, 5.5, -1.1]
            ],
            [
                [7.7, -4.5, 3.4, 0],
                [2.2, -6.6, 3.0, 4.4]
            ]
        ], dtype=tf.float32))


        self.assertEqual(tensor.num_pruned,3)
        self.assertEqual(tensor.num_active_weights(),24-3)
        self.assertEqual(tensor.num_inactive_weights(),3)
        self.assertFalse(tensor.is_saturated())
        self.assertEqual(tensor.num_weights(),24)
        tensor.reset_prune_and_regrow_stats()
        self.assertEqual(tensor.num_pruned,0)

    def test_prune_and_regrow_stats2(self):
        '''
        '''

        # se tutti i valori sono interi, non funziona
        tensor = models.SparseTensor(tf.constant([[0., -0],
                                                  [0,   0]]))

        tensor.prune(rho=0.5)
        tensor.rowmajor_reorder()
        tensor_check = models.SparseTensor(tf.constant([[0., 0],
                                                        [0,  0]]))


        self.assertEqual(tensor.num_pruned,0)
        self.assertEqual(tensor.num_active_weights(),0)
        self.assertEqual(tensor.num_inactive_weights(),4)
        self.assertFalse(tensor.is_saturated())
        self.assertEqual(tensor.num_weights(),4)
        tensor.reset_prune_and_regrow_stats()
        self.assertEqual(tensor.num_pruned,0)

    def test_prune_and_regrow_stats3(self):
        '''
        '''
        tensor = models.SparseTensor(tf.constant([[0.1, -0.3],
                                                  [0.01, 0.4]]))

        num_pruned = tensor.prune(rho=0.0)
        tensor.rowmajor_reorder()
        tensor_check = models.SparseTensor(tf.constant([[0.1, -0.3],
                                                        [0.01, 0.4]]))

        self.assertEqual(tensor.num_pruned,0)
        self.assertEqual(tensor.num_active_weights(),4)
        self.assertEqual(tensor.num_inactive_weights(),0)
        self.assertTrue(tensor.is_saturated())
        self.assertEqual(tensor.num_weights(),4)
        tensor.reset_prune_and_regrow_stats()
        self.assertEqual(tensor.num_pruned,0)

class Prune(unittest.TestCase):
    # ------------------------- Prune Layer --------------------------------
    def test_A(self):
        '''
        rho = 0.5
        '''
        tensor = models.SparseTensor(tf.constant([[0.1, -0.3],
                                                  [0.01,  0.4]]))

        num_pruned = tensor.prune(rho=0.5)

        # devono rimanere i due piu grandi in valore assoluto
        tensor_check = models.SparseTensor(tf.constant([[0, -0.3],
                                                        [0,  0.4]]))
        self.assertTrue(tensor_check==tensor)
        self.assertTrue(num_pruned==2)

    def test_B(self):
        '''
        rho = 1
        '''

        tensor = models.SparseTensor(tf.constant([[0.1,  -0.3],
                                                  [0.01,  0.4]]))
        num_pruned = tensor.prune(rho=1)
        tensor_check = models.SparseTensor(tf.constant([[.0, .0],
                                                        [.0, .0]]))

        self.assertTrue(tensor_check==tensor)
        self.assertTrue(num_pruned==4)

    def test_C(self):
        '''
        tutti valori uguali
        '''
        tensor = models.SparseTensor(tf.constant([[0.1, 0.1],
                                                  [0.1, 0.1]]))
        num_pruned = tensor.prune(rho=0.5)
        self.assertTrue(np.array_equal(np.array([0.1, 0.1], dtype=np.float32), tensor.values.numpy()))
        self.assertTrue(num_pruned == 2)

    def test_D(self):
        '''
        rho = 0.25
        '''
        tensor = models.SparseTensor(tf.constant([[0.1, -0.3],
                                                 [0.01, 0.4]]))

        num_pruned = tensor.prune(rho=0.25)
        tensor_check = models.SparseTensor(tf.constant([[0.1, -0.3],
                                                        [0,    0.4]]))
        self.assertTrue(tensor_check == tensor)
        self.assertTrue(num_pruned == 1)

    def test_E(self):
        '''
        rho = 0
        '''
        tensor = models.SparseTensor(tf.constant([[0.1, -0.3],
                                                  [0.01, 0.4]]))

        num_pruned = tensor.prune(rho=0.0)
        # ci dovrebbe essere reorder() anche sugli altri, solo che per caso ritorna gia valori ordinati
        tensor.rowmajor_reorder()
        tensor_check = models.SparseTensor(tf.constant([[0.1, -0.3],
                                                        [0.01, 0.4]]))
        self.assertTrue(tensor_check == tensor)
        self.assertTrue(num_pruned == 0)

    def test_F(self):
        '''
        3d tensor
        '''
        # 3x2x4 = 24
        tensor =  models.SparseTensor(tf.constant([
            [
                [3.1, 1.2, -2.5, 7.8],
                [4.2, 9.1, 1.3, -0.7]
            ],
            [
                [6.7, -3.3, 2.3, 8.8],
                [0.5, 2.0, 5.5, -1.1]
            ],
            [
                [7.7, -4.5, 3.4, 1.0],
                [2.2, -6.6, 3.0, 4.4]
            ]
        ], dtype=tf.float32))

        # 24*0.1 = 2.4
        num_pruned = tensor.prune(rho=0.1)
        tensor.rowmajor_reorder()

        tensor_check =  models.SparseTensor(tf.constant([
            [
                [3.1, 1.2, -2.5, 7.8],
                [4.2, 9.1, 1.3, 0]
            ],
            [
                [6.7, -3.3, 2.3, 8.8],
                [0, 2.0, 5.5, -1.1]
            ],
            [
                [7.7, -4.5, 3.4, 0],
                [2.2, -6.6, 3.0, 4.4]
            ]
        ], dtype=tf.float32))

        self.assertTrue(tensor_check == tensor)
        self.assertTrue(num_pruned == 3)

    def test_G(self):
        '''
        prune an empty tensor
        '''

        # se tutti i valori sono interi, non funziona
        tensor = models.SparseTensor(tf.constant([[0., -0],
                                                  [0,   0]]))

        num_pruned = tensor.prune(rho=0.5)
        tensor.rowmajor_reorder()
        tensor_check = models.SparseTensor(tf.constant([[0., 0],
                                                        [0,  0]]))
        self.assertTrue(tensor_check == tensor)
        self.assertTrue(num_pruned == 0)

    def test_I(self):
        '''
        3d tensor
        '''
        # 3x2x4 = 24
        tensor =  models.SparseTensor(tf.constant([
            [
                [0, 2.3, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 2.0, -1, -0]
            ],
            [
                [0, 0, -3.4, 0],
                [0, 0, 0, 0]
            ]
        ], dtype=tf.float32))

        num_pruned = tensor.prune(rho=0.5)
        tensor.rowmajor_reorder()

        tensor_check =  models.SparseTensor(tf.constant([
            [
                [0, 2.3, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, -0]
            ],
            [
                [0, 0, -3.4, 0],
                [0, 0, 0, 0]
            ]
        ], dtype=tf.float32))

        self.assertTrue(tensor_check == tensor)
        self.assertTrue(num_pruned == 2)


    # ------------------------- Prune Model --------------------------------

    def test_H(self):
        model = models.ResNet50_sparse2(sparsity=0.5)
        del model.stage2
        del model.stage3
        del model.stage4
        del model.stage5
        del model.conv1_w
        del model.fc_w

        #tot = 3*2*4 = 24 -- 3 posizioni libere -- 21 occupate
        model.W1 = models.SparseTensor([3,2,4],0.1,name="W1")

        #tot = 216 -- 22 posizioni libere -- 194 occupate
        model.W2 = models.SparseTensor([3,2,4,9],0.1,name = "W2")

        #tot = 600 -- 60 posizioni libere -- 540 occupate
        model.W3 = models.SparseTensor([30,20],0.1,name = "W3")

        model.sparse_tensors = model._collect_sparse_tensors()

        tot_pruned = model.prune(rho = 0.6)
        # W1: 21*0.6 = 12.6 -> 13 pruned
        # W2: 194*0.6 = 116.39 -> 117 pruned
        # W3: 540*0.6 = 324 -> 324 pruned
        # tot = 454

        self.assertEqual(tot_pruned,454)
        self.assertEqual(model.W1.num_inactive_weights(),3+13) # ce n'erano 3 libere e se ne sono liberate 3
        self.assertEqual(model.W2.num_inactive_weights(),22+117)
        self.assertEqual(model.W3.num_inactive_weights(),60+324)
        self.assertEqual(model.num_pruned(),454)
        self.assertEqual(model.W1.num_pruned,13)
        self.assertEqual(model.W2.num_pruned,117)
        self.assertEqual(model.W3.num_pruned,324)
        self.assertEqual(model.W1.num_active_weights(),21-13)
        self.assertEqual(model.W2.num_active_weights(),194-117)
        self.assertEqual(model.W3.num_active_weights(),540-324)

        print(model.prune_summary())

class Regrow(unittest.TestCase):
    @staticmethod
    def sparse_indices_set_difference(A_indices, B_indices):
        """
        Returns the indices in A_indices that are not in B_indices.

        Parameters:
            A_indices: tf.Tensor of shape [n, rank]
            B_indices: tf.Tensor of shape [m, rank]

        Returns:
            tf.Tensor of shape [k, rank], the set difference A_indices \ B_indices
        """
        A_exp = tf.expand_dims(A_indices, axis=1)  # [n, 1, rank]
        B_exp = tf.expand_dims(B_indices, axis=0)  # [1, m, rank]

        # Compare all combinations
        equality_matrix = tf.reduce_all(tf.equal(A_exp, B_exp), axis=-1)  # [n, m]

        # Does each A_index match any B_index?
        any_match = tf.reduce_any(equality_matrix, axis=1)  # [n]

        # Keep only A_indices that are not in B
        difference = tf.boolean_mask(A_indices, ~any_match)

        return difference

    @staticmethod
    def sparse_indices_is_subset(indicesA, indicesB):
        """
        Returns True if all elements in indicesA are also present in indicesB.

        Args:
            indicesA: tf.Tensor of shape [n, rank]
            indicesB: tf.Tensor of shape [m, rank]

        Returns:
            tf.Tensor of shape [], dtype tf.bool
        """
        A_exp = tf.expand_dims(indicesA, axis=1)  # [n, 1, rank]
        B_exp = tf.expand_dims(indicesB, axis=0)  # [1, m, rank]

        equal_matrix = tf.reduce_all(tf.equal(A_exp, B_exp), axis=-1)  # [n, m]
        match_found = tf.reduce_any(equal_matrix, axis=1)  # [n]

        return tf.reduce_all(match_found)  # scalar bool: True if all A in B

    @staticmethod
    def sparse_indices_set_equal(indicesA, indicesB):
        return tf.logical_and(
            Regrow.sparse_indices_is_subset(indicesA, indicesB),
            Regrow.sparse_indices_is_subset(indicesB, indicesA)
        )

    @staticmethod
    def sparse_tensor_indices_equal_to_val(tens, ind, val):
        """
        Checks that all indices in `ind` exist in `tens`, and that their corresponding
        values are all equal to `val`. Raises an error if any index in `ind` is missing.

        Args:
            tens: instance of SparseTensor (custom class)
            ind: tf.Tensor of shape [k, rank], dtype int64
            val: scalar value (same dtype as tens.values)

        Returns:
            tf.Tensor scalar boolean
        """
        ind_exp = tf.expand_dims(ind, axis=1)  # [k, 1, rank]
        tens_indices_exp = tf.expand_dims(tens.indices, 0)  # [1, n, rank]

        equal_matrix = tf.reduce_all(tf.equal(ind_exp, tens_indices_exp), axis=-1)  # [k, n]
        match_found = tf.reduce_any(equal_matrix, axis=1)  # [k] boolean

        # Assert all indices in `ind` are present in tens.indices
        tf.debugging.assert_equal(
            tf.reduce_all(match_found),
            True,
            message="Some indices in `ind` are not present in the SparseTensor."
        )

        # Get positions in tens.values that match
        matched_indices_in_values = tf.argmax(tf.cast(equal_matrix, tf.int32), axis=1)  # [k]
        matched_values = tf.gather(tens.values, matched_indices_in_values)  # [k]

        # Check if all matched values equal val
        return tf.reduce_all(tf.equal(matched_values, val))

    # ----------------------------- Regrow Layer ------------------------------------------

    def test_A(self):
        '''
        request = 1
        '''

        request = 1

        #3x2x4
        # 4 spazi disponibili
        tensor = models.SparseTensor(tf.constant([
            [
                [0, 0, -2.5, 7.8],
                [4.2, 9.1, 1.3, -0.7]
            ],
            [
                [6.7, -3.3, 0, 8.8],
                [0.5, 2.0, 5.5, -1.1]
            ],
            [
                [7.7, 0, 3.4, 1.0],
                [2.2, -6.6, 3.0, 4.4]
            ]
        ], dtype=tf.float32))

        indices_before = tensor.indices

        regrown = tensor.regrow(requested_growth=request)
        indices_after = tensor.indices

        # diff sono quelli aggiunti
        diff = self.sparse_indices_set_difference(indices_after,indices_before)
        self.assertEqual(diff.shape[0],request) #ci deve essere un elemento perchè è stato richiesto uno
        self.assertEqual(regrown,request)

        available = tf.constant([ [0,0,0],[0,0,1],[1,0,2],[2,0,1] ],dtype = tf.int64) # il nuovo indice deve essere uno tra quelli liberi
        self.assertTrue(self.sparse_indices_is_subset(diff,available))

        # controllo che tensor nelle nuove posizioni sia 1.7243673
        self.assertTrue(self.sparse_tensor_indices_equal_to_val(tensor, diff, 0))

    def test_B(self):
        '''
        request = 2
        '''

        request = 2

        #3x2x4
        # 4 spazi disponibili
        tensor = models.SparseTensor(tf.constant([
            [
                [0, 0, -2.5, 7.8],
                [4.2, 9.1, 1.3, -0.7]
            ],
            [
                [6.7, -3.3, 0, 8.8],
                [0.5, 2.0, 5.5, -1.1]
            ],
            [
                [7.7, 0, 3.4, 1.0],
                [2.2, -6.6, 3.0, 4.4]
            ]
        ], dtype=tf.float32))

        indices_before = tensor.indices

        regrown = tensor.regrow(requested_growth=request)
        indices_after = tensor.indices

        diff = self.sparse_indices_set_difference(indices_after,indices_before)
        self.assertEqual(diff.shape[0],request)
        self.assertEqual(regrown,request)

        available = tf.constant([ [0,0,0],[0,0,1],[1,0,2],[2,0,1] ],dtype = tf.int64)

        self.assertTrue(self.sparse_indices_is_subset(diff,available))
        self.assertTrue(self.sparse_tensor_indices_equal_to_val(tensor, diff, 0))
    def test_C(self):
        '''
        request = 3
        '''

        request = 3

        #3x2x4
        # 4 spazi disponibili
        tensor = models.SparseTensor(tf.constant([
            [
                [0, 0, -2.5, 7.8],
                [4.2, 9.1, 1.3, -0.7]
            ],
            [
                [6.7, -3.3, 0, 8.8],
                [0.5, 2.0, 5.5, -1.1]
            ],
            [
                [7.7, 0, 3.4, 1.0],
                [2.2, -6.6, 3.0, 4.4]
            ]
        ], dtype=tf.float32))

        indices_before = tensor.indices

        regrown = tensor.regrow(requested_growth=request)
        indices_after = tensor.indices

        diff = self.sparse_indices_set_difference(indices_after,indices_before)
        self.assertEqual(diff.shape[0],request)
        self.assertEqual(regrown,request)

        available = tf.constant([ [0,0,0],[0,0,1],[1,0,2],[2,0,1] ],dtype = tf.int64)

        self.assertTrue(self.sparse_indices_is_subset(diff,available))
        self.assertTrue(self.sparse_tensor_indices_equal_to_val(tensor, diff, 0))
    def test_D(self):
        '''
        request = max_available
        '''

        max_available = 4
        request = max_available

        #3x2x4
        # 4 spazi disponibili
        tensor = models.SparseTensor(tf.constant([
            [
                [0, 0, -2.5, 7.8],
                [4.2, 9.1, 1.3, -0.7]
            ],
            [
                [6.7, -3.3, 0, 8.8],
                [0.5, 2.0, 5.5, -1.1]
            ],
            [
                [7.7, 0, 3.4, 1.0],
                [2.2, -6.6, 3.0, 4.4]
            ]
        ], dtype=tf.float32))

        indices_before = tensor.indices

        regrown = tensor.regrow(requested_growth=request)
        indices_after = tensor.indices

        diff = self.sparse_indices_set_difference(indices_after,indices_before)
        self.assertEqual(diff.shape[0],request)
        self.assertEqual(regrown,request)

        available = tf.constant([ [0,0,0],[0,0,1],[1,0,2],[2,0,1] ],dtype = tf.int64)


        #self.assertTrue(self.sparse_indices_is_subset(diff,available))
        self.assertTrue(self.sparse_indices_set_equal(diff,available))
        self.assertTrue(self.sparse_tensor_indices_equal_to_val(tensor, diff, 0))
    def test_E(self):
        '''
        request = max_available + 10
        '''

        max_available = 4
        request = max_available + 10

        #3x2x4
        # 4 spazi disponibili
        tensor = models.SparseTensor(tf.constant([
            [
                [0, 0, -2.5, 7.8],
                [4.2, 9.1, 1.3, -0.7]
            ],
            [
                [6.7, -3.3, 0, 8.8],
                [0.5, 2.0, 5.5, -1.1]
            ],
            [
                [7.7, 0, 3.4, 1.0],
                [2.2, -6.6, 3.0, 4.4]
            ]
        ], dtype=tf.float32))

        indices_before = tensor.indices

        regrown = tensor.regrow(requested_growth=request)
        indices_after = tensor.indices

        diff = self.sparse_indices_set_difference(indices_after,indices_before)
        self.assertEqual(diff.shape[0],max_available)
        self.assertEqual(regrown,max_available)

        available = tf.constant([ [0,0,0],[0,0,1],[1,0,2],[2,0,1] ], dtype = tf.int64)

        #self.assertTrue(self.sparse_indices_is_subset(diff,available))
        self.assertTrue(self.sparse_indices_set_equal(diff,available))
        self.assertTrue(self.sparse_tensor_indices_equal_to_val(tensor, diff, 0))
    def test_F(self):
        '''
        0 available
        '''

        max_available = 0
        request = max_available + 10

        #3x2x4
        # 0 spazi disponibili
        tensor = models.SparseTensor(tf.constant([
            [
                [0.1, 0.2, -2.5, 7.8],
                [4.2, 9.1, 1.3, -0.7]
            ],
            [
                [6.7, -3.3, -0.9, 8.8],
                [0.5, 2.0, 5.5, -1.1]
            ],
            [
                [7.7, 0.01, 3.4, 1.0],
                [2.2, -6.6, 3.0, 4.4]
            ]
        ], dtype=tf.float32))

        indices_before = tensor.indices

        regrown = tensor.regrow(requested_growth=request)
        indices_after = tensor.indices

        diff = self.sparse_indices_set_difference(indices_after,indices_before)
        self.assertEqual(diff.shape[0],max_available)
        self.assertEqual(regrown,max_available)

        self.assertTrue(self.sparse_indices_set_equal(indices_before,indices_after))
    def test_G(self):
        '''
        request = 0
        '''

        max_available = 4
        request = 0

        #3x2x4
        # 4 spazi disponibili
        tensor = models.SparseTensor(tf.constant([
            [
                [0, 0, -2.5, 7.8],
                [4.2, 9.1, 1.3, -0.7]
            ],
            [
                [6.7, -3.3, 0, 8.8],
                [0.5, 2.0, 5.5, -1.1]
            ],
            [
                [7.7, 0, 3.4, 1.0],
                [2.2, -6.6, 3.0, 4.4]
            ]
        ], dtype=tf.float32))

        indices_before = tensor.indices

        regrown = tensor.regrow(requested_growth=request)
        indices_after = tensor.indices

        self.assertEqual(regrown,0)
        self.assertTrue(self.sparse_indices_set_equal(indices_before,indices_after))
    # ----------------------------- Regrow Model ------------------------------------------
    def test_H(self):
        '''
        base
        '''
        model = models.ResNet50_sparse2(sparsity=0.5)
        del model.stage2
        del model.stage3
        del model.stage4
        del model.stage5
        del model.conv1_w
        del model.fc_w

        #tot = 3*2*4 = 24 -- 3 posizioni libere -- 21 occupate
        model.W1 = models.SparseTensor([3,2,4],0.1,name="W1")
        model.W1.mean_momentum = 5

        #tot = 216 -- 22 posizioni liberec -- 194 occupate
        model.W2 = models.SparseTensor([3,2,4,9],0.1,name = "W2")
        model.W2.mean_momentum = 4

        #tot = 600 -- 60 posizioni libere -- 540 occupate
        model.W3 = models.SparseTensor([30,20],0.1,name = "W3")
        model.W3.mean_momentum = 1

        model.sparse_tensors = model._collect_sparse_tensors()

        #tot liberi = 85
        model.regrow(to_regrow=80)

        self.assertEqual(model.W1.num_inactive_weights(),0)
        self.assertEqual(model.W2.num_inactive_weights(),0)
        self.assertEqual(model.W3.num_inactive_weights(),5)
        self.assertEqual(model.num_regrown(),80)
        self.assertEqual(model.num_pruned(),0)
        self.assertEqual(model.num_active_weights(),21+194+540+80)
        self.assertEqual(model.num_inactive_weights(),5)

        print(model.regrow_summary())

    def test_I(self):
        '''

        '''
        model = models.ResNet50_sparse2(sparsity=0.5)
        del model.stage2
        del model.stage3
        del model.stage4
        del model.stage5
        del model.conv1_w
        del model.fc_w

        #tot = 3*2*4 = 24 -- 3 posizioni libere
        model.W1 = models.SparseTensor([3,2,4],0.1,name="W1")
        model.W1.mean_momentum = 5

        #tot = 216 -- 22 posizioni libere
        model.W2 = models.SparseTensor([3,2,4,9],0.1,name = "W2")
        model.W2.mean_momentum = 4

        #tot = 600 -- 60 posizioni libere
        model.W3 = models.SparseTensor([30,20],0.1,name = "W3")
        model.W3.mean_momentum = 20

        model.sparse_tensors = model._collect_sparse_tensors()

        #tot liberi = 85
        model.regrow(to_regrow=81)

        self.assertEqual(model.W1.num_inactive_weights(),0)
        self.assertEqual(model.W2.num_inactive_weights(),4)
        self.assertEqual(model.W3.num_inactive_weights(),0)

        print(model.regrow_summary())

    def test_L(self):
        '''
        qui il debito è distribuito su due rimanenti
        '''
        model = models.ResNet50_sparse2(sparsity=0.5)
        del model.stage2
        del model.stage3
        del model.stage4
        del model.stage5
        del model.conv1_w
        del model.fc_w

        # 3 posizioni libere
        model.W1 = models.SparseTensor([3,2,4],0.1,name="W1")
        model.W1.mean_momentum = 1

        # 65 posizioni libere
        model.W2 = models.SparseTensor([3,2,4,9,3],0.1,name = "W2")
        model.W2.mean_momentum = 5

        # 60 posizioni libere
        model.W3 = models.SparseTensor([30,20],0.1,name = "W3")
        model.W3.mean_momentum = 1

        model.sparse_tensors = model._collect_sparse_tensors()

        model.regrow(to_regrow=81)

        self.assertEqual(model.W1.num_inactive_weights(),0)
        self.assertEqual(model.W2.num_inactive_weights(),0)
        self.assertEqual(model.W3.num_inactive_weights(),47)

        print(model.regrow_summary())

    def test_M(self):
        '''
        request = max available
        '''
        model = models.ResNet50_sparse2(sparsity=0.5)
        del model.stage2
        del model.stage3
        del model.stage4
        del model.stage5
        del model.conv1_w
        del model.fc_w

        # 3 posizioni libere
        model.W1 = models.SparseTensor([3,2,4],0.1,name="W1")
        model.W1.mean_momentum = 1

        # 65 posizioni libere
        model.W2 = models.SparseTensor([3,2,4,9,3],0.1,name = "W2")
        model.W2.mean_momentum = 50

        # 60 posizioni libere
        model.W3 = models.SparseTensor([30,20],0.1,name = "W3")
        model.W3.mean_momentum = 1

        model.sparse_tensors = model._collect_sparse_tensors()

        model.regrow(to_regrow=128)

        self.assertEqual(model.W1.num_inactive_weights(),0)
        self.assertEqual(model.W2.num_inactive_weights(),0)
        self.assertEqual(model.W3.num_inactive_weights(),0)

        print(model.regrow_summary())


    def test_N(self):
        '''
        mean_momentum = 0
        '''
        model = models.ResNet50_sparse2(sparsity=0.5)
        del model.stage2
        del model.stage3
        del model.stage4
        del model.stage5
        del model.conv1_w
        del model.fc_w

        # 3 posizioni libere
        model.W1 = models.SparseTensor([3,2,4],0.1,name="W1")
        model.W1.mean_momentum = 0

        # 65 posizioni libere
        model.W2 = models.SparseTensor([3,2,4,9,3],0.1,name = "W2")
        model.W2.mean_momentum = 2

        # 60 posizioni libere
        model.W3 = models.SparseTensor([30,20],0.1,name = "W3")
        model.W3.mean_momentum = 0

        model.sparse_tensors = model._collect_sparse_tensors()

        model.regrow(to_regrow=128)

        self.assertEqual(model.W1.num_inactive_weights(),0)
        self.assertEqual(model.W2.num_inactive_weights(),0)
        self.assertEqual(model.W3.num_inactive_weights(),0)

        print(model.regrow_summary())


    def test_O(self):
        '''
        mean_momentum = 0
        '''
        model = models.ResNet50_sparse2(sparsity=0.5)
        del model.stage2
        del model.stage3
        del model.stage4
        del model.stage5
        del model.conv1_w
        del model.fc_w

        # 48 posizioni libere
        model.W1 = models.SparseTensor([3,2,4,20],0.1,name="W1")
        model.W1.mean_momentum = 0

        # 65 posizioni libere
        model.W2 = models.SparseTensor([3,2,4,9,3],0.1,name = "W2")
        model.W2.mean_momentum = 0

        # 60 posizioni libere
        model.W3 = models.SparseTensor([30,20],0.1,name = "W3")
        model.W3.mean_momentum = 0

        model.sparse_tensors = model._collect_sparse_tensors()

        #tot liberi = 173

        model.regrow(to_regrow=80)

        self.assertEqual(model.W1.num_inactive_weights(),20)
        self.assertEqual(model.W2.num_inactive_weights(),39)
        self.assertEqual(model.W3.num_inactive_weights(),34)

        print(model.regrow_summary())


    def test_P(self):
        '''
        regrow > available
        '''
        model = models.ResNet50_sparse2(sparsity=0.5)
        del model.stage2
        del model.stage3
        del model.stage4
        del model.stage5
        del model.conv1_w
        del model.fc_w

        # 48 posizioni libere
        model.W1 = models.SparseTensor([3,2,4,20],0.1,name="W1")
        model.W1.mean_momentum = 0

        # 65 posizioni libere
        model.W2 = models.SparseTensor([3,2,4,9,3],0.1,name = "W2")
        model.W2.mean_momentum = 0

        # 60 posizioni libere
        model.W3 = models.SparseTensor([30,20],0.1,name = "W3")
        model.W3.mean_momentum = 0

        model.sparse_tensors = model._collect_sparse_tensors()

        #tot liberi = 173

        with self.assertRaises(Exception) as context:
            model.regrow(to_regrow=174)
        self.assertIn("Cannot regrow", str(context.exception))

class Prune_and_Regrow(unittest.TestCase):
    # non eseguire
    def test_A(self):
        model = models.ResNet50_sparse2(sparsity=0.5)
        del model.stage2
        del model.stage3
        del model.stage4
        del model.stage5
        del model.conv1_w
        del model.conv1_b
        del model.bn1
        del model.fc_w
        del model.fc_b

        model.conv1_w = models.SparseTensor([3, 3, 1, 5], 0.5,name = "conv1_w")
        model.conv1_b = tf.Variable(tf.zeros([5]), name="conv1_b")

        model.conv2_w = models.SparseTensor([3, 3, 5, 8], 0.5, name = "conv2_w")
        model.conv2_b = tf.Variable(tf.zeros([8]), name="conv2_b")

        model.fc1_w = models.SparseTensor([7 * 7 * 8, 128], 0.5, name="fc1_w")
        model.fc1_b = tf.Variable(tf.zeros([128]), name="fc1_b")

        model.fc2_w = models.SparseTensor([128, 10], 0.5, name="fc2_w")
        model.fc2_b = tf.Variable(tf.zeros([10]), name="fc2_b")

        model.sparse_tensors = model._collect_sparse_tensors()

        def new_call(self, *args, **kwargs):
            x = args[0]
            x = conv.sparse_to_dense_conv2d(x, self.conv1_w, stride=1, padding='SAME')
            x = tf.nn.bias_add(x, self.conv1_b)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
            x = conv.sparse_to_dense_conv2d(x, self.conv2_w, stride=1, padding='SAME')
            x = tf.nn.bias_add(x, self.conv2_b)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
            x = tf.reshape(x, [x.shape[0], -1])
            x = conv.sparse_to_dense_matmul(x, self.fc1_w) + self.fc1_b
            x = tf.nn.relu(x)
            logits = conv.sparse_to_dense_matmul(x, self.fc2_w) + self.fc2_b
            return logits

        models.ResNet50_sparse2.__call__ = new_call

        (X_tr,y_tr), _ = utils.load_mnist_data(flatten=False)

        SEED = 0
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        t = model.trainable_variables

        prune_and_regrow_stride = 5
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        dataset = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(100)
        it = 0
        for epoch in range(1):
            for step, (x_batch, y_batch) in enumerate(dataset):
                it = it + 1
                with tf.GradientTape() as tape:
                    logits = model(x_batch, training=True)
                    loss = loss_fn(y_batch, logits)
                    print(f"loss: {loss}")

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if it % prune_and_regrow_stride == 0:

                    t = model.trainable_variables

                    print("Prune & Regrow")
                    #print(model)
                    model.prune_and_regrow(0.5,optimizer)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=float(optimizer.learning_rate.numpy()))

                    t = model.trainable_variables
                    pass

    def test_B(self):
        model = models.ResNet50_sparse2(sparsity=0.5)
        del model.stage2
        del model.stage3
        del model.stage4
        del model.stage5
        del model.conv1_w
        del model.fc_w

        #tot = 3*2*4 = 24 -- 3 posizioni libere -- 21 occupate
        model.W1 = models.SparseTensor([3,2,4],0.1,name="W1")

        #tot = 216 -- 22 posizioni libere -- 194 occupate
        model.W2 = models.SparseTensor([3,2,4,9],0.1,name = "W2")

        #tot = 600 -- 60 posizioni libere -- 540 occupate
        model.W3 = models.SparseTensor([30,20],0.1,name = "W3")

        model.sparse_tensors = model._collect_sparse_tensors()

        tot_pruned = model.prune(rho = 0.6)
        # W1: 21*0.6 = 12.6 -> 13 pruned
        # W2: 194*0.6 = 116.39 -> 117 pruned
        # W3: 540*0.6 = 324 -> 324 pruned
        # tot = 454

        self.assertEqual(tot_pruned,454)
        self.assertEqual(model.W1.num_inactive_weights(),3+13) # ce n'erano 3 libere e se ne sono liberate 3
        self.assertEqual(model.W2.num_inactive_weights(),22+117)
        self.assertEqual(model.W3.num_inactive_weights(),60+324)
        self.assertEqual(model.num_pruned(),454)
        self.assertEqual(model.W1.num_pruned,13)
        self.assertEqual(model.W2.num_pruned,117)
        self.assertEqual(model.W3.num_pruned,324)
        self.assertEqual(model.W1.num_active_weights(),21-13)
        self.assertEqual(model.W2.num_active_weights(),194-117)
        self.assertEqual(model.W3.num_active_weights(),540-324)
        print(model.prune_summary())

        model.W1.rowmajor_reorder()
        model.W2.rowmajor_reorder()
        model.W3.rowmajor_reorder()

        model.W1.mean_momentum = 1
        model.W2.mean_momentum = 1
        model.W3.mean_momentum = 1

        model.regrow(tot_pruned)

        self.assertTrue(model.W1.is_saturated())
        self.assertTrue(model.W2.is_saturated())
        self.assertEqual(model.W3.num_regrown,299)
        self.assertEqual(model.num_regrown(),tot_pruned)
        self.assertEqual(model.num_inactive_weights(),384-151-147-1)

        print(model.regrow_summary())

if __name__ == '__main__':
    unittest.main()
