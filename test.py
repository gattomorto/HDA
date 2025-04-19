import unittest
import v4
import numpy as np
import tensorflow as tf
'''
    remaining * true_prop_contributions[0]
    np.float64(0.5974581465125084)
    5 * true_prop_contributions[0]
    np.float32(0.5974581)
    5 * true_prop_contributions[1]
    np.float32(0.6830293)
    5 * true_prop_contributions[2]
    np.float32(3.719513)
    true_prop_contributions
    [np.float32(0.11949163), np.float32(0.13660586), np.float32(0.74390256)]
    sum(true_prop_contributions)
    np.float32(1.0)
    nota che se usi round() piuttosto di int() vengono allocati 1+1+4 = 6 che è addirittura piu grande di 5 
    '''
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





if __name__ == '__main__':
    unittest.main()
