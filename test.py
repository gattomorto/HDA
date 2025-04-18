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

    def test_F(self):
        pass
        #testare quando momentum sono a zero


if __name__ == '__main__':
    unittest.main()
