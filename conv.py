import numpy as np
import tensorflow as tf
from functools import reduce
import math

# c_in = 1, c_out = 1, Q dense
def dense_to_sparse(X,Q):
    H = X.shape[0]
    W = X.shape[1]
    K = Q.shape[0]

    def fai_quadro(riga_gen, col_gen):
        indici_quadro = []
        for r in range(K):
            for c in range(K):
                indici_quadro.append([riga_gen * (W - K + 1) + col_gen, riga_gen * W + col_gen + r * W + c])

        return indici_quadro

    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    out_indici = []

    for riga_gen in range(num_rig_gen):
        for col_gen in range(num_col_gen):
            indici_quadro = fai_quadro(riga_gen, col_gen)
            out_indici.extend(indici_quadro)

    indices = tf.constant(out_indici, dtype=tf.int64)
    values = tf.reshape(Q, [-1])  # ensure flat
    values = tf.tile(values, [num_rig_gen * num_col_gen])  # repeat for each patch
    values = tf.cast(values, dtype=tf.float32)
    shape = [num_rig_gen * num_col_gen, H * W]
    S = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    return S

def conv(X, Q):
    H = X.shape[0]
    W = X.shape[1]
    K = Q.shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    S = dense_to_sparse(X,Q)
    X_flat = tf.reshape(X, [-1])
    X_flat = tf.cast(X_flat, tf.float32)
    # nota che result_dense è 1D vec dove ogni posizione rappresenta un quadro
    result_dense = tf.sparse.sparse_dense_matmul(S, tf.expand_dims(X_flat, axis=-1))
    result_dense = tf.reshape(result_dense, [num_rig_gen, num_col_gen])

    return result_dense



# c_in = 2, c_out = 1, Q dense
def dense_to_sparse2(X, Q):
    #sto facendo c_out = 1, c_in = 2
    chan_in = Q.shape[2]
    chan_out = Q.shape[3]

    in_channel_0 = X[0, :, :, 0]
    in_channel_1 = X[0, :, :, 1]

    ker0 = Q[:,:,0,0]
    ker1 = Q[:,:,1,0]

    r1 = dense_to_sparse(in_channel_0, ker0)
    r2 = dense_to_sparse(in_channel_1, ker1)
    stacked = tf.sparse.concat( 1, [r1,r2])
    return stacked

def conv2(X, Q):
    H = X.shape[1]
    W = X.shape[2]
    K = Q.shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    S = dense_to_sparse2(X,Q)

    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # Shape: [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])

    # nota che result_dense è 1D vec dove ogni posizione rappresenta un quadro
    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)

    result_dense = tf.reshape(result_dense, [num_rig_gen, num_col_gen])
    return result_dense


#c_in = x, c_out = 1, Q dense
def dense_to_sparse3(X, Q):
    batch_size, H, W, C_in = X.shape
    _, _, Q_cin, Q_cout = Q.shape
    assert batch_size == 1, "Only batch size 1 is supported"
    assert C_in == Q_cin, "Input channels in X and Q must match"
    assert Q_cout == 1, "Only one output channel supported"

    sparse_list = []
    for i in range(C_in):
        input_channel = X[0, :, :, i]
        kernel_channel = Q[:, :, i, 0]
        sparse_result = dense_to_sparse(input_channel, kernel_channel)  # Must return tf.SparseTensor
        sparse_list.append(sparse_result)

    # concatena a destra per ogni input channel
    stacked = tf.sparse.concat(axis=1, sp_inputs=sparse_list)
    return stacked

def conv3(X, Q):
    H = X.shape[1]
    W = X.shape[2]
    K = Q.shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    S = dense_to_sparse3(X,Q)

    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # Shape: [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])

    # nota che result_dense è 1D vec dove ogni posizione rappresenta un quadro
    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)

    result_dense = tf.reshape(result_dense, [num_rig_gen, num_col_gen])
    return result_dense



#c_in = x, c_out = y, Q dense
def dense_to_sparse4(X, Q):
    batch_size, H, W, C_in = X.shape
    KH, KW, Q_cin, Q_cout = Q.shape
    assert batch_size == 1, "Only batch size 1 is supported"
    assert C_in == Q_cin, "Input channels in X and Q must match"

    per_output_sparse = []

    for out_c in range(Q_cout):
        sparse_list = []
        for in_c in range(C_in):
            input_channel = X[0, :, :, in_c]
            kernel_channel = Q[:, :, in_c, out_c]
            sparse_result = dense_to_sparse(input_channel, kernel_channel)  # Must return tf.SparseTensor
            sparse_list.append(sparse_result)

        stacked = tf.sparse.concat(axis=1, sp_inputs=sparse_list)  # concat across input channels
        per_output_sparse.append(stacked)

    # concat across output channels (rows)
    final_sparse = tf.sparse.concat(axis=0, sp_inputs=per_output_sparse)
    return final_sparse

def conv4(X, Q):
    H = X.shape[1]
    W = X.shape[2]
    K = Q.shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    C_out = Q.shape[3]

    S = dense_to_sparse4(X, Q)  # handles multiple output channels

    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])  # [C*H*W, N]

    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * num_rig_gen * num_col_gen, N]

    # Now reshape to [C_out, num_rig_gen, num_col_gen, N]
    result_reshaped = tf.reshape(result_dense, [C_out, num_rig_gen, num_col_gen, N])

    # Transpose to [N, num_rig_gen, num_col_gen, C_out]
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])
    return result_final


#c_in = x, c_out = y, N = z, Q dense
def dense_to_sparse5(X, Q):
    # questo è uguale alla dense_to_sparse4 dato che la matrice non cambia, se N > 1

    batch_size, H, W, C_in = X.shape
    KH, KW, Q_cin, Q_cout = Q.shape
    assert C_in == Q_cin, "Input channels in X and Q must match"

    per_output_sparse = []

    for out_c in range(Q_cout):
        sparse_list = []
        for in_c in range(C_in):
            try:
                input_channel = X[0, :, :, in_c]
            except:
                pass
            kernel_channel = Q[:, :, in_c, out_c]
            sparse_result = dense_to_sparse(input_channel, kernel_channel)  # Must return tf.SparseTensor
            sparse_list.append(sparse_result)

        stacked = tf.sparse.concat(axis=1, sp_inputs=sparse_list)  # concat across input channels
        per_output_sparse.append(stacked)

    # concat across output channels (rows)
    final_sparse = tf.sparse.concat(axis=0, sp_inputs=per_output_sparse)
    return final_sparse

def conv5(X, Q):
    H = X.shape[1]
    W = X.shape[2]
    K = Q.shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    C_out = Q.shape[3]

    S = dense_to_sparse5(X, Q)  # handles multiple output channels

    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])  # [C*H*W, N]

    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * num_rig_gen * num_col_gen, N]

    # Now reshape to [C_out, num_rig_gen, num_col_gen, N]
    result_reshaped = tf.reshape(result_dense, [C_out, num_rig_gen, num_col_gen, N])

    # Transpose to [N, num_rig_gen, num_col_gen, C_out]
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])
    return result_final

###############################################################################

#c_in = 1, c_out = 1, N = 1, Q sparse
def sparse_to_sparse(X, Q_sparse):

    H = X.shape[0]
    W = X.shape[1]
    K = Q_sparse.dense_shape[0].numpy()

    num_rig_gen = H - K + 1
    num_col_gen = W - K + 1

    # cosi invece di out_indices = [] perchè se Q è tutta = 0, non si riesce a creare S perchè si aspetta indices di dimensioni (x,2)
    out_indices = tf.zeros((0,2), tf.int64)
    out_values = []

    for riga_gen in range(num_rig_gen):
        for col_gen in range(num_col_gen):
            patch_index = riga_gen * num_col_gen + col_gen
            for i in range(tf.shape(Q_sparse.indices)[0]): # i = 0:num_indices
                r, c = Q_sparse.indices[i].numpy()
                value = Q_sparse.values[i].numpy()
                row = patch_index
                col = (riga_gen + r) * W + (col_gen + c)
                #out_indices.append([row, col])
                out_indices = np.vstack([out_indices, [row, col]])

                out_values.append(value)

    indices = tf.constant(out_indices, dtype=tf.int64)
    values = tf.constant(out_values, dtype=tf.float32)
    shape = [num_rig_gen * num_col_gen, H * W]


    S = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

    return S

def conv_sparse(X, Q):
    H = X.shape[0]
    W = X.shape[1]
    K = Q.shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    S = sparse_to_sparse(X,Q)
    X_flat = tf.reshape(X, [-1])
    X_flat = tf.cast(X_flat, tf.float32)
    # nota che result_dense è 1D vec dove ogni posizione rappresenta un quadro
    result_dense = tf.sparse.sparse_dense_matmul(S, tf.expand_dims(X_flat, axis=-1))
    result_dense = tf.reshape(result_dense, [num_rig_gen, num_col_gen])

    return result_dense


#c_in = x, c_out = y, N = z, Q sparse

def sparse_to_sparse2(X, Q_sp):
    def extract_kernel_channel(Q_sparse, in_c, out_c):
        # Q_sparse: tf.sparse.SparseTensor of shape [K, K, C_in, C_out]

        # Step 1: Extract the indices and values
        indices = Q_sparse.indices  # shape [N, 4]
        values = Q_sparse.values

        # Step 2: Create a mask for the desired in_c and out_c
        mask = tf.logical_and(
            tf.equal(indices[:, 2], in_c),
            tf.equal(indices[:, 3], out_c)
        )

        # Step 3: Apply the mask
        new_indices = tf.boolean_mask(indices, mask)
        new_values = tf.boolean_mask(values, mask)

        # Step 4: Drop the in_c and out_c dimensions (since we know them)
        new_indices = new_indices[:, :2]  # keep only the [k_row, k_col] parts

        # Step 5: Create a new sparse tensor of shape [K, K]
        K = Q_sparse.dense_shape[0]  # assuming square kernel
        kernel_channel_sp = tf.sparse.SparseTensor(
            indices=new_indices,
            values=new_values,
            dense_shape=[K, K]
        )
        return kernel_channel_sp

    batch_size, H, W, C_in = X.shape
    #KH, KW, Q_cin, Q_cout = Q.shape
    KH, KW, Q_cin, Q_cout = Q_sp.dense_shape
    assert C_in == Q_cin, "Input channels in X and Q must match"

    per_output_sparse = []

    for out_c in range(Q_cout):
        sparse_list = []
        for in_c in range(C_in):
            input_channel = X[0, :, :, in_c]
            #kernel_channel_sp = Q[:, :, in_c, out_c]
            kernel_channel_sp = extract_kernel_channel(Q_sp, in_c, out_c)
            sparse_result = sparse_to_sparse(input_channel, kernel_channel_sp)  # Must return tf.SparseTensor
            sparse_list.append(sparse_result)

        stacked = tf.sparse.concat(axis=1, sp_inputs=sparse_list)  # concat across input channels
        per_output_sparse.append(stacked)

    final_sparse = tf.sparse.concat(axis=0, sp_inputs=per_output_sparse)
    return final_sparse

def conv_sparse2(X, Q_sp):
    H = X.shape[1]
    W = X.shape[2]
    K = Q_sp.dense_shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    C_out = Q_sp.dense_shape[3]

    S = sparse_to_sparse2(X, Q_sp)  # handles multiple output channels

    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])  # [C*H*W, N]

    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * num_rig_gen * num_col_gen, N]

    # Now reshape to [C_out, num_rig_gen, num_col_gen, N]
    result_reshaped = tf.reshape(result_dense, [C_out, num_rig_gen, num_col_gen, N])

    # Transpose to [N, num_rig_gen, num_col_gen, C_out]
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])
    return result_final

######################################################################
#c_in = 1, c_out = 1, N = 1, Q sparse, fast
# vettorizza sparse_to_sparse
def sparse_to_sparse_fast(X, Q_sparse):
    H= tf.cast(tf.shape(X)[0], dtype=tf.int64)
    W = tf.cast(tf.shape(X)[1], dtype=tf.int64)
    K = Q_sparse.dense_shape[0].numpy()

    num_rig_gen = H - K + 1
    num_col_gen = W - K + 1
    num_patches = num_rig_gen * num_col_gen

    # Q_sparse.indices: [nnz, 2] — with (r, c)
    # Q_sparse.values: [nnz]
    Q_indices = Q_sparse.indices  # shape [nnz, 2]
    Q_values = Q_sparse.values    # shape [nnz]
    nnz = tf.shape(Q_indices)[0]

    # Create patch indices (row indices in the output)
    row_ids = tf.range(num_patches, dtype=tf.int64)  # [num_patches]

    # Compute grid of top-left corners for patches
    r0 = tf.range(num_rig_gen, dtype=tf.int64)
    c0 = tf.range(num_col_gen, dtype=tf.int64)
    rr, cc = tf.meshgrid(r0, c0, indexing='ij')  # shape [num_rig_gen, num_col_gen]

    # Flatten to get list of all patch top-left positions
    patch_base_r = tf.reshape(rr, [-1])  # [num_patches]
    patch_base_c = tf.reshape(cc, [-1])  # [num_patches]

    # Expand to broadcast with the nnz elements
    patch_base_r_tiled = tf.repeat(patch_base_r, repeats=nnz)
    patch_base_c_tiled = tf.repeat(patch_base_c, repeats=nnz)
    patch_ids_tiled = tf.repeat(row_ids, repeats=nnz)

    Q_r = tf.tile(Q_indices[:, 0], [num_patches])
    Q_c = tf.tile(Q_indices[:, 1], [num_patches])
    Q_v = tf.tile(Q_values, [num_patches])

    row_indices = patch_ids_tiled
    col_indices = (patch_base_r_tiled + Q_r) * W + (patch_base_c_tiled + Q_c)

    # Final SparseTensor
    indices = tf.stack([row_indices, col_indices], axis=1)
    values = Q_v
    shape = [num_patches, H * W]

    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
def conv_sparse_fast(X, Q):
    H = tf.shape(X)[0]
    W = tf.shape(X)[1]
    K = tf.shape(Q)[0]  # use tf.shape in case it's dynamic

    num_rig_gen = H - K + 1
    num_col_gen = W - K + 1

    S = sparse_to_sparse_fast(X, Q)

    X_flat = tf.reshape(tf.cast(X, tf.float32), [-1])

    result = tf.sparse.sparse_dense_matmul(S, tf.expand_dims(X_flat, axis=-1))
    return tf.reshape(result, [num_rig_gen, num_col_gen])


#c_in = x, c_out = y, N = z, Q sparse, fast
# usa sparse_to_sparse2 ma all'interno chiama sparse_to_sparse_fast
def sparse_to_sparse_fast2(X, Q_sp):
    def extract_kernel_channel(Q_sparse, in_c, out_c):
        # Q_sparse: tf.sparse.SparseTensor of shape [K, K, C_in, C_out]

        # Step 1: Extract the indices and values
        indices = Q_sparse.indices  # shape [N, 4]
        values = Q_sparse.values

        # Step 2: Create a mask for the desired in_c and out_c
        mask = tf.logical_and(
            tf.equal(indices[:, 2], in_c),
            tf.equal(indices[:, 3], out_c)
        )

        # Step 3: Apply the mask
        new_indices = tf.boolean_mask(indices, mask)
        new_values = tf.boolean_mask(values, mask)

        # Step 4: Drop the in_c and out_c dimensions (since we know them)
        new_indices = new_indices[:, :2]  # keep only the [k_row, k_col] parts

        # Step 5: Create a new sparse tensor of shape [K, K]
        K = Q_sparse.dense_shape[0]  # assuming square kernel
        kernel_channel_sp = tf.sparse.SparseTensor(
            indices=new_indices,
            values=new_values,
            dense_shape=[K, K]
        )
        return kernel_channel_sp

    batch_size, H, W, C_in = X.shape
    #KH, KW, Q_cin, Q_cout = Q.shape
    KH, KW, Q_cin, Q_cout = Q_sp.dense_shape
    assert C_in == Q_cin, "Input channels in X and Q must match"

    per_output_sparse = []

    for out_c in range(Q_cout):
        sparse_list = []
        for in_c in range(C_in):
            input_channel = X[0, :, :, in_c]
            #kernel_channel_sp = Q[:, :, in_c, out_c]
            kernel_channel_sp = extract_kernel_channel(Q_sp, in_c, out_c)
            sparse_result = sparse_to_sparse_fast(input_channel, kernel_channel_sp)  # Must return tf.SparseTensor
            sparse_list.append(sparse_result)

        stacked = tf.sparse.concat(axis=1, sp_inputs=sparse_list)  # concat across input channels
        per_output_sparse.append(stacked)

    final_sparse = tf.sparse.concat(axis=0, sp_inputs=per_output_sparse)
    return final_sparse
def conv_sparse_fast2(X, Q_sp):
    H = X.shape[1]
    W = X.shape[2]
    K = Q_sp.dense_shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    C_out = Q_sp.dense_shape[3]

    S = sparse_to_sparse_fast2(X, Q_sp)  # handles multiple output channels

    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])  # [C*H*W, N]

    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * num_rig_gen * num_col_gen, N]

    # Now reshape to [C_out, num_rig_gen, num_col_gen, N]
    result_reshaped = tf.reshape(result_dense, [C_out, num_rig_gen, num_col_gen, N])

    # Transpose to [N, num_rig_gen, num_col_gen, C_out]
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])
    return result_final


#c_in = x, c_out = y, N = z, Q sparse, fast
# vettorizza sparse_to_sparse2 e all'interno chiama sparse_to_sparse_fast
def sparse_to_sparse_fast3(X, Q_sp):
    batch_size, H, W, C_in = X.shape
    KH, KW, Q_cin, Q_cout = Q_sp.dense_shape
    assert C_in == Q_cin, "Input channels in X and Q must match"

    num_rows_per_output = (H - KH + 1) * (W - KW + 1)
    total_output_rows = num_rows_per_output * Q_cout
    total_output_cols = H * W * C_in  # important: account for all input channels

    # Handle completely empty sparse kernel
    if tf.shape(Q_sp.indices)[0] == 0:
        return tf.sparse.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.float32),
            dense_shape=[total_output_rows, total_output_cols]
        )

    # Group kernel entries by (in_c, out_c)
    indices_np = Q_sp.indices.numpy()
    values_np = Q_sp.values.numpy()
    kernel_dict = {}
    for i in range(indices_np.shape[0]):
        r, c, in_c, out_c = indices_np[i]
        key = (int(in_c), int(out_c))
        if key not in kernel_dict:
            kernel_dict[key] = ([], [])
        kernel_dict[key][0].append([r, c])
        kernel_dict[key][1].append(values_np[i])

    per_output_sparse = []

    for out_c in range(Q_cout):
        sparse_list = []
        for in_c in range(C_in):
            input_channel = X[0, :, :, in_c]
            key = (in_c, out_c)
            if key not in kernel_dict:
                continue
            idxs, vals = kernel_dict[key]
            if len(idxs) == 0:
                continue

            kernel_channel_sp = tf.sparse.SparseTensor(
                indices=tf.constant(idxs, dtype=tf.int64),
                values=tf.constant(vals, dtype=tf.float32),
                dense_shape=[KH, KW]
            )
            sparse_result = sparse_to_sparse_fast(input_channel, kernel_channel_sp)

            # Shift column indices by in_c * H * W
            offset = in_c * H * W
            shifted_indices = tf.stack(
                [sparse_result.indices[:, 0], sparse_result.indices[:, 1] + offset],
                axis=1
            )
            shifted_sparse = tf.sparse.SparseTensor(
                indices=shifted_indices,
                values=sparse_result.values,
                dense_shape=[num_rows_per_output, total_output_cols]
            )
            sparse_list.append(shifted_sparse)

        if sparse_list:
            stacked = reduce(tf.sparse.add, sparse_list)
        else:
            stacked = tf.sparse.SparseTensor(
                indices=tf.zeros([0, 2], dtype=tf.int64),
                values=tf.zeros([0], dtype=tf.float32),
                dense_shape=[num_rows_per_output, total_output_cols]
            )
        per_output_sparse.append(stacked)

    final_sparse = tf.sparse.concat(axis=0, sp_inputs=per_output_sparse)
    return final_sparse
def conv_sparse_fast3(X, Q_sp):
    H = X.shape[1]
    W = X.shape[2]
    K = Q_sp.dense_shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    C_out = Q_sp.dense_shape[3]


    S = sparse_to_sparse_fast3(X, Q_sp)


    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])  # [C*H*W, N]

    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * num_rig_gen * num_col_gen, N]

    # Now reshape to [C_out, num_rig_gen, num_col_gen, N]
    result_reshaped = tf.reshape(result_dense, [C_out, num_rig_gen, num_col_gen, N])

    # Transpose to [N, num_rig_gen, num_col_gen, C_out]
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])
    return result_final

#c_in = x, c_out = y, N = z, Q sparse, fast
# sparse_to_sparse_fast3 fused con sparse_to_sparse_fast
def sparse_to_sparse_fast4(X, Q_sp):
    batch_size, H, W, C_in = X.shape
    KH, KW, Q_cin, Q_cout = Q_sp.dense_shape
    assert C_in == Q_cin, "Input channels in X and Q must match"

    KH = int(KH)
    KW = int(KW)
    num_rows_per_output = (H - KH + 1) * (W - KW + 1)
    total_output_rows = num_rows_per_output * Q_cout
    total_output_cols = H * W * C_in

    if tf.shape(Q_sp.indices)[0] == 0:
        return tf.sparse.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.float32),
            dense_shape=[total_output_rows, total_output_cols]
        )

    indices_np = Q_sp.indices.numpy()
    values_np = Q_sp.values.numpy()

    row_blocks = []
    val_blocks = []

    for out_c in range(Q_cout):
        block_indices = []
        block_values = []
        for in_c in range(C_in):
            # filter relevant kernel values
            mask = (indices_np[:, 2] == in_c) & (indices_np[:, 3] == out_c)
            selected = indices_np[mask]
            if selected.shape[0] == 0:
                continue
            selected_vals = values_np[mask]

            r_idxs, c_idxs = selected[:, 0], selected[:, 1]
            nnz = selected.shape[0]

            r0 = tf.range(H - KH + 1, dtype=tf.int64)
            c0 = tf.range(W - KW + 1, dtype=tf.int64)
            rr, cc = tf.meshgrid(r0, c0, indexing='ij')
            patch_r = tf.reshape(rr, [-1])
            patch_c = tf.reshape(cc, [-1])
            patch_ids = tf.range(num_rows_per_output, dtype=tf.int64)

            patch_r_tiled = tf.repeat(patch_r, repeats=nnz)
            patch_c_tiled = tf.repeat(patch_c, repeats=nnz)
            patch_ids_tiled = tf.repeat(patch_ids, repeats=nnz)

            r_offsets = tf.tile(tf.constant(r_idxs, dtype=tf.int64), [num_rows_per_output])
            c_offsets = tf.tile(tf.constant(c_idxs, dtype=tf.int64), [num_rows_per_output])
            vals = tf.tile(tf.constant(selected_vals, dtype=tf.float32), [num_rows_per_output])

            row_ids = patch_ids_tiled + out_c * num_rows_per_output
            col_ids = (patch_r_tiled + r_offsets) * W + (patch_c_tiled + c_offsets) + in_c * H * W

            block_indices.append(tf.stack([row_ids, col_ids], axis=1))
            block_values.append(vals)

        if block_indices:
            row_blocks.append(tf.concat(block_indices, axis=0))
            val_blocks.append(tf.concat(block_values, axis=0))

    if not row_blocks:
        return tf.sparse.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.float32),
            dense_shape=[total_output_rows, total_output_cols]
        )

    final_indices = tf.concat(row_blocks, axis=0)
    final_values = tf.concat(val_blocks, axis=0)

    return tf.sparse.SparseTensor(
        indices=final_indices,
        values=final_values,
        dense_shape=[total_output_rows, total_output_cols]
    )
def conv_sparse_fast4(X, Q_sp):
    H = X.shape[1]
    W = X.shape[2]
    K = Q_sp.dense_shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    C_out = Q_sp.dense_shape[3]


    S = sparse_to_sparse_fast4(X, Q_sp)


    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])  # [C*H*W, N]

    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * num_rig_gen * num_col_gen, N]

    # Now reshape to [C_out, num_rig_gen, num_col_gen, N]
    result_reshaped = tf.reshape(result_dense, [C_out, num_rig_gen, num_col_gen, N])

    # Transpose to [N, num_rig_gen, num_col_gen, C_out]
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])
    return result_final

#c_in = x, c_out = y, N = z, Q sparse, fast
# questa proprio non ha for
def sparse_to_sparse_fast5(X, Q_sp):
    batch_size, H, W, C_in = X.shape
    KH, KW, Q_cin, Q_cout = Q_sp.dense_shape
    KH, KW, Q_cin, Q_cout = map(int, [KH, KW, Q_cin, Q_cout])
    assert C_in == Q_cin, "Input channels in X and Q must match"

    num_rows_per_output = (H - KH + 1) * (W - KW + 1)
    total_output_rows = num_rows_per_output * Q_cout
    total_output_cols = H * W * C_in

    if tf.shape(Q_sp.indices)[0] == 0:
        return tf.sparse.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.float32),
            dense_shape=[total_output_rows, total_output_cols]
        )

    Q_indices = Q_sp.indices  # [nnz, 4]
    Q_values = Q_sp.values    # [nnz]

    # Extract dimensions from indices
    r_idxs = tf.cast(Q_indices[:, 0], tf.int64)
    c_idxs = tf.cast(Q_indices[:, 1], tf.int64)
    in_chs = tf.cast(Q_indices[:, 2], tf.int64)
    out_chs = tf.cast(Q_indices[:, 3], tf.int64)
    vals = Q_values

    # Output patch coordinates
    r0 = tf.range(H - KH + 1, dtype=tf.int64)
    c0 = tf.range(W - KW + 1, dtype=tf.int64)
    rr, cc = tf.meshgrid(r0, c0, indexing='ij')
    patch_r = tf.reshape(rr, [-1])  # [P]
    patch_c = tf.reshape(cc, [-1])  # [P]
    num_patches = tf.shape(patch_r)[0]  # = num_rows_per_output

    # Expand patches and kernel entries
    patch_ids = tf.range(num_patches, dtype=tf.int64)
    patch_r_tiled = tf.repeat(patch_r, tf.shape(vals)[0])  # [P * nnz]
    patch_c_tiled = tf.repeat(patch_c, tf.shape(vals)[0])  # [P * nnz]
    patch_ids_tiled = tf.repeat(patch_ids, tf.shape(vals)[0])

    r_offsets = tf.tile(r_idxs, [num_patches])
    c_offsets = tf.tile(c_idxs, [num_patches])
    in_chs_tile = tf.tile(in_chs, [num_patches])
    out_chs_tile = tf.tile(out_chs, [num_patches])
    vals_tile = tf.tile(vals, [num_patches])

    row_ids = patch_ids_tiled + out_chs_tile * num_rows_per_output
    col_ids = (patch_r_tiled + r_offsets) * W + (patch_c_tiled + c_offsets) + in_chs_tile * H * W

    final_indices = tf.stack([row_ids, col_ids], axis=1)

    return tf.sparse.SparseTensor(
        indices=final_indices,
        values=vals_tile,
        dense_shape=[total_output_rows, total_output_cols]
    )
def conv_sparse_fast5(X, Q_sp):
    H = X.shape[1]
    W = X.shape[2]
    K = Q_sp.dense_shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    C_out = Q_sp.dense_shape[3]


    S = sparse_to_sparse_fast5(X, Q_sp)


    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])  # [C*H*W, N]

    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * num_rig_gen * num_col_gen, N]

    # Now reshape to [C_out, num_rig_gen, num_col_gen, N]
    result_reshaped = tf.reshape(result_dense, [C_out, num_rig_gen, num_col_gen, N])

    # Transpose to [N, num_rig_gen, num_col_gen, C_out]
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])
    return result_final

#c_in = x, c_out = y, N = z, Q sparse, fast
# piu o meno veloce di sparse_to_sparse_fast5 dipende dai casi credo
def sparse_to_sparse_fast6(X, Q_sp):
    batch_size, H, W, C_in = X.shape
    KH, KW, Q_cin, Q_cout = Q_sp.dense_shape
    KH, KW, Q_cin, Q_cout = map(int, [KH, KW, Q_cin, Q_cout])
    assert C_in == Q_cin, "Input channels in X and Q must match"

    num_rows_per_output = (H - KH + 1) * (W - KW + 1)
    total_output_rows = num_rows_per_output * Q_cout
    total_output_cols = H * W * C_in

    if tf.shape(Q_sp.indices)[0] == 0:
        return tf.sparse.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.float32),
            dense_shape=[total_output_rows, total_output_cols]
        )

    Q_indices = Q_sp.indices  # [nnz, 4]
    Q_values = Q_sp.values    # [nnz]
    nnz = tf.shape(Q_values)[0]

    r_idxs = tf.cast(Q_indices[:, 0], tf.int64)
    c_idxs = tf.cast(Q_indices[:, 1], tf.int64)
    in_chs = tf.cast(Q_indices[:, 2], tf.int64)
    out_chs = tf.cast(Q_indices[:, 3], tf.int64)
    vals = Q_values

    r0 = tf.range(H - KH + 1, dtype=tf.int64)
    c0 = tf.range(W - KW + 1, dtype=tf.int64)
    rr, cc = tf.meshgrid(r0, c0, indexing='ij')
    patch_r = tf.reshape(rr, [-1])  # [P]
    patch_c = tf.reshape(cc, [-1])  # [P]
    P = tf.shape(patch_r)[0]

    patch_ids = tf.range(P, dtype=tf.int64)
    nnz_ids = tf.range(nnz, dtype=tf.int64)
    patch_idx, nnz_idx = tf.meshgrid(patch_ids, nnz_ids, indexing='ij')  # [P, nnz]

    patch_idx_flat = tf.reshape(patch_idx, [-1])  # [P * nnz]
    nnz_idx_flat = tf.reshape(nnz_idx, [-1])      # [P * nnz]

    r_offset = tf.gather(r_idxs, nnz_idx_flat)
    c_offset = tf.gather(c_idxs, nnz_idx_flat)
    in_ch = tf.gather(in_chs, nnz_idx_flat)
    out_ch = tf.gather(out_chs, nnz_idx_flat)
    val = tf.gather(vals, nnz_idx_flat)

    pr = tf.gather(patch_r, patch_idx_flat)
    pc = tf.gather(patch_c, patch_idx_flat)

    row_ids = patch_idx_flat + out_ch * num_rows_per_output
    col_ids = (pr + r_offset) * W + (pc + c_offset) + in_ch * H * W

    final_indices = tf.stack([row_ids, col_ids], axis=1)

    return tf.sparse.SparseTensor(
        indices=final_indices,
        values=val,
        dense_shape=[total_output_rows, total_output_cols]
    )
def conv_sparse_fast6(X, Q_sp):
    H = X.shape[1]
    W = X.shape[2]
    K = Q_sp.dense_shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    C_out = Q_sp.dense_shape[3]


    S = sparse_to_sparse_fast6(X, Q_sp)


    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])  # [C*H*W, N]

    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * num_rig_gen * num_col_gen, N]

    # Now reshape to [C_out, num_rig_gen, num_col_gen, N]
    result_reshaped = tf.reshape(result_dense, [C_out, num_rig_gen, num_col_gen, N])

    # Transpose to [N, num_rig_gen, num_col_gen, C_out]
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])
    return result_final

#c_in = x, c_out = y, N = z, Q sparse, fast
# piu veloce di sparse_to_sparse_fast5/6
def sparse_to_sparse_fast7(X, Q_sp):
    batch_size, H, W, C_in = X.shape
    KH, KW, Q_cin, Q_cout = Q_sp.dense_shape
    KH, KW, Q_cin, Q_cout = map(int, [KH, KW, Q_cin, Q_cout])
    assert C_in == Q_cin, "Input channels in X and Q must match"

    num_rows_per_output = (H - KH + 1) * (W - KW + 1)
    total_output_rows = num_rows_per_output * Q_cout
    total_output_cols = H * W * C_in

    if tf.shape(Q_sp.indices)[0] == 0:
        return tf.sparse.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.float32),
            dense_shape=[total_output_rows, total_output_cols]
        )

    # Precompute all possible patch positions
    r0 = tf.range(H - KH + 1, dtype=tf.int64)
    c0 = tf.range(W - KW + 1, dtype=tf.int64)
    rr, cc = tf.meshgrid(r0, c0, indexing='ij')
    patch_r = tf.reshape(rr, [-1])  # [P]
    patch_c = tf.reshape(cc, [-1])  # [P]
    num_patches = tf.shape(patch_r)[0]

    # Extract and cast Q components once
    Q_indices = tf.cast(Q_sp.indices, tf.int64)
    r_idxs, c_idxs, in_chs, out_chs = tf.unstack(Q_indices, axis=1)
    vals = Q_sp.values

    # Compute the cartesian product between patches and kernel entries
    # Using broadcasting instead of explicit tiling/repeating
    patch_r_exp = tf.expand_dims(patch_r, 1)  # [P, 1]
    patch_c_exp = tf.expand_dims(patch_c, 1)  # [P, 1]
    patch_ids_exp = tf.range(num_patches, dtype=tf.int64)[:, None]  # [P, 1]

    # Compute row and column indices using broadcasting
    row_ids = (patch_ids_exp + out_chs * num_rows_per_output)  # [P, nnz]
    col_ids = ((patch_r_exp + r_idxs) * W +
               (patch_c_exp + c_idxs) +
               in_chs * H * W)  # [P, nnz]

    # Reshape to final form
    row_ids_flat = tf.reshape(row_ids, [-1])
    col_ids_flat = tf.reshape(col_ids, [-1])
    vals_flat = tf.tile(vals, [num_patches])

    final_indices = tf.stack([row_ids_flat, col_ids_flat], axis=1)

    return tf.sparse.SparseTensor(
        indices=final_indices,
        values=vals_flat,
        dense_shape=[total_output_rows, total_output_cols]
    )
def conv_sparse_fast7(X, Q_sp):
    H = X.shape[1]
    W = X.shape[2]
    K = Q_sp.dense_shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    C_out = Q_sp.dense_shape[3]


    S = sparse_to_sparse_fast7(X, Q_sp)


    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])  # [C*H*W, N]

    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * num_rig_gen * num_col_gen, N]

    # Now reshape to [C_out, num_rig_gen, num_col_gen, N]
    result_reshaped = tf.reshape(result_dense, [C_out, num_rig_gen, num_col_gen, N])

    # Transpose to [N, num_rig_gen, num_col_gen, C_out]
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])
    return result_final

#sembra piu veloce di 7
def sparse_to_sparse_fast8(X, Q_sp):
    batch_size, H, W, C_in = X.shape
    KH, KW, Q_cin, Q_cout = Q_sp.dense_shape
    KH, KW, Q_cin, Q_cout = map(int, [KH, KW, Q_cin, Q_cout])
    assert C_in == Q_cin, "Input channels in X and Q must match"

    num_rows_per_output = (H - KH + 1) * (W - KW + 1)
    total_output_rows = num_rows_per_output * Q_cout
    total_output_cols = H * W * C_in

    if tf.shape(Q_sp.indices)[0] == 0:
        return tf.sparse.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.float32),
            dense_shape=[total_output_rows, total_output_cols]
        )

    # Precompute all possible patch positions
    r0 = tf.range(H - KH + 1, dtype=tf.int64)
    c0 = tf.range(W - KW + 1, dtype=tf.int64)
    rr, cc = tf.meshgrid(r0, c0, indexing='ij')
    patch_r = tf.reshape(rr, [-1])  # [P]
    patch_c = tf.reshape(cc, [-1])  # [P]
    num_patches = tf.shape(patch_r)[0]

    # Extract and cast Q components once
    Q_indices = tf.cast(Q_sp.indices, tf.int64)
    r_idxs, c_idxs, in_chs, out_chs = tf.unstack(Q_indices, axis=1)
    vals = Q_sp.values

    # Compute the cartesian product between patches and kernel entries
    # Using broadcasting instead of explicit tiling/repeating
    patch_r_exp = tf.expand_dims(patch_r, 1)  # [P, 1]
    patch_c_exp = tf.expand_dims(patch_c, 1)  # [P, 1]
    patch_ids_exp = tf.range(num_patches, dtype=tf.int64)[:, None]  # [P, 1]

    # Compute row and column indices using broadcasting
    row_ids = (patch_ids_exp + out_chs * num_rows_per_output)  # [P, nnz]
    col_ids = ((patch_r_exp + r_idxs) * W +
               (patch_c_exp + c_idxs) +
               in_chs * H * W)  # [P, nnz]

    # Reshape to final form
    row_ids_flat = tf.reshape(row_ids, [-1])
    col_ids_flat = tf.reshape(col_ids, [-1])

    # Use broadcasting to create the values tensor
    vals_flat = tf.reshape(vals, [1, -1]) * tf.ones([num_patches, 1], dtype=tf.float32)
    vals_flat = tf.reshape(vals_flat, [-1])

    final_indices = tf.stack([row_ids_flat, col_ids_flat], axis=1)

    return tf.sparse.SparseTensor(
        indices=final_indices,
        values=vals_flat,
        dense_shape=[total_output_rows, total_output_cols]
    )
def conv_sparse_fast8(X, Q_sp):
    H = X.shape[1]
    W = X.shape[2]
    K = Q_sp.dense_shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    C_out = Q_sp.dense_shape[3]


    S = sparse_to_sparse_fast8(X, Q_sp)


    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H, W, N]
    X_flat = tf.reshape(X_perm, [C * H * W, N])  # [C*H*W, N]

    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * num_rig_gen * num_col_gen, N]

    # Now reshape to [C_out, num_rig_gen, num_col_gen, N]
    result_reshaped = tf.reshape(result_dense, [C_out, num_rig_gen, num_col_gen, N])

    # Transpose to [N, num_rig_gen, num_col_gen, C_out]
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])
    return result_final

#########################################################################
# faccio dense matmul invece di sparse matmul -- molto lento
def conv_sparse_fast_D8(X, Q_sp):
    H = X.shape[1]
    W = X.shape[2]
    K = Q_sp.dense_shape[0]
    num_col_gen = W - K + 1
    num_rig_gen = H - K + 1
    C_out = Q_sp.dense_shape[3]


    S = sparse_to_sparse_fast8(X, Q_sp)
    Sd = tf.sparse.to_dense(S, validate_indices=False)

    N, H, W, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])
    X_flat = tf.reshape(X_perm, [C * H * W, N])

    #result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)
    R = tf.matmul(Sd, X_flat, b_is_sparse=True)

    # Now reshape to [C_out, num_rig_gen, num_col_gen, N]
    result_reshaped = tf.reshape(R, [C_out, num_rig_gen, num_col_gen, N])

    # Transpose to [N, num_rig_gen, num_col_gen, C_out]
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])
    return result_final

########################################################################

# conv_sparse_fast8 & padding = same
def conv_sparse_fast8_padding(X, Q_sp):
    N, H, W, C = X.shape
    K = Q_sp.dense_shape[0]
    C_out = Q_sp.dense_shape[3]

    # Calculate padding
    pad_total = K - 1
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    # Apply symmetric padding
    X_padded = tf.pad(
        X,
        paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        mode='CONSTANT',
        constant_values=0
    )

    # New H and W after padding
    H_pad = H + pad_top + pad_bottom
    W_pad = W + pad_left + pad_right

    # Generate sparse convolution matrix
    S = sparse_to_sparse_fast8(X_padded, Q_sp)

    # Flatten and reshape input
    X_perm = tf.transpose(X_padded, perm=[3, 1, 2, 0])  # [C, H_pad, W_pad, N]
    X_flat = tf.reshape(X_perm, [C * H_pad * W_pad, N])  # [C*H_pad*W_pad, N]

    # Perform sparse matrix multiplication
    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * H * W, N]

    # Reshape result to output shape: [N, H, W, C_out]
    result_reshaped = tf.reshape(result_dense, [C_out, H, W, N])
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])  # [N, H, W, C_out]

    return result_final

# conv_sparse_fast8 con padding che è un parametro
def conv_sparse_fast8_padding_v2(X, Q_sp, padding="VALID"):
    assert padding in ("VALID", "SAME"), "padding must be 'valid' or 'same'"

    N, H, W, C = X.shape
    K = Q_sp.dense_shape[0]
    C_out = Q_sp.dense_shape[3]

    if padding == "SAME":
        # Calculate symmetric padding
        pad_total = K - 1
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        # Apply padding to input
        X = tf.pad(
            X,
            paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode='CONSTANT',
            constant_values=0
        )

        H_out, W_out = H, W
    else:  # "VALID"
        H_out = H - K + 1
        W_out = W - K + 1

    # Create sparse convolution matrix
    S = sparse_to_sparse_fast8(X, Q_sp)

    # Flatten input
    N, H_pad, W_pad, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H_pad, W_pad, N]
    X_flat = tf.reshape(X_perm, [C * H_pad * W_pad, N])  # [C*H_pad*W_pad, N]

    # Multiply
    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * H_out * W_out, N]

    # Reshape result
    result_reshaped = tf.reshape(result_dense, [C_out, H_out, W_out, N])
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])  # [N, H_out, W_out, C_out]

    return result_final

#########################################################################

# conv_sparse_fast8 con padding & stride
def sparse_to_sparse_fast8_stride(X, Q_sp, stride, OH, OW):
    batch_size, H, W, C_in = X.shape
    KH, KW, Q_cin, Q_cout = Q_sp.dense_shape
    KH, KW, Q_cin, Q_cout = map(int, [KH, KW, Q_cin, Q_cout])
    assert C_in == Q_cin, "Input channels in X and Q must match"

    num_rows_per_output = OH * OW
    total_output_rows = num_rows_per_output * Q_cout
    total_output_cols = H * W * C_in

    if tf.shape(Q_sp.indices)[0] == 0:
        return tf.sparse.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.float32),
            dense_shape=[total_output_rows, total_output_cols]
        )

    # Patch top-left positions with stride
    r0 = tf.range(0, H - KH + 1, stride, dtype=tf.int64)
    c0 = tf.range(0, W - KW + 1, stride, dtype=tf.int64)
    rr, cc = tf.meshgrid(r0, c0, indexing='ij')
    patch_r = tf.reshape(rr, [-1])  # [P]
    patch_c = tf.reshape(cc, [-1])  # [P]
    num_patches = tf.shape(patch_r)[0]

    # Extract and cast Q components
    Q_indices = tf.cast(Q_sp.indices, tf.int64)
    r_idxs, c_idxs, in_chs, out_chs = tf.unstack(Q_indices, axis=1)
    vals = Q_sp.values

    # Compute indices using broadcasting
    patch_r_exp = tf.expand_dims(patch_r, 1)
    patch_c_exp = tf.expand_dims(patch_c, 1)
    patch_ids_exp = tf.range(num_patches, dtype=tf.int64)[:, None]

    row_ids = patch_ids_exp + out_chs * num_rows_per_output
    col_ids = ((patch_r_exp + r_idxs) * W +
               (patch_c_exp + c_idxs) +
               in_chs * H * W)

    row_ids_flat = tf.reshape(row_ids, [-1])
    col_ids_flat = tf.reshape(col_ids, [-1])

    vals_flat = tf.reshape(vals, [1, -1]) * tf.ones([num_patches, 1], dtype=tf.float32)
    vals_flat = tf.reshape(vals_flat, [-1])

    final_indices = tf.stack([row_ids_flat, col_ids_flat], axis=1)

    return tf.sparse.SparseTensor(
        indices=final_indices,
        values=vals_flat,
        dense_shape=[total_output_rows, total_output_cols]
    )
def conv_sparse_fast8_padding_v2_stride(X, Q_sp, padding="VALID", stride=1):
    # In TensorFlow (and many deep learning frameworks), the "SAME" padding is designed to ensure the output shape is ceil(input / stride)
    assert padding in ("VALID", "SAME"), "padding must be 'VALID' or 'SAME'"

    N, H, W, C = X.shape
    K = int(Q_sp.dense_shape[0])
    C_out = int(Q_sp.dense_shape[3])

    if padding == "SAME":
        H_out = math.ceil(H / stride)
        W_out = math.ceil(W / stride)

        pad_along_height = max((H_out - 1) * stride + K - H, 0)
        pad_along_width = max((W_out - 1) * stride + K - W, 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        X = tf.pad(
            X,
            paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode='CONSTANT',
            constant_values=0
        )
    else:  # VALID
        H_out = (H - K) // stride + 1
        W_out = (W - K) // stride + 1

    # Generate sparse convolution matrix
    S = sparse_to_sparse_fast8_stride(X, Q_sp, stride=stride, OH=H_out, OW=W_out)

    N, H_pad, W_pad, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H_pad, W_pad, N]
    X_flat = tf.reshape(X_perm, [C * H_pad * W_pad, N])  # [C*H_pad*W_pad, N]

    # Apply sparse matmul
    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * H_out * W_out, N]

    result_reshaped = tf.reshape(result_dense, [C_out, H_out, W_out, N])
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])  # [N, H_out, W_out, C_out]

    return result_final




if __name__ == '__main__':
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
            [[0.0, 0], [0, 0], [0, 0]],
            [[0.0, 0], [0, 0], [0, 0]]
        ],
        [  # Second row of kernel
            [[0.0, 0], [0, 0.0], [0, 0]],
            [[0.0, 0], [0, 0.0], [0, 0]]
        ]
    ], dtype=tf.float32)  # Shape [2, 2, 3, 2]

    Q_sp = tf.sparse.from_dense(Q)

    custom_out = conv_sparse_fast2(X, Q_sp)  # Should return [H', W', 1]

    # X_tf = tf.reshape(X, [1, 3, 3, 3])  # [batch, height, width, in_channels]
    tf_out = tf.nn.conv2d(X, Q, strides=1, padding="VALID")
    # tf_out = tf.squeeze(tf_out)  # Remove batch and channel dims -> [H', W']

    # Compare results
    close = tf.reduce_all(tf.abs(custom_out - tf_out) < 1e-5)
    print(close.numpy())


















