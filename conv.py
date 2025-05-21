import numpy as np
import tensorflow as tf
from functools import reduce
import math
import funzioni
import v4
from tensorflow.python.ops.gen_sparse_ops import sparse_reorder, sparse_tensor_dense_mat_mul, sparse_to_dense


# trasforma il kernel in matrice e fa sparse matmul
# consuma tantissima memoria perchè quasi tutti gli elementi sono replicati
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
    '''Sv = S.values
    print(S.shape)
    print(Q_sp.shape)
    print("S.values shape:",Sv.shape)
    print("S.indices shape:", S.indices.shape)
    #print(S.dense_shape)
    unique_values, _ = tf.unique(S.values)  # `_` is the indices (not needed here)
    #print("Unique values:", unique_values.numpy())  # Convert to numpy for readability
    print("Number of unique values:", len(unique_values.numpy()))
    #print(S.values)
    exit()'''

    N, H_pad, W_pad, C = X.shape
    X_perm = tf.transpose(X, perm=[3, 1, 2, 0])  # [C, H_pad, W_pad, N]
    X_flat = tf.reshape(X_perm, [C * H_pad * W_pad, N])  # [C*H_pad*W_pad, N]

    # Apply sparse matmul
    result_dense = tf.sparse.sparse_dense_matmul(S, X_flat)  # [C_out * H_out * W_out, N]

    result_reshaped = tf.reshape(result_dense, [C_out, H_out, W_out, N])
    result_final = tf.transpose(result_reshaped, perm=[3, 1, 2, 0])  # [N, H_out, W_out, C_out]

    return result_final

###########################################################################

# vari tentativi di fare conv diretta solo sugli elementi != 0
# piu lento rispetto a conv_sparse_fast8_padding_v2_stride generalmente
def direct_sparse_conv2d(input_tensor, sparse_filter, stride=1, padding='SAME'):
    """
    Performs sparse 2D convolution without densifying the kernel.
    Args:
        input_tensor: tf.Tensor of shape [B, H, W, Cin]
        sparse_filter: tf.SparseTensor of shape [Kh, Kw, Cin, Cout]
        stride: int
        padding: 'SAME' or 'VALID'
    Returns:
        tf.Tensor of shape [B, H_out, W_out, Cout]
    """
    B, H, W, Cin = input_tensor.shape
    Kh, Kw, Cin_f, Cout = [int(d) for d in sparse_filter.dense_shape.numpy()]

    assert Cin == Cin_f, "Input channels do not match"

    # Compute padding manually
    if padding == 'SAME':
        pad_h_total = max((H + stride - 1) // stride * stride + Kh - stride - H, 0)
        pad_w_total = max((W + stride - 1) // stride * stride + Kw - stride - W, 0)
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
    else:
        pad_top = pad_bottom = pad_left = pad_right = 0

    input_padded = tf.pad(input_tensor, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    H_padded, W_padded = input_padded.shape[1], input_padded.shape[2]
    H_out = (H_padded - Kh) // stride + 1
    W_out = (W_padded - Kw) // stride + 1

    output = tf.zeros([B, H_out, W_out, Cout], dtype=tf.float32)

    for idx, val in zip(sparse_filter.indices, sparse_filter.values):
        fh, fw, cin, cout = idx.numpy()

        patch = input_padded[:, fh:fh + H_out * stride:stride,
                                 fw:fw + W_out * stride:stride,
                                 cin]  # Shape: [B, H_out, W_out]

        patch = tf.expand_dims(patch, -1)  # [B, H_out, W_out, 1]
        contribution = patch * val  # [B, H_out, W_out, 1]
        mask = tf.one_hot(cout, depth=Cout, dtype=tf.float32)
        contribution_full = contribution * mask  # [B, H_out, W_out, Cout]

        output += contribution_full

    return output
def direct_sparse_conv2d_2(X, Q_sp, stride=1, padding='SAME'):
    """
    Fast 2D sparse convolution with square kernel and stride using TensorFlow.

    Args:
        X: [N, H, W, C_in] float32 tensor
        Q_sp: tf.sparse.SparseTensor with shape [K, K, C_in, C_out]
        stride: int
        padding: "SAME" or "VALID"

    Returns:
        Output tensor of shape [N, H_out, W_out, C_out]
    """
    # Extract input dimensions
    N, H, W, C_in = X.shape
    K = Q_sp.dense_shape[0]  # kernel height == width
    C_out = Q_sp.dense_shape[3]

    # Extract sparse kernel info
    indices = Q_sp.indices.numpy()  # [num_nonzeros, 4]
    values = Q_sp.values  # [num_nonzeros]

    # Extract input patches
    patches = tf.image.extract_patches(
        images=X,
        sizes=[1, K, K, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding=padding
    )  # shape: [N, H_out, W_out, K*K*C_in]

    # Get output dimensions
    N, H_out, W_out, _ = patches.shape

    # Prepare output
    output = tf.zeros([N, H_out, W_out, C_out], dtype=tf.float32)

    # Loop over sparse kernel entries
    for i in range(len(values)):
        kh, kw, cin, cout = indices[i]
        v = values[i]

        # Flat index into K*K*C_in dimension
        patch_idx = kh * K * C_in + kw * C_in + cin

        # Extract relevant input values
        input_patch_values = patches[..., patch_idx]  # shape: [N, H_out, W_out]

        # Accumulate to the correct output channel
        output = tf.tensor_scatter_nd_add(
            output,
            indices=tf.constant([[0, 0, 0, cout]], dtype=tf.int32),
            updates=tf.expand_dims(input_patch_values, -1) * v
        )

    return output
def direct_sparse_conv2d_3(X, Q_sp, stride=1, padding='SAME'):
    """
    Fast 2D sparse convolution with square kernel and stride using TensorFlow.

    Args:
        X: [N, H, W, C_in] float32 tensor
        Q_sp: tf.sparse.SparseTensor with shape [K, K, C_in, C_out]
        stride: int
        padding: "SAME" or "VALID"

    Returns:
        Output tensor of shape [N, H_out, W_out, C_out]
    """
    # Input and kernel info
    N, H, W, C_in = X.shape
    K = Q_sp.dense_shape[0]
    C_out = Q_sp.dense_shape[3]

    # Extract input patches [N, H_out, W_out, K*K*C_in]
    patches = tf.image.extract_patches(
        images=X,
        sizes=[1, K, K, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding=padding
    )  # shape: [N, H_out, W_out, K*K*C_in]

    # Output shape
    N, H_out, W_out, _ = patches.shape
    output = tf.zeros([N, H_out, W_out, C_out], dtype=tf.float32)

    # Unpack sparse filter
    indices = Q_sp.indices.numpy()  # shape: [nnz, 4]
    values = Q_sp.values  # shape: [nnz]

    # Loop over non-zero weights
    for idx, v in zip(indices, values):
        kh, kw, cin, cout = idx
        patch_idx = kh * K * C_in + kw * C_in + cin  # flat index
        selected = patches[..., patch_idx]  # shape: [N, H_out, W_out]
        #output += tf.expand_dims(selected, -1) * v * tf.one_hot(cout, C_out)
        #output += tf.expand_dims(selected, -1) * v * tf.one_hot(tf.cast(cout, tf.int32), C_out)
        output += tf.expand_dims(selected, -1) * v * tf.one_hot(tf.cast(cout, tf.int32), tf.cast(C_out, tf.int32))

    return output
def direct_sparse_conv2d_4(X, Q_sp, stride=1, padding='SAME'):
    """
    Optimized 2D sparse convolution with square kernel and stride using TensorFlow.

    Args:
        X: [N, H, W, C_in] float32 tensor
        Q_sp: tf.sparse.SparseTensor with shape [K, K, C_in, C_out]
        stride: int
        padding: "SAME" or "VALID"

    Returns:
        Output tensor of shape [N, H_out, W_out, C_out]
    """
    # Input and kernel info
    N, H, W, C_in = X.shape
    K = Q_sp.dense_shape[0]
    C_out = Q_sp.dense_shape[3]

    # Extract input patches [N, H_out, W_out, K*K*C_in]
    patches = tf.image.extract_patches(
        images=X,
        sizes=[1, K, K, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding=padding
    )

    # Get sparse tensor components
    indices = Q_sp.indices  # shape: [nnz, 4]
    values = Q_sp.values  # shape: [nnz]

    # Precompute the patch indices for all sparse weights
    kh = indices[:, 0]
    kw = indices[:, 1]
    cin = indices[:, 2]
    cout = indices[:, 3]
    patch_indices = kh * (K * C_in) + kw * C_in + cin  # shape: [nnz]

    # Gather all relevant patches in one operation [nnz, N, H_out, W_out]
    selected_patches = tf.gather(patches, patch_indices, axis=3)  # [N, H_out, W_out, nnz]
    selected_patches = tf.transpose(selected_patches, [3, 0, 1, 2])  # [nnz, N, H_out, W_out]

    # Multiply by sparse weights [nnz, N, H_out, W_out]
    weighted_patches = selected_patches * tf.reshape(values, [-1, 1, 1, 1])

    # Initialize output tensor
    N, H_out, W_out, _ = patches.shape
    output = tf.zeros([N, H_out, W_out, C_out], dtype=tf.float32)

    # Scatter-add the weighted patches to the output channels
    # We use tf.tensor_scatter_nd_add for efficient accumulation
    cout_indices = tf.cast(cout, tf.int32)
    batch_indices = tf.range(N)[:, None, None, None]  # [N, 1, 1, 1]
    h_indices = tf.range(H_out)[None, :, None, None]  # [1, H_out, 1, 1]
    w_indices = tf.range(W_out)[None, None, :, None]  # [1, 1, W_out, 1]

    # Create full indices tensor [nnz, N, H_out, W_out, 4]
    scatter_indices = tf.stack([
        tf.broadcast_to(batch_indices, [N, H_out, W_out, tf.shape(indices)[0]]),
        tf.broadcast_to(h_indices, [N, H_out, W_out, tf.shape(indices)[0]]),
        tf.broadcast_to(w_indices, [N, H_out, W_out, tf.shape(indices)[0]]),
        tf.broadcast_to(cout_indices[None, None, None, :], [N, H_out, W_out, tf.shape(indices)[0]])
    ], axis=-1)

    # Reshape for scatter operation
    scatter_indices = tf.reshape(scatter_indices, [-1, 4])
    weighted_patches = tf.reshape(tf.transpose(weighted_patches, [1, 2, 3, 0]), [-1])

    # Perform scatter add
    output = tf.tensor_scatter_nd_add(
        output,
        scatter_indices,
        weighted_patches
    )

    return output
def direct_sparse_conv2d_5(X, Q_sp, stride=1, padding='SAME'):
    """
    Fast and optimized 2D sparse convolution using TensorFlow.

    Args:
        X: [N, H, W, C_in] float32 tensor
        Q_sp: tf.sparse.SparseTensor with shape [K, K, C_in, C_out]
        stride: int
        padding: 'SAME' or 'VALID'

    Returns:
        Tensor of shape [N, H_out, W_out, C_out]
    """
    N, H, W, C_in = tf.unstack(tf.shape(X))
    K = Q_sp.dense_shape[0]
    C_out = Q_sp.dense_shape[3]

    # Extract patches [N, H_out, W_out, K*K*C_in]
    patches = tf.image.extract_patches(
        images=X,
        sizes=[1, K, K, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding=padding
    )  # shape: [N, H_out, W_out, K*K*C_in]

    N, H_out, W_out, _ = tf.unstack(tf.shape(patches))

    # Reshape to [N*H_out*W_out, K*K*C_in]
    patches_2d = tf.reshape(patches, [N * H_out * W_out, -1])  # [NHW, K*K*C_in]

    # Build sparse weight matrix: rows = output_channels, cols = K*K*C_in
    kh, kw, cin, cout = tf.unstack(Q_sp.indices, axis=1)
    #weight_flat_idx = kh * (K * C_in) + kw * C_in + cin  # shape: [nnz]

    K = tf.cast(Q_sp.dense_shape[0], tf.int64)
    C_in = tf.cast(Q_sp.dense_shape[2], tf.int64)
    kh = tf.cast(kh, tf.int64)
    kw = tf.cast(kw, tf.int64)
    cin = tf.cast(cin, tf.int64)
    weight_flat_idx = kh * (K * C_in) + kw * C_in + cin

    # Shape: [nnz,]
    sparse_weights = tf.SparseTensor(
        indices=tf.stack([cout, weight_flat_idx], axis=1),
        values=Q_sp.values,
        dense_shape=[C_out, K * K * C_in]
    )

    # Transpose so we can multiply [NHW, KKC] @ [KKC, C_out]^T = [NHW, C_out]
    output_2d = tf.sparse.sparse_dense_matmul(patches_2d, tf.sparse.transpose(sparse_weights))  # [NHW, C_out]

    # Reshape back to [N, H_out, W_out, C_out]
    #output = tf.reshape(output_2d, [N, H_out, W_out, C_out])
    output = tf.reshape(output_2d, tf.stack([
        tf.cast(N, tf.int32),
        tf.cast(H_out, tf.int32),
        tf.cast(W_out, tf.int32),
        tf.cast(C_out, tf.int32)
    ]))

    return output
def direct_sparse_conv2d_6(X, Q_sp, stride=1, padding='SAME'):
    """
    Optimized 2D sparse convolution for small kernels and stride 1/2.
    """
    X_shape = tf.shape(X)
    N, H, W, C_in = X_shape[0], X_shape[1], X_shape[2], X_shape[3]
    K = tf.cast(Q_sp.dense_shape[0], tf.int32)
    C_out = tf.cast(Q_sp.dense_shape[3], tf.int32)

    # Extract patches: [N, H_out, W_out, K*K*C_in]
    patches = tf.image.extract_patches(
        images=X,
        sizes=[1, K, K, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding=padding
    )

    patches_shape = tf.shape(patches)
    N, H_out, W_out = patches_shape[0], patches_shape[1], patches_shape[2]
    patches_2d = tf.reshape(patches, [N * H_out * W_out, K * K * C_in])  # [NHW, KKC]

    # Compute flat indices for the sparse kernel
    indices = Q_sp.indices
    values = Q_sp.values
    kh, kw, cin, cout = tf.split(indices, 4, axis=1)
    kh = tf.cast(kh, tf.int32)
    kw = tf.cast(kw, tf.int32)
    cin = tf.cast(cin, tf.int32)
    cout = tf.cast(cout, tf.int32)

    flat_idx = kh * (K * C_in) + kw * C_in + cin  # [nnz, 1]
    flat_idx = tf.reshape(flat_idx, [-1])
    cout = tf.reshape(cout, [-1])

    # Build sparse matrix: [C_out, K*K*C_in]
    sp_indices = tf.stack([cout, flat_idx], axis=1)

    sp_weights = tf.SparseTensor(
        indices=tf.cast(sp_indices, tf.int64),  # <-- cast to int64
        values=values,
        dense_shape=tf.cast([C_out, K * K * C_in], tf.int64)  # also cast shape
    )

    sp_weights = tf.sparse.reorder(sp_weights)

    # Multiply: [NHW, KKC] @ [KKC, C_out]^T = [NHW, C_out]
    output_2d = tf.sparse.sparse_dense_matmul(patches_2d, tf.sparse.transpose(sp_weights))

    output = tf.reshape(output_2d, [N, H_out, W_out, C_out])
    return output
def direct_sparse_conv2d_7(X, Q_sp, stride=1, padding='SAME'):
    """
    Specialized sparse 2D convolution for kernel size 1 or 3, stride 1 or 2.

    Args:
        X: [N, H, W, C_in] float32 tensor
        Q_sp: tf.sparse.SparseTensor with shape [K, K, C_in, C_out]
        stride: 1 or 2
        padding: 'SAME' or 'VALID'
    """
    assert stride in (1, 2), "Stride must be 1 or 2"
    K = int(Q_sp.dense_shape[0])
    assert K in (1, 3), "Kernel size must be 1 or 3"

    X_shape = tf.shape(X)
    N, H, W, C_in = X_shape[0], X_shape[1], X_shape[2], X_shape[3]
    C_out = tf.cast(Q_sp.dense_shape[3], tf.int32)

    if K == 1:
        # For 1x1 conv, this is just matmul at each location
        X_flat = tf.reshape(X, [-1, C_in])  # [N*H*W, C_in]

        # Extract weight matrix [C_in, C_out] from sparse kernel
        indices = Q_sp.indices
        values = Q_sp.values
        cin = tf.cast(indices[:, 2], tf.int32)
        cout = tf.cast(indices[:, 3], tf.int32)
        weight_indices = tf.stack([cin, cout], axis=1)
        weight_matrix = tf.sparse.SparseTensor(
            indices=tf.cast(weight_indices, tf.int64),
            values=values,
            dense_shape=[C_in, C_out]
        )
        weight_matrix = tf.sparse.reorder(weight_matrix)

        Y_flat = tf.sparse.sparse_dense_matmul(X_flat, weight_matrix)  # [N*H*W, C_out]
        Y = tf.reshape(Y_flat, [N, H, W, C_out])

        if stride == 2:
            Y = Y[:, ::2, ::2, :]

        return Y

    elif K == 3:
        # Use extract_patches to get local 3x3 regions
        patches = tf.image.extract_patches(
            images=X,
            sizes=[1, 3, 3, 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding=padding
        )  # [N, H_out, W_out, 3*3*C_in]

        patch_shape = tf.shape(patches)
        N, H_out, W_out = patch_shape[0], patch_shape[1], patch_shape[2]
        patches_flat = tf.reshape(patches, [N * H_out * W_out, 9 * C_in])

        # Build sparse kernel matrix [C_out, 9*C_in]
        indices = Q_sp.indices
        values = Q_sp.values
        kh = tf.cast(indices[:, 0], tf.int32)
        kw = tf.cast(indices[:, 1], tf.int32)
        cin = tf.cast(indices[:, 2], tf.int32)
        cout = tf.cast(indices[:, 3], tf.int32)

        flat_idx = kh * C_in * 3 + kw * C_in + cin  # [nnz]
        weight_indices = tf.stack([cout, flat_idx], axis=1)

        weight_matrix = tf.SparseTensor(
            indices=tf.cast(weight_indices, tf.int64),
            values=values,
            dense_shape=[C_out, 9 * C_in]
        )
        weight_matrix = tf.sparse.reorder(weight_matrix)

        Y_flat = tf.sparse.sparse_dense_matmul(patches_flat, tf.sparse.transpose(weight_matrix))
        Y = tf.reshape(Y_flat, [N, H_out, W_out, C_out])
        return Y
def direct_sparse_conv2d_8(X, Q_sp, stride=1, padding='SAME'):
    K = tf.cast(Q_sp.dense_shape[0], tf.int32)
    assert K in (1, 3), "Only 1x1 or 3x3 kernels supported"

    N, H, W, C_in = tf.unstack(tf.shape(X))
    C_out = tf.cast(Q_sp.dense_shape[3], tf.int32)

    if K == 1:
        # Preprocess: sparse [C_in, C_out]
        indices = Q_sp.indices
        values = Q_sp.values
        cin = tf.cast(indices[:, 2], tf.int32)
        cout = tf.cast(indices[:, 3], tf.int32)

        weight_indices = tf.stack([cin, cout], axis=1)
        weight_matrix = tf.SparseTensor(
            indices=tf.cast(weight_indices, tf.int64),
            values=values,
            dense_shape=[C_in, C_out]
        )
        weight_matrix = tf.sparse.reorder(weight_matrix)

        X_flat = tf.reshape(X, [-1, C_in])
        Y_flat = tf.sparse.sparse_dense_matmul(X_flat, weight_matrix)
        Y = tf.reshape(Y_flat, [N, H, W, C_out])

        if stride == 2:
            Y = Y[:, ::2, ::2, :]

        return Y

    else:
        # Extract 3x3 patches
        patches = tf.image.extract_patches(
            images=X,
            sizes=[1, 3, 3, 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding=padding
        )  # [N, H_out, W_out, 9*C_in]

        N, H_out, W_out = tf.unstack(tf.shape(patches))[:3]
        patch_dim = 9 * C_in
        patches_flat = tf.reshape(patches, [N * H_out * W_out, patch_dim])

        # Sparse kernel flattening
        indices = Q_sp.indices
        values = Q_sp.values
        kh = tf.cast(indices[:, 0], tf.int32)
        kw = tf.cast(indices[:, 1], tf.int32)
        cin = tf.cast(indices[:, 2], tf.int32)
        cout = tf.cast(indices[:, 3], tf.int32)

        flat_idx = kh * 3 * C_in + kw * C_in + cin
        weight_indices = tf.stack([cout, flat_idx], axis=1)

        weight_matrix = tf.SparseTensor(
            indices=tf.cast(weight_indices, tf.int64),
            values=values,
            dense_shape=[C_out, patch_dim]
        )
        weight_matrix = tf.sparse.reorder(weight_matrix)

        # Sparse matmul: [N*H_out*W_out, patch_dim] x [patch_dim, C_out]
        Y_flat = tf.sparse.sparse_dense_matmul(patches_flat, tf.sparse.transpose(weight_matrix))
        Y = tf.reshape(Y_flat, [N, H_out, W_out, C_out])
        return Y
def direct_sparse_conv2d_9(X, Q_sp, stride=1, padding='SAME'):
    K = tf.cast(Q_sp.dense_shape[0], tf.int32)
    assert K in (1, 3), "Only 1x1 or 3x3 kernels supported"

    N, H, W, C_in = tf.unstack(tf.shape(X))
    C_out = tf.cast(Q_sp.dense_shape[3], tf.int32)

    if K == 1:
        # Preprocess: sparse [C_in, C_out]
        indices = Q_sp.indices
        values = Q_sp.values
        cin = tf.cast(indices[:, 2], tf.int32)
        cout = tf.cast(indices[:, 3], tf.int32)

        weight_indices = tf.stack([cin, cout], axis=1)
        weight_matrix = tf.SparseTensor(
            indices=tf.cast(weight_indices, tf.int64),
            values=values,
            dense_shape=[C_in, C_out]
        )
        weight_matrix = tf.sparse.reorder(weight_matrix)

        X_flat = tf.reshape(X, [-1, C_in])
        Y_flat = tf.sparse.sparse_dense_matmul(X_flat, weight_matrix)
        Y = tf.reshape(Y_flat, [N, H, W, C_out])

        if stride == 2:
            Y = Y[:, ::2, ::2, :]

        return Y

    else:
        # Extract 3x3 patches
        patches = tf.image.extract_patches(
            images=X,
            sizes=[1, 3, 3, 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding=padding
        )  # [N, H_out, W_out, 9*C_in]

        N, H_out, W_out, _ = tf.unstack(tf.shape(patches))
        patch_dim = 9 * C_in
        patches_flat = tf.reshape(patches, [N * H_out * W_out, patch_dim])

        # Sparse kernel flattening
        indices = Q_sp.indices
        values = Q_sp.values
        kh = tf.cast(indices[:, 0], tf.int32)
        kw = tf.cast(indices[:, 1], tf.int32)
        cin = tf.cast(indices[:, 2], tf.int32)
        cout = tf.cast(indices[:, 3], tf.int32)

        flat_idx = kh * 3 * C_in + kw * C_in + cin
        weight_indices = tf.stack([flat_idx, cout], axis=1)  # Note the order change here

        weight_matrix = tf.SparseTensor(
            indices=tf.cast(weight_indices, tf.int64),
            values=values,
            dense_shape=[patch_dim, C_out]
        )
        weight_matrix = tf.sparse.reorder(weight_matrix)

        # Sparse matmul: [N*H_out*W_out, patch_dim] x [patch_dim, C_out]
        Y_flat = tf.sparse.sparse_dense_matmul(patches_flat, weight_matrix)
        Y = tf.reshape(Y_flat, [N, H_out, W_out, C_out])
        return Y

#############################################################################


# versioni di con in cui X diventa una matrice e filters è vettore
def classic_conv2d(input, filters_sparse, stride=1, padding='SAME'):
    def get_padding(padding, input_size, kernel_size, stride):
        input_size = tf.cast(input_size, tf.float32)
        kernel_size = tf.cast(kernel_size, tf.float32)
        stride = tf.cast(stride, tf.float32)

        if padding == 'SAME':
            out_size = tf.math.ceil(input_size / stride)
            pad_needed = tf.cast(tf.maximum((out_size - 1) * stride + kernel_size - input_size, 0), tf.int32)
            pad_before = pad_needed // 2
            pad_after = pad_needed - pad_before
            return pad_before, pad_after
        elif padding == 'VALID':
            return tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)
        else:
            raise ValueError("padding must be 'SAME' or 'VALID'")

    input_shape = tf.shape(input)
    N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    K = tf.cast(filters_sparse.dense_shape[0], tf.int32)
    F = tf.cast(filters_sparse.dense_shape[3], tf.int32)

    pad_h1, pad_h2 = get_padding(padding, H, K, stride)
    pad_w1, pad_w2 = get_padding(padding, W, K, stride)

    paddings = tf.stack([
        [0, 0],
        [pad_h1, pad_h2],
        [pad_w1, pad_w2],
        [0, 0]
    ])
    input_padded = tf.pad(input, paddings)

    # Extract patches
    patches = tf.image.extract_patches(
        images=input_padded,
        sizes=[1, K, K, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )  # shape: (N, H_out, W_out, K*K*C)

    patches_shape = tf.shape(patches)
    N_out, H_out, W_out = patches_shape[0], patches_shape[1], patches_shape[2]
    patch_dim = tf.shape(patches)[-1]

    patches_reshaped = tf.reshape(patches, [N_out, H_out * W_out, patch_dim])  # [N, HW, K*K*C]

    # Convert sparse filters to shape [K*K*C, F]
    filters_sparse_flat = tf.sparse.reshape(filters_sparse, [patch_dim, F])

    # Manually do batch sparse matmul
    outputs = []
    for i in range(N_out):
        patch_i = patches_reshaped[i]  # [HW, patch_dim]
        out_i = tf.sparse.sparse_dense_matmul(patch_i, filters_sparse_flat)  # [HW, F]
        outputs.append(out_i)

    output = tf.stack(outputs, axis=0)  # [N, HW, F]
    output = tf.reshape(output, [N_out, H_out, W_out, F])
    return output
# piu o meno uguale a v1
def classic_conv2d_2(input, filters_sparse, stride=1, padding='SAME'):
    def get_padding(padding, input_size, kernel_size, stride):
        input_size = tf.cast(input_size, tf.float32)
        kernel_size = tf.cast(kernel_size, tf.float32)
        stride = tf.cast(stride, tf.float32)

        if padding == 'SAME':
            out_size = tf.math.ceil(input_size / stride)
            pad_needed = tf.cast(tf.maximum((out_size - 1) * stride + kernel_size - input_size, 0), tf.int32)
            pad_before = pad_needed // 2
            pad_after = pad_needed - pad_before
            return pad_before, pad_after
        elif padding == 'VALID':
            return tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)
        else:
            raise ValueError("padding must be 'SAME' or 'VALID'")

    input_shape = tf.shape(input)
    N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    K = tf.cast(filters_sparse.dense_shape[0], tf.int32)
    F = tf.cast(filters_sparse.dense_shape[3], tf.int32)

    pad_h1, pad_h2 = get_padding(padding, H, K, stride)
    pad_w1, pad_w2 = get_padding(padding, W, K, stride)

    paddings = tf.stack([
        [0, 0],
        [pad_h1, pad_h2],
        [pad_w1, pad_w2],
        [0, 0]
    ])
    input_padded = tf.pad(input, paddings)

    # Extract patches
    patches = tf.image.extract_patches(
        images=input_padded,
        sizes=[1, K, K, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )  # shape: (N, H_out, W_out, K*K*C)

    patches_shape = tf.shape(patches)
    N_out, H_out, W_out, patch_dim = patches_shape[0], patches_shape[1], patches_shape[2], patches_shape[3]
    HW = H_out * W_out

    # Reshape to [N * HW, patch_dim]
    patches_reshaped = tf.reshape(patches, [N_out * HW, patch_dim])

    # Sparse filter reshape [patch_dim, F]
    filters_sparse_flat = tf.sparse.reshape(filters_sparse, [patch_dim, F])

    # Sparse-dense matmul: [N * HW, F]
    output_flat = tf.sparse.sparse_dense_matmul(patches_reshaped, filters_sparse_flat)

    # Reshape back to [N, H_out, W_out, F]
    output = tf.reshape(output_flat, [N_out, H_out, W_out, F])
    return output
def classic_conv2d_3(input, filters_sparse, stride=1, padding='SAME'):
    input_shape = input.shape
    K = filters_sparse.dense_shape[0]  # Kernel height = kernel width
    F = filters_sparse.dense_shape[3]  # Number of output filters

    # Precompute padding statically
    def compute_padding(padding, input_dim, kernel_dim, stride_dim):
        if padding == 'SAME':
            out_dim = (input_dim + stride_dim - 1) // stride_dim
            pad_total = max((out_dim - 1) * stride_dim + kernel_dim - input_dim, 0)
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            return pad_before, pad_after
        elif padding == 'VALID':
            return 0, 0
        else:
            raise ValueError("padding must be 'SAME' or 'VALID'")

    pad_top, pad_bottom = compute_padding(padding, input_shape[1], K, stride)
    pad_left, pad_right = compute_padding(padding, input_shape[2], K, stride)

    # Pad input
    input_padded = tf.pad(input, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    # Extract patches
    patches = tf.image.extract_patches(
        images=input_padded,
        sizes=[1, K, K, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'  # Already handled manually
    )

    # Flatten patches to [N * H_out * W_out, K*K*C]
    N, H_out, W_out, patch_dim = patches.shape
    patches_flat = tf.reshape(patches, [-1, patch_dim])

    # Reshape sparse filters [patch_dim, F]
    filters_sparse_reshaped = tf.sparse.reshape(filters_sparse, [patch_dim, F])

    # Multiply: [N * H_out * W_out, patch_dim] x [patch_dim, F] = [N * H_out * W_out, F]
    output_flat = tf.sparse.sparse_dense_matmul(patches_flat, filters_sparse_reshaped)

    # Reshape to [N, H_out, W_out, F]
    output = tf.reshape(output_flat, [N, H_out, W_out, F])
    return output
def classic_conv2d_4(input, filters_sparse, stride=1, padding='SAME'):
    """
    Final working version with correct matrix dimensions
    """
    # Get shapes with consistent dtype (int32)
    input_shape = tf.shape(input, out_type=tf.int32)
    N = input_shape[0]  # Batch size
    H = input_shape[1]  # Input height
    W = input_shape[2]  # Input width
    C = input_shape[3]  # Input channels

    # Convert all dimensions to int32
    K = tf.cast(filters_sparse.dense_shape[0], tf.int32)  # Kernel size
    F = tf.cast(filters_sparse.dense_shape[3], tf.int32)  # Output filters
    stride = tf.cast(stride, tf.int32)

    # Compute output dimensions
    if padding == 'SAME':
        H_out = (H + stride - 1) // stride
        W_out = (W + stride - 1) // stride
        pad_top = ((H_out - 1) * stride + K - H) // 2
        pad_bottom = ((H_out - 1) * stride + K - H) - pad_top
        pad_left = ((W_out - 1) * stride + K - W) // 2
        pad_right = ((W_out - 1) * stride + K - W) - pad_left
    elif padding == 'VALID':
        H_out = (H - K + stride) // stride
        W_out = (W - K + stride) // stride
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    else:
        raise ValueError("padding must be 'SAME' or 'VALID'")

    # Early return if output would be empty
    if tf.logical_or(H_out <= 0, W_out <= 0):
        return tf.zeros(tf.stack([N, H_out, W_out, F]), dtype=input.dtype)

    # Apply padding if needed
    if tf.reduce_any([pad_top > 0, pad_bottom > 0, pad_left > 0, pad_right > 0]):
        input_padded = tf.pad(input, [
            [0, 0],
            [pad_top, pad_bottom],
            [pad_left, pad_right],
            [0, 0]
        ])
    else:
        input_padded = input

    # Extract patches - shape [N, H_out, W_out, K*K*C]
    patches = tf.image.extract_patches(
        images=input_padded,
        sizes=[1, K, K, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    # Reshape patches to [N*H_out*W_out, K*K*C]
    patches_flat = tf.reshape(patches, [-1, K * K * C])

    # Ensure filter dtype matches input dtype
    if filters_sparse.dtype != input.dtype:
        filters_sparse = tf.SparseTensor(
            indices=filters_sparse.indices,
            values=tf.cast(filters_sparse.values, input.dtype),
            dense_shape=filters_sparse.dense_shape
        )

    # Reshape filters to [K*K*C, F]
    filters_reshaped = tf.sparse.reshape(
        filters_sparse,
        [K * K * C, F]
    )

    # Correct matrix multiplication:
    # We want output = patches_flat @ filters_reshaped
    # So we need to transpose both matrices for sparse_dense_matmul:
    # (filters_reshaped^T @ patches_flat^T)^T
    output_flat = tf.transpose(
        tf.sparse.sparse_dense_matmul(
            tf.sparse.transpose(filters_reshaped),
            tf.transpose(patches_flat),
            adjoint_a=False,
            adjoint_b=False
        )
    )

    # Reshape output to [N, H_out, W_out, F]
    return tf.reshape(output_flat, [N, H_out, W_out, F])

#############################################################################

'''def sparse_to_dense_conv2d(input, sp_filter, stride=1, padding='SAME'):
    tf_sp = sp_filter.to_tf_sparse()
    return direct_sparse_conv2d_3(input, tf_sp, stride=stride, padding=padding)

    #tt = v4.create_tensor_row_major(3, 1, 5)
    #conv1 = sp_filter.to_tf_dense()

    #print(tf.reduce_max(tf.abs(dense_filter-conv1)))

    return tf.nn.conv2d(input,dense_filter,stride,padding)'''
def sparse_to_dense_conv2d(input, sp_filter, stride=1, padding='SAME'):
    dense_filter = tf.sparse.to_dense(sp_filter.to_tf_sparse())



    return tf.nn.conv2d(input,dense_filter,stride,padding)

def matmul(X,Y_sp):
    Y_dense = sparse_to_dense(Y_sp.indices, Y_sp.shape,Y_sp.values,
                        default_value=0,
                        validate_indices=True)
    #TODO: b_is_sparse = True dà risultati diversi -- forse gli indici non sono corretti
    return tf.matmul(X, Y_dense, b_is_sparse=False)


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


















