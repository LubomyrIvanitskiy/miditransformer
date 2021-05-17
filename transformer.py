import tensorflow as tf
import numpy as np


# ref https://colab.research.google.com/drive/19t18NjxEgFGDA25tzm5T6J5Lvy8VrLc4#scrollTo=Eb8gVwOP5swz
# ref https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb

def scaled_dot_product_attention(q, k, v, mask, prnt=False):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    if prnt: print("q@k.T:\n", matmul_qk.numpy())

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_qk = matmul_qk / tf.math.sqrt(dk)
    if prnt: print("scaled_attention_logits:\n", scaled_qk.numpy())

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_qk += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    softmax_qk = tf.nn.softmax(scaled_qk, axis=-1)  # (..., seq_len_q, seq_len_k)
    if prnt: print("softmax:\n", softmax_qk.numpy())

    output = tf.matmul(softmax_qk, v)  # (..., seq_len_q, depth_v)

    return output, softmax_qk


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, prnt=False):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        if prnt: print("multihead q", q)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        if prnt: print("concat_attention", concat_attention.shape)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                                 rate)

        self.final_layer = tf.keras.layers.Dense(d_model)

    def call(self, inp, training):
        enc_output = self.tokenizer(inp, training, None)  # (batch_size, inp_seq_len, d_model)

        final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output


### TRAIN


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


### TEST

def test_attention():
    def print_out(q, k, v):
        temp_out, temp_attn = scaled_dot_product_attention(
            q, k, v, None)
        print('Attention weights are:')
        print(temp_attn)
        print('Output is:')
        print(temp_out)

    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)


def test_multihead():
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print('test_multihead out.shape', out.shape, 'attn.shape', attn.shape)


def test_feed_forward():
    sample_ffn = point_wise_feed_forward_network(512, 2048)
    print('test_feed_forward', sample_ffn(tf.random.uniform((64, 50, 512))).shape)


def test_encoder_layer():
    sample_encoder_layer = EncoderLayer(512, 8, 2048)

    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)

    print('test_encoder_layer', sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)


def test_encoder():
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    print('test_encoder', sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

def test_transformer():
    """
    NOT WORKING
    :return:
    """
    transformer = Transformer(num_layers=2, d_model=512, num_heads=8,
                             dff=2048)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_transformer_output = transformer(temp_input, temp_input, training=False, enc_padding_mask=None,
                                            look_ahead_mask=None,
                                            dec_padding_mask=None)

    print('test_transformer', sample_transformer_output.shape)  # (batch_size, input_seq_len, d_model)


import midiwrap as mw
import os
from preprocessing import note_encoder as ne


def test_pipeline():
    midi = mw.MidiFile('data/fur_elise.mid')
    dim = 32
    d_model = dim * 6
    num_heads = 6
    track_names = midi.track_names()

    X = ne.encode_notes(midi, P=ne.coprime_P, track_name=track_names[0])

    # temp_mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    # y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    # out, attn = temp_mha(X[None], k=X[None], q=X[None], mask=None)
    # print('KOOKOO out', out.shape)  # (batch_size, input_seq_len, d_model)
    ###############
    # sample_encoder_layer = EncoderLayer(d_model, num_heads, 2048)
    #
    # sample_encoder_layer_output = sample_encoder_layer(
    #     X[None], False, None)
    #
    # print('BOOBOO test_encoder_layer', sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
    # out = sample_encoder_layer_output
    ##################

    # sample_encoder = Encoder(num_layers=4, d_model=d_model, num_heads=num_heads,
    #                          dff=2048)
    #
    # sample_encoder_output = sample_encoder(X[None], training=False, mask=None)
    #
    # print('DOODOO', sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
    # out = sample_encoder_output
    ####################
    transformer = Transformer(num_layers=4, d_model=d_model, num_heads=num_heads,
                              dff=2048)

    transformer.compile(loss='MSE', optimizer="adam")

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # checkpoint_path = "training_2/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    #
    # # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)
    #
    # transformer.fit(X[None], X[None], epochs=5000, callbacks=[cp_callback])

    transformer.load_weights("training_1/cp.ckpt")

    sample_transformer_output = transformer(X[None], training=False)

    print('MOOMOO test_transformer', sample_transformer_output.shape)  # (batch_size, input_seq_len, d_model)
    out = sample_transformer_output

    #######
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(X, aspect='auto')
    ax[1].imshow(out[0].numpy(), aspect='auto')
    plt.show()
    ####################
    # print('test_pipeline out.shape', out.shape, 'attn.shape', attn.shape)
    decoded = ne.decode_X(out[0].numpy(), P=ne.coprime_P)
    # decoded = ne.decode_X(X, P=ne.coprime_P)


    melody = mw.MelodyBuilder()
    instrument = 'piano'
    for pitch, time, duration in decoded:
        melody.add_note(pitch%127, time%ne.max_time, duration%ne.max_duration, instrument)
    melody.write_to_file('transformer_melody_cossim_saved.mid')
    pass


if __name__ == "__main__":
    # test_attention()
    # test_multihead()
    # test_feed_forward()
    # test_encoder_layer()
    # test_encoder()
    # test_transformer()
    test_pipeline()
    pass
