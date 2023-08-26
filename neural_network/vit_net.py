import tensorflow as tf
import vit_keras
import os
import random
import numpy as np

os.environ['PYTHONHASHSEED'] = '0'
seed = 42
tf.random.set_seed(seed)
# Set the random seed for NumPy
np.random.seed(seed)
# Set the random seed for Python's built-in random module
random.seed(seed)

tf.keras.utils.set_random_seed(seed)

def VisualTransformers(cf):
    """ Input """
    inputs = tf.keras.layers.Input((cf["image_size"], cf["image_size"], cf["num_channels"])) ## (None, 512, 512, 3)

    patch_embed = tf.keras.layers.Conv2D(
        filters=cf["hidden_dim"],
        kernel_size=cf["patch_size"],
        strides=cf["patch_size"],
        padding="valid",
        name="embedding",
    )(inputs)

    """ Patch Embeddings """
    patch_embed = tf.keras.layers.Reshape((patch_embed.shape[1] * patch_embed.shape[2], cf["hidden_dim"]))(patch_embed)

    """ Position Embeddings """
    x = vit_keras.vit.layers.ClassToken(name="class_token")(patch_embed)
    x = vit_keras.vit.layers.AddPositionEmbs(name="Transformer/posembed_input")(x)

    """ Transformer Encoder """
    from vit_keras import vit
    for n in range(cf["num_layers"]):
        x, _ = vit_keras.vit.layers.TransformerBlock(
            num_heads=cf["num_heads"],
            mlp_dim=cf['mlp_dim'],
            dropout=0.1,
            name=f"Transformer/encoderblock_{n}",
        )(x)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(x)
    x = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
    x = tf.keras.layers.Dense(cf['num_classes'], name="head", activation='softmax')(x)
    model = tf.keras.models.Model(inputs, x)
    return model
