#=================================================================================================#
# TensorFlow implementation of Self-Attention GAN
#
# [Self-Attention Generative Adversarial Networks]
# (https://arxiv.org/pdf/1805.08318.pdf)
#=================================================================================================#

import tensorflow as tf
import argparse
import functools
from model import GAN
from network import SAGAN
from attrdict import AttrDict as Param

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model")
parser.add_argument('--filenames', type=str, nargs="+", default=["nsynth_train.tfrecord"])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--train", action="store_true")
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():

    tf.set_random_seed(0)

    pggan = SAGAN(
        min_resolution=[4, 4],
        max_resolution=[256, 256],
        min_channels=16,
        max_channels=512
    )

    gan = GAN(
        discriminator=pggan.discriminator,
        generator=pggan.generator,
        real_input_fn=None,
        fake_input_fn=None,
        hyper_params=Param(
            discriminator_learning_rate=4e-4,
            discriminator_beta1=0.0,
            discriminator_beta2=0.9,
            generator_learning_rate=2e-4,
            generator_beta1=0.0,
            generator_beta2=0.9
        ),
        name=args.model_dir
    )

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=args.gpu,
            allow_growth=True
        )
    )

    with tf.Session(config=config) as session:

        gan.initialize()
        gan.train(args.total_steps)
