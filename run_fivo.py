# Adapted from https://github.com/tensorflow/models/tree/master/research/fivo
# Chris J. Maddison*, Dieterich Lawson*, George Tucker*, Nicolas Heess, Mohammad Norouzi, Andriy Mnih, Arnaud Doucet, and Yee Whye Teh. 
# "Filtering Variational Objectives." NIPS 2017.

"""A script to run training for VRNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from fivo import runner

# Shared flags.
tf.app.flags.DEFINE_enum("mode", "train",
                         ["train", "eval", "sample"],
                         "The mode of the binary.")
tf.app.flags.DEFINE_string("model", "vrnn",
                         "Model choice.")
tf.app.flags.DEFINE_integer("latent_size", 64,
                            "The size of the latent state of the model.")
tf.app.flags.DEFINE_string("dataset_path", "",
                           "Path to load the dataset from.")
tf.app.flags.DEFINE_integer("data_dimension", 18,
                            "The dimension of each vector in the data sequence.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                            "Batch size.")
tf.app.flags.DEFINE_integer("num_samples", 4,
                            "The number of samples (or particles) for multisample algorithms.")
tf.app.flags.DEFINE_string("logdir", "/tmp/smc_vi",
                           "The directory to keep checkpoints and summaries in.")
tf.app.flags.DEFINE_integer("random_seed", None,
                            "A random seed for seeding the TensorFlow graph.")
tf.app.flags.DEFINE_integer("parallel_iterations", 30,
                            "The number of parallel iterations to use for the while "
                            "loop that computes the bounds.")

# Training flags.
tf.app.flags.DEFINE_enum("bound", "fivo",
                         ["elbo", "iwae", "fivo"],
                         "The bound to optimize.")
tf.app.flags.DEFINE_boolean("normalize_by_seq_len", True,
                            "If true, normalize the loss by the number of timesteps "
                            "per sequence.")
tf.app.flags.DEFINE_float("learning_rate", 0.0002,
                          "The learning rate for ADAM.")
tf.app.flags.DEFINE_integer("max_steps", int(1e9),
                            "The number of gradient update steps to train for.")
tf.app.flags.DEFINE_integer("summarize_every", 50,
                            "The number of steps between summaries.")
tf.app.flags.DEFINE_enum("resampling_type", "multinomial",
                         ["multinomial", "relaxed"],
                         "The resampling strategy to use for training.")
tf.app.flags.DEFINE_float("relaxed_resampling_temperature", 0.5,
                          "The relaxation temperature for relaxed resampling.")
tf.app.flags.DEFINE_enum("proposal_type", "filtering",
                         ["prior", "filtering", "smoothing"],
                         "The type of proposal to use.")

# Distributed training flags.
tf.app.flags.DEFINE_string("master", "",
                           "The BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_integer("task", 0,
                            "Task id of the replica running the training.")
tf.app.flags.DEFINE_integer("ps_tasks", 0,
                            "Number of tasks in the ps job. If 0 no ps job is used.")
tf.app.flags.DEFINE_boolean("stagger_workers", True,
                            "If true, bring one worker online every 1000 steps.")

# Evaluation flags.
tf.app.flags.DEFINE_enum("split", "train",
                         ["train", "test", "valid"],
                         "Split to evaluate the model on.")

# Sampling flags.
tf.app.flags.DEFINE_integer("sample_length", 50,
                            "The number of timesteps to sample for.")
tf.app.flags.DEFINE_integer("prefix_length", 25,
                            "The number of timesteps to condition the model on before sampling.")
tf.app.flags.DEFINE_string("sample_out_dir", None,
                           "The directory to write the samples to."
                           "Defaults to logdir.")

FLAGS = tf.app.flags.FLAGS

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.model == "vrnn":
    if FLAGS.mode == "train":
      runner.run_train(FLAGS)
    elif FLAGS.mode == "eval":
      runner.run_eval(FLAGS)
    elif FLAGS.mode == "sample":
      runner.run_sample(FLAGS)

if __name__ == "__main__":
  tf.app.run(main)
