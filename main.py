import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax
import tensorflow_datasets as tfds
from flax.training import train_state
import jax.nn as jnn
import tensorflow as tf
from typing import Any, Callable, Sequence, Optional
import math
from functools import partial

def hard_sigmoid(x):
    return jnp.clip((x + 1) / 2, 0, 1)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def binarize(x: jnp.ndarray, rng_key: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    probab_positive = hard_sigmoid(x)
    if is_training:
        ret = jnp.where(random.bernoulli(rng_key, probab_positive, x.shape), 1, -1)
    else:
        ret = jnp.where(x > 0, 1, -1)
    return ret.astype(x.dtype)

def binarize_fwd(x, rng_key, is_training):
    y = binarize(x, rng_key, is_training)
    return y, tuple()

def binarize_bwd(is_training, res, g):
    return (g, None)

binarize.defvjp(binarize_fwd, binarize_bwd)


class BinaryDense(nn.Module):
    features: int
    @nn.compact
    def __call__(self, inputs, is_training):
        weight = self.param('weight', nn.initializers.normal(3.0), (inputs.shape[-1], self.features))
        weight = binarize(weight, self.make_rng("binarize"), is_training)
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        return jnp.dot(inputs, weight) + bias

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x, is_training):
        x = x.reshape((x.shape[0], -1))

        x = BinaryDense(1024)(x, is_training)
        x = nn.BatchNorm(use_running_average=not is_training)(x)
        x = jnn.relu(x)

        x = BinaryDense(1024)(x, is_training)
        x = nn.BatchNorm(use_running_average=not is_training)(x)
        x = jnn.relu(x)

        x = BinaryDense(1024)(x, is_training)
        x = nn.BatchNorm(use_running_average=not is_training)(x)
        x = jnn.relu(x)

        x = BinaryDense(10)(x, is_training)
        
        return x


def create_train_state(model, initial_learning_rate, total_steps):
    init_rngs = {'params': jax.random.key(0)}
    variables = model.init(init_rngs, jnp.ones([10, 1, 28, 28]), is_training=True)
    params = variables['params']
    batch_stats = variables['batch_stats']

    learning_rate_schedule = optax.exponential_decay(
        init_value=initial_learning_rate,
        transition_steps=total_steps,
        decay_rate=0.3,
        transition_begin=0,
        staircase=False
    )
    optimizer = optax.sgd(learning_rate=learning_rate_schedule)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer), batch_stats


def compute_loss(params, batch_stats, batch, apply_fn, rng):
    inputs, targets = batch
    logits, new_model_state = apply_fn({'params': params, 'batch_stats': batch_stats},
                                       inputs, is_training=True, mutable=['batch_stats'], rngs={'binarize': rng})
    one_hot_targets = jnn.one_hot(targets, 10)
    loss = optax.softmax_cross_entropy(logits, one_hot_targets).mean()
    return loss, new_model_state['batch_stats']


def compute_metrics(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return accuracy

@jax.jit
def train_step(state: train_state.TrainState, batch_stats, batch, rng):
    rng_for_use, new_rng = jax.random.split(rng)
    def loss_fn(params):
        loss, new_batch_stats = compute_loss(params, batch_stats, batch, state.apply_fn, rng_for_use)
        return loss, new_batch_stats

    (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    def clip_weights(param_name, param_value):
        if 'weight' in param_name:
            return jnp.clip(param_value, -1, 1)
        return param_value
    clipped_params = jax.tree_util.tree_map_with_path(clip_weights, new_state.params)
    return loss, new_state, new_batch_stats, new_rng


@jax.jit
def eval_step(state, batch_stats, batch, rng):
    rng_for_use, new_rng = jax.random.split(rng)
    inputs, targets = batch
    logits, _ = state.apply_fn({'params': state.params, 'batch_stats': batch_stats},
                            inputs, is_training=True, mutable=['batch_stats'], rngs={'binarize': rng})
    return compute_metrics(logits, targets), new_rng


def prepare_dataloader():
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()

    def preprocess_fn(sample):
        image = sample["image"]
        image = tf.cast(image, tf.float32) / 255.0
        label = sample["label"]
        return image, label

    train_ds = ds_builder.as_dataset(split="train", shuffle_files=True)
    train_ds = train_ds.map(preprocess_fn).shuffle(1024).batch(256)

    test_ds = ds_builder.as_dataset(split="test", shuffle_files=False)
    test_ds = test_ds.map(preprocess_fn).batch(1024)

    return train_ds, test_ds


def train_model(epochs, learning_rate):
    rng = jax.random.key(0)
    model = MLP()
    
    train_ds, test_ds = prepare_dataloader()
    total_steps = epochs * len(train_ds)
    state, batch_stats = create_train_state(model, learning_rate, total_steps)

    for epoch in range(epochs):
        for batch in tfds.as_numpy(train_ds):
            loss, state, batch_stats, rng = train_step(state, batch_stats, batch, rng)

        test_accuracy = 0
        for batch in tfds.as_numpy(test_ds):
            batch_acc, rng = eval_step(state, batch_stats, batch, rng)
            test_accuracy += batch_acc
        test_accuracy /= len(test_ds)

        print(f"Epoch {epoch + 1}, Test accuracy: {test_accuracy * 100:.2f}%")


train_model(epochs=100, learning_rate=0.5)
