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
        return jax.random.bernoulli(rng_key, p=probab_positive) * 2 - 1
    else:
        return jnp.where(x > 0, 1, -1)

def binarize_fwd(x, rng_key, is_training):
    y = binarize(x, rng_key, is_training)
    return y, (x, rng_key, is_training)

def binarize_bwd(res, g):
    x, rng_key, is_training = res
    return (g, None, None)

binarize.defvjp(binarize_fwd, binarize_bwd)


class BinaryDense(nn.Module):
    features: int
    @nn.compact
    def __call__(self, inputs, rng, is_training):
        weight = self.param('weight', nn.initializers.lecun_normal(), (inputs.shape[-1], self.features))
        weight = binarize(weight, rng, is_training)
        return jnp.dot(inputs, weight)

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x, rng, is_training):
        x = x.reshape((x.shape[0], -1))

        x = BinaryDense(1024)(x, rng, is_training)
        x = nn.BatchNorm(use_running_average=not is_training)(x)
        x = jnn.relu(x)

        x = BinaryDense(1024)(x, rng, is_training)
        x = nn.BatchNorm(use_running_average=not is_training)(x)
        x = jnn.relu(x)

        x = BinaryDense(1024)(x, rng, is_training)
        x = nn.BatchNorm(use_running_average=not is_training)(x)
        x = jnn.relu(x)

        x = BinaryDense(10)(x, rng, is_training)
        
        return x


def create_train_state(rng, model, initial_learning_rate):
    variables = model.init(rng, jnp.ones([1, 28, 28]), rng, is_training=True)
    params = variables['params']
    batch_stats = variables['batch_stats']

    learning_rate_schedule = optax.exponential_decay(
        init_value=initial_learning_rate,
        transition_steps=10000,
        decay_rate=math.e,
        transition_begin=0,
        staircase=False
    )
    optimizer = optax.sgd(learning_rate=initial_learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer), batch_stats


def compute_loss(params, batch_stats, batch, apply_fn, rng):
    inputs, targets = batch
    logits, new_model_state = apply_fn({'params': params, 'batch_stats': batch_stats},
                                       inputs, rng, is_training=True, mutable=['batch_stats'])
    one_hot_targets = jnn.one_hot(targets, 10)
    loss = optax.softmax_cross_entropy(logits, one_hot_targets).mean()
    return loss, new_model_state['batch_stats']


def compute_metrics(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return accuracy


@jax.jit
def train_step(state: train_state.TrainState, batch_stats, batch, rng):
    def loss_fn(params):
        loss, new_batch_stats = compute_loss(params, batch_stats, batch, state.apply_fn, rng)
        return loss, new_batch_stats

    (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state, new_batch_stats


@jax.jit
def eval_step(state, batch_stats, batch, rng):
    inputs, targets = batch
    logits = state.apply_fn({'params': state.params, 'batch_stats': batch_stats},
                            inputs, rng, is_training=False, mutable=False)
    return compute_metrics(logits, targets)


def prepare_dataloader():
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()

    def preprocess_fn(sample):
        image = sample["image"]
        image = tf.cast(image, tf.float32) / 255.0
        label = sample["label"]
        return image, label

    train_ds = ds_builder.as_dataset(split="train", shuffle_files=True)
    train_ds = train_ds.map(preprocess_fn).shuffle(1024).batch(64)

    test_ds = ds_builder.as_dataset(split="test", shuffle_files=False)
    test_ds = test_ds.map(preprocess_fn).batch(1024)

    return train_ds, test_ds


def train_model(epochs, learning_rate):
    rng = random.PRNGKey(0)
    model = MLP()
    state, batch_stats = create_train_state(rng, model, learning_rate)

    train_ds, test_ds = prepare_dataloader()

    for epoch in range(epochs):
        for batch in tfds.as_numpy(train_ds):
            rng, _ = random.split(rng)
            loss, state, batch_stats = train_step(state, batch_stats, batch, rng)

        test_accuracy = 0
        for batch in tfds.as_numpy(test_ds):
            test_accuracy += eval_step(state, batch_stats, batch, rng)
        test_accuracy /= len(test_ds)

        print(f"Epoch {epoch + 1}, Test accuracy: {test_accuracy * 100:.2f}%")


train_model(epochs=10000, learning_rate=0.07)
