import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def compute_nrem_ratio(files):
    total_pos = 0
    total_pts = 0

    for file in files:
        d = np.load(file, allow_pickle=True)

        y = d["consensus_trace"]
        scoring = d["scoring"]

        scoring = np.repeat(scoring, 1000)

        mask = scoring == 3

        total_pos += np.sum(y[mask] >= 0.5)
        total_pts += np.sum(mask)

    return total_pos / total_pts


def segment_generator(
    files,
    seg_len=1000,
    stride=500,
    keep_only_positive_segments=False,
    neg_keep_prob=0.2,
    seed=42,
):

    rng = np.random.default_rng(seed)

    for file in files:
        d = np.load(file, allow_pickle=True)

        pre = d["preprocessed_data"].astype(np.float32)
        y = d["consensus_trace"].astype(np.float32)
        scoring = d["scoring"].astype(np.int8)
        # upsample from 1Hz → 1000Hz
        scoring = np.repeat(scoring, 1000)

        # z-score
        pre = (pre - pre.mean()) / (pre.std() + 1e-8)

        T = len(pre)

        for start in range(0, T - seg_len + 1, stride):
            end = start + seg_len

            x_seg = pre[start:end]
            y_seg = y[start:end]
            s_seg = scoring[start:end]

            if not np.all(s_seg == 3):
                continue

            if np.sum(y_seg) == 0:
                if keep_only_positive_segments:
                    continue
                if rng.random() > neg_keep_prob:
                    continue

            yield x_seg[:, None], y_seg[:, None]


def make_dataset(
    files, training, seg_len, stride, batch_size=16, shuffle_buffer=8192, **gen_kwargs
):

    x_spec = tf.TensorSpec(shape=(seg_len, 1), dtype=tf.float32)
    y_spec = tf.TensorSpec(shape=(seg_len, 1), dtype=tf.float32)

    ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(files, seg_len=seg_len, stride=stride, **gen_kwargs),
        output_signature=(x_spec, y_spec),
    )

    if training:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.repeat()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def focal_loss(alpha=0.25, gamma=2.0, epsilon=1e-6):
    def focal_loss_calc(y_true, y_probs):
        y_true = tf.cast(y_true, tf.float32)
        y_probs = tf.clip_by_value(y_probs, epsilon, 1.0 - epsilon)

        y_pos = tf.cast(y_true >= 0.5, tf.float32)
        y_neg = 1.0 - y_pos

        p_t = y_pos * y_probs + y_neg * (1.0 - y_probs)
        a_t = y_pos * alpha + y_neg * (1.0 - alpha)

        loss = -a_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    return focal_loss_calc


# Paths
project_root = Path.cwd().resolve().parents[1]
data_root = project_root / "training_data"

train_list = sorted(data_root.glob("trial_00[1-4].npz"))
val_list = sorted(data_root.glob("trial_005.npz"))

print("Train:", train_list)
print("Val:", val_list)


# Load pretrained model
best_model_pkl = project_root / "RNN_LSTM" / "model_training" / "best_model.pkl"

with open(best_model_pkl, "rb") as f:
    best_model = pickle.load(f)
threshold = best_model["threshold"]
distance = best_model["distance"]
width = best_model["width"]

old_model_path = project_root / "RNN_LSTM" / "model_training" / best_model["model_file"]
print("Loading model:", old_model_path)

model = keras.models.load_model(old_model_path)
model.summary()


# Dataset config
fs = 1000
segment_seconds = 1.0
stride_ratio = 0.5

seg_len = int(segment_seconds * fs)
stride = int(seg_len * stride_ratio)

batch_size = 16


# print("=== DEBUG GENERATOR ===")
#
# gen = segment_generator(train_list, seg_len=seg_len, stride=stride)
#
# count = 0
# for x, y in gen:
#     print("one sample:", x.shape, y.shape, "pos ratio:", y.mean())
#     count += 1
#     if count > 10:
#         break
#
# print("Total (approx):", count)


# Build dataset
ds_train = make_dataset(
    train_list,
    training=True,
    seg_len=seg_len,
    stride=stride,
    batch_size=batch_size,
    keep_only_positive_segments=False,
    neg_keep_prob=0.3,
)

ds_val = make_dataset(
    val_list,
    training=False,
    seg_len=seg_len,
    stride=stride,
    batch_size=batch_size,
    keep_only_positive_segments=False,
    neg_keep_prob=1.0,
)

train_ratio = compute_nrem_ratio(train_list)
val_ratio = compute_nrem_ratio(val_list)

print(f"[Train ratio] {train_ratio:.6f}")
print(f"[Val ratio]   {val_ratio:.6f}")


# Compile (IMPORTANT CHANGE)

lr = 1e-4

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss=focal_loss(alpha=0.5, gamma=2),
    metrics=[keras.metrics.AUC(name="pr_auc", curve="PR")],
)


# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(
            project_root / "RNN_LSTM" / "model_training" / "retrained_model.h5"
        ),
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True
    ),
]


# Train
epochs = 50
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=epochs,
    steps_per_epoch=1000,
    callbacks=callbacks,
    verbose=1,
)


# Save model
best_model_path = project_root / "RNN_LSTM" / "model_training" / "retrained_model.h5"
model.save(best_model_path)

print("Saved:", best_model_path)
print("Done.")
