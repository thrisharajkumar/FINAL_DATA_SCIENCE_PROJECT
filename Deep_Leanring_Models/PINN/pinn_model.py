import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError


class PINN(Model):
    def __init__(self, input_dim, output_dim, hidden_units=128, num_layers=3, dropout_rate=0.1):
        super(PINN, self).__init__()
        self.hidden_layers = []
        for _ in range(num_layers):
            self.hidden_layers.append(Dense(hidden_units, activation='relu'))
            self.hidden_layers.append(Dropout(dropout_rate))
        self.output_layer = Dense(output_dim, activation='linear')
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mse = MeanSquaredError()

    def call(self, inputs):
        x = inputs["features"]
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def compute_physics_loss(self, inputs, predictions):
        Vbus = tf.expand_dims(inputs["Vbus"], axis=1)

        rise_pred = predictions[:, 0]
        fall_pred = predictions[:, 2]
        ring_pred = predictions[:, -1]
        overshoot_pred = predictions[:, 8]

        overshoot_norm_pred = overshoot_pred / (Vbus[:, 0] + 1e-6)

        physics_loss = (
            self.mse(rise_pred, inputs["t_rise_est"]) +
            self.mse(fall_pred, inputs["t_fall_est"]) +
            self.mse(ring_pred, inputs["f_ring_est"]) +
            self.mse(overshoot_norm_pred, inputs["overshoot_norm_1"])
        )
        return physics_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            preds = self({"features": x, **x}, training=True)
            data_loss = self.mse(y, preds)
            physics_loss = self.compute_physics_loss(x, preds)
            total_loss = data_loss + 0.2 * physics_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        preds = self({"features": x, **x}, training=False)
        total_loss = self.mse(y, preds)
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}
