import tensorflow as tf
import random


from DQN import DQN
from ReplayBuffer import ReplayBuffer


class DQNAgent:
    def __init__(self, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):

        self.dqn = DQN()
        self.target_dqn = DQN()


        self.target_dqn.set_weights(self.dqn.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.memory = ReplayBuffer(50000)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.training_loss = []
        self.training_accuracy = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(10)

        state = tf.expand_dims(state, 0)
        q_values = self.dqn(state)
        return tf.argmax(q_values[0]).numpy()

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):

        next_q_values = self.target_dqn(next_states)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        # target_q = rewards + (1 - dones) * self.gamma * max_next_q

        target_q = rewards

        with tf.GradientTape() as tape:

            current_q = self.dqn(states)

            action_masks = tf.one_hot(actions, 10)
            predicted_q = tf.reduce_sum(current_q * action_masks, axis=1)

            # Compute loss
            loss = tf.reduce_mean(tf.square(target_q - predicted_q))

        # Update weights
        gradients = tape.gradient(loss, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables))

        return loss

    def train_batch(self, batch_size):
        if len(self.memory) < batch_size:
            return 0


        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)


        states = tf.stack(states)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.stack(next_states)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Train
        loss = self.train_step(states, actions, rewards, next_states, dones)

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.numpy()

    def update_target_network(self):
        self.target_dqn.set_weights(self.dqn.get_weights())

    @tf.function
    def evaluate_step(self, images):
        q_values = self.dqn(images)
        return tf.argmax(q_values, axis=1)

    def evaluate(self, test_dataset):
        correct = 0
        total = 0

        for images, labels in test_dataset:
            predicted = self.evaluate_step(images)
            # correct += tf.reduce_sum(tf.cast(predicted == labels, tf.int32))
            # total += len(labels)
            labels = tf.cast(labels, predicted.dtype)
            correct += tf.reduce_sum(tf.cast(predicted == labels, tf.int32))
            total += len(labels)

        return (correct / total * 100).numpy()