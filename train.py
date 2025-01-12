import tensorflow as tf

from DQNAgent import DQNAgent


def train_mnist_dqn():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

    agent = DQNAgent()

    # Training parameters
    num_epochs = 10
    batch_size = 128
    update_target_every = 1000
    steps = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_ds):
            images = tf.cast(images, tf.float32)

            for i in range(len(images)):
                state = images[i]
                true_label = labels[i].numpy()

                # Select action
                action = agent.select_action(state)

                # Calculate reward
                reward = 1.0 if action == true_label else -1.0

                # Store transition
                agent.memory.push(state, action, reward, state, True)

                # Train
                loss = agent.train_batch(batch_size)
                if loss > 0:
                    epoch_loss += loss

                # Update target network
                steps += 1
                if steps % update_target_every == 0:
                    agent.update_target_network()

                # Track accuracy
                correct += (action == true_label)
                total += 1

            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'Loss: {epoch_loss/(batch_idx+1):.4f}, '
                      f'Accuracy: {100.*correct/total:.2f}%, '
                      f'Epsilon: {agent.epsilon:.4f}')

        # Evaluate on test set
        test_accuracy = agent.evaluate(test_ds)
        print(f'Epoch {epoch} Test Accuracy: {test_accuracy:.2f}%')

        agent.training_loss.append(epoch_loss/len(train_ds))
        agent.training_accuracy.append(100.*correct/total)

    return agent