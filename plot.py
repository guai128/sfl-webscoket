from matplotlib import pyplot as plt

directory = "history"
num_users = 20

# read data
all_data = [[] for _ in range(num_users)]
for i in range(num_users):
    with open(f"{directory}/client{i}_loss_accuracy.txt", "r") as f:
        for line in f:
            data = line.split()
            all_data[i].append([float(d) for d in data])

# plot
plt.figure()
plt.subplot(2, 2, 1)
for i in range(num_users):
    losses = [data[0] for data in all_data[i]]
    plt.plot(losses, label=f'client{i} train loss')
    plt.title('Train Loss')

plt.subplot(2, 2, 2)
for i in range(num_users):
    accuracies = [data[1] for data in all_data[i]]
    plt.plot(accuracies, label=f'client{i} train accuracy')
    plt.title('Train Accuracy')

plt.subplot(2, 2, 3)
for i in range(num_users):
    losses = [data[2] for data in all_data[i]]
    plt.plot(losses, label=f'client{i} test loss')
    plt.title('Test Loss')

plt.subplot(2, 2, 4)
for i in range(num_users):
    accuracies = [data[3] for data in all_data[i]]
    plt.plot(accuracies, label=f'client{i} test accuracy')
    plt.title('Test Accuracy')

plt.show()


