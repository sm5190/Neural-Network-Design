import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(5806)

#%%
print("------------------Ques1------------------------------")

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize the Multilayer Perceptron (MLP)
def initialize_mlp(num_inputs, neurons_per_hidden, num_outputs):
    layers = [num_inputs] + list(neurons_per_hidden) + [num_outputs]
    weights = []
    biases = []

    for i in range(len(layers) - 1):
        weight = np.random.randn(layers[i], layers[i+1])
        bias = np.random.randn(layers[i + 1])
        weights.append(weight)
        biases.append(bias)



    return weights, biases

# Feedforward process
def feedforward(inputs, weights, biases):
    activations = [inputs]
    pre_activations = []


    for i, (w, b) in enumerate(zip(weights, biases)):
        z = np.dot(activations[-1], w) + b
        pre_activations.append(z)

        if i == len(weights) - 1:

            a = z
        else:

            a = sigmoid(z)
        activations.append(a)

    return activations, pre_activations




num_hidden_layers = int(input("Enter the number of hidden layers: "))
neurons_per_hidden = []
for i in range(num_hidden_layers):
    neurons = int(input(f"Enter the number of neurons for hidden layer {i + 1}: "))
    neurons_per_hidden.append(neurons)
num_outputs = int(input("Enter the number of output neurons: "))
num_observations = int(input("Enter the number of input samples: "))
input_dim=int(input("Enter the  input dimension: "))

weights, biases = initialize_mlp(input_dim, neurons_per_hidden, num_outputs)


p = np.linspace(-2 * np.pi, 2 * np.pi, num_observations)
inputs = np.sin(p).reshape(-1, 1)


activations, _= feedforward(inputs, weights, biases)


output_activations= activations[-1].reshape(-1)  # Flatten the output for plotting


plt.figure(figsize=(10, 6))
plt.plot(p, output_activations, label='MLP Response')
plt.plot(p, np.sin(p), label='sin(p)', linestyle='--')
plt.legend()
plt.title('Response of Untrained 4-layer MLP Network (3 Hidden Layers)')
plt.xlabel('p')
plt.ylabel('Activation / sin(p)')
plt.show()


for i, (w, b) in enumerate(zip(weights, biases), start=1):
    print(f"Layer {i} Weights:\n{w}\nBiases:\n{b}")

#%%
#####Ques2

print("------------------Ques2------------------------------")
# Backpropagation process

def backpropagate(activations, targets, weights):
    deltas = []
    delta_M = -2 * (targets - activations[-1])  # Linear output layer sensitivity
    deltas.append(delta_M)

    for m in range(len(activations) - 2, 0, -1):
        derivative = sigmoid_derivative(activations[m])
        delta = np.dot(deltas[0], weights[m].T) * derivative
        deltas.insert(0, delta)

    return deltas

# Update weights and biases
def update_parameters(weights, biases, activations, deltas, learning_rate):
    for m in range(len(weights)):
        weights[m] -= learning_rate * np.dot(activations[m].T, deltas[m])
        biases[m] -= learning_rate * np.sum(deltas[m], axis=0)

    return weights, biases

# Training the network
def train_network(inputs, targets, neurons_per_hidden, num_outputs, learning_rate, max_iterations=200000, sse_cutoff=0.01):
    num_inputs = inputs.shape[1]
    weights, biases = initialize_mlp(num_inputs, neurons_per_hidden, num_outputs)
    print("weights:", weights)
    print("biases:", biases)

    errors = [1000]
    for iteration in range(max_iterations):
        total_loss = 0
        for i in range(inputs.shape[0]):
            activations, _ = feedforward(inputs[i].reshape(-1,1), weights, biases)

            deltas = backpropagate(activations, targets[i].reshape(1, -1), weights)
            weights, biases = update_parameters(weights, biases, activations, deltas, learning_rate)

            loss = np.power((targets[i] - activations[-1]), 2).sum()
            # print(loss)
            total_loss += loss
        errors.append(total_loss)
        #print("total ", total_loss)

        if total_loss < sse_cutoff:
            print(f"Early stopping: SSE < {sse_cutoff} at iteration {iteration + 1}")
            break

    return weights, biases, errors, iteration + 1




def create_dataset(func, range_vals, num_samples):
    inputs = np.linspace(range_vals[0], range_vals[1], num_samples).reshape(num_samples, 1)
    targets = func(inputs)
    return inputs, targets



target_funcs = [
    (lambda p: np.sin(p), (-2 * np.pi, 2 * np.pi)),
    (lambda p: np.power(p, 2), (-2, 2)),
    (lambda p: np.exp(p), (0, 2)),
    (lambda p: np.power(np.sin(p), 2) + np.power(np.cos(p), 3), (-2 * np.pi, 2 * np.pi))
]




def customized_plot_results(inputs, targets, weights, biases, final_iteration, layers, sse_history, function_number):
    activations, _= feedforward(inputs, weights, biases)
    output = activations[-1].flatten()

    # Plot target vs network response
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title(f'Function {function_number}: Target vs Network Response\n'
              f'Number of iterations = {final_iteration}\n'
              f'Number of layers = {len(layers)}\n'
              f'Number of neurons = {layers}\n'
              f'Learning ratio = {learning_rate}\n'
              f'SSE error cut off = {sse_cutoff}')
    plt.plot(inputs.flatten(), targets.flatten(), 'g-', label='Target')
    plt.scatter(inputs.flatten(), output, label='Network after training', color='red')
    plt.xlabel('p')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title(f'Sum of squared errors vs Number of iterations\n'
              f'SSE={np.round(sse_history[-1], 3)}\n'
              f'Learning ratio = {learning_rate}\n'
              f'Number of neurons = {layers}\n'
              f'SSE error cut off = {sse_cutoff}')
    plt.plot(range(1, len(sse_history) + 1), sse_history, 'b-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of iterations (log scale)')
    plt.ylabel('Sum of squared errors (log scale)')

    plt.tight_layout()
    plt.grid(True)
    plt.show()

choice = int(input("Input function choice (1-4): "))
print(f"For function no {choice}:")


num_hidden_layers = int(input("Enter the number of hidden layers: "))
neurons_per_hidden = []
for j in range(num_hidden_layers):
    neurons = int(input(f"Enter the number of neurons for hidden layer {j + 1}: "))
    neurons_per_hidden.append(neurons)
num_outputs = int(input("Enter the number of output neurons: "))
num_samples = int(input("Enter the number of input samples: "))



learning_rate = float(input("Enter the learning rate: "))
max_iterations = int(input("Enter the maximum number of iterations: "))
sse_cutoff = float(input("Enter the SSE cutoff: "))


layers = neurons_per_hidden + [num_outputs]

func, range_vals = target_funcs[choice-1]
inputs, targets = create_dataset(func, range_vals, num_samples)


weights, biases, sse_history, final_iteration = train_network(
                inputs, targets,
                neurons_per_hidden=neurons_per_hidden,
                num_outputs=num_outputs,
                learning_rate=learning_rate,
                max_iterations=max_iterations,
                sse_cutoff=sse_cutoff,
            )


customized_plot_results(inputs, targets, weights, biases, final_iteration, layers, sse_history, choice)

#%%
##Ques 3

print("------------------Ques3------------------------------")
learning_rates = [0.01, 0.03, 0.05, 0.07, 0.09]


results = []



num_hidden_layers = int(input("Enter the number of hidden layers: "))
neurons_per_hidden = []
for j in range(num_hidden_layers):
    neurons = int(input(f"Enter the number of neurons for hidden layer {j + 1}: "))
    neurons_per_hidden.append(neurons)
num_outputs = int(input("Enter the number of output neurons: "))
num_samples = int(input("Enter the number of input samples: "))
input_dim = int(input("Enter the input dimension: "))
max_iterations = int(input("Enter the maximum number of iterations: "))
learning_rate = float(input("Enter the learning rate: "))


layers = neurons_per_hidden + [num_outputs]

func, range_vals= target_funcs[0]


inputs, targets = create_dataset(func, range_vals, num_samples)


weights, biases, sse_history, final_iteration = train_network(
        inputs,
        targets,
        neurons_per_hidden=neurons_per_hidden,
        num_outputs=num_outputs,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        sse_cutoff=0.01,


    )
results.append({
        'α (Learning Rate)': learning_rate,
        'SSE': sse_history[-1],
        '# of observations': 100,
        'SSE Cut off': 0.01,
        '# of iterations': final_iteration
    })


results_df = pd.DataFrame(results)

print(results_df)

# ###Ques 4
#%%

print("------------------Ques4------------------------------")


# Define the mychirp function
def mychirp(t, f0, t1, f1, phase):
    t0 = t[0]
    T = t1 - t0
    k = (f1 - f0) / T
    x = np.sin(2 * np.pi * (f0 * t + (k / 2) * t**2) + phase)
    return x



f = 100
N = 1000  
step = 1 / f
t0 = 0
t1 = 1
x = np.linspace(t0, t1, N)
f0 = 1
f1 = f / 20
t = mychirp(x, f0, t1, f1, phase=0)

inputs = x.reshape(-1, 1)
targets = t.reshape(-1, 1)

num_hidden_layers = int(input("Enter the number of hidden layers: "))
neurons_per_hidden = []
for j in range(num_hidden_layers):
    neurons = int(input(f"Enter the number of neurons for hidden layer {j + 1}: "))
    neurons_per_hidden.append(neurons)
num_outputs = int(input("Enter the number of output neurons: "))
num_samples = int(input("Enter the number of input samples: "))
input_dim = int(input("Enter the input dimension: "))
max_iterations = int(input("Enter the maximum number of iterations: "))
sse_cutoff= float(input("Enter the SSE cutoff: "))
learning_rate = float(input("Enter the learning rate: "))

layers = neurons_per_hidden + [num_outputs]



weights, biases, sse_history, final_iteration = train_network(
    inputs, targets,
    neurons_per_hidden=neurons_per_hidden,
    num_outputs=num_outputs,
    learning_rate=learning_rate,
    max_iterations=max_iterations,
    sse_cutoff=sse_cutoff
)

customized_plot_results(inputs, targets, weights, biases, final_iteration,layers, sse_history, "mychirp")


for i, (w, b) in enumerate(zip(weights, biases), start=1):
    print(f"Layer {i} Weights:\n{w}\nBiases:\n{b}")
#%%

print("------------------Ques5------------------------------")


def initialize_mlp_momentum(num_inputs, neurons_per_hidden, num_outputs):
    layers = [num_inputs] + list(neurons_per_hidden) + [num_outputs]
    weights = []
    biases = []
    velocity_weights = []
    velocity_biases = []


    for i in range(len(layers) - 1):
        weight = np.random.randn(layers[i], layers[i + 1])
        bias = np.random.randn(layers[i + 1])
        velocity_w = np.zeros_like(weight)
        velocity_b = np.zeros_like(bias)
        weights.append(weight)
        biases.append(bias)
        velocity_weights.append(velocity_w)
        velocity_biases.append(velocity_b)

    return weights, biases, velocity_weights, velocity_biases




def update_parameters_with_momentum(weights, biases, activations, deltas, learning_rate, velocity_weights, velocity_biases, beta):
    for m in range(len(weights)):

        velocity_weights[m] = beta * velocity_weights[m] - (1-beta)* learning_rate * np.dot(activations[m].T, deltas[m])
        velocity_biases[m] = beta * velocity_biases[m] - (1-beta) * learning_rate * np.sum(deltas[m], axis=0)


        weights[m] += velocity_weights[m]
        biases[m] += velocity_biases[m]

    return weights, biases, velocity_weights, velocity_biases



def train_network_momentum(inputs, targets, neurons_per_hidden, num_outputs, learning_rate, beta, sse_cutoff, max_iterations):
    num_inputs = inputs.shape[1]
    weights, biases, velocity_weights, velocity_biases = initialize_mlp_momentum(num_inputs, neurons_per_hidden, num_outputs)

    errors = [1000]
    for iteration in range(max_iterations):
        total_loss = 0
        for i in range(inputs.shape[0]):
            activations, _ = feedforward(inputs[i].reshape(1, -1), weights, biases)
            loss = 0.5*np.power((targets[i] - activations[-1]), 2).sum()
            total_loss += loss

            deltas = backpropagate(activations, targets[i].reshape(1, -1), weights)
            weights, biases, velocity_weights, velocity_biases = update_parameters_with_momentum(
                weights, biases, activations, deltas, learning_rate, velocity_weights, velocity_biases, beta)

        errors.append(total_loss)

        if total_loss < sse_cutoff:
            print(f"Early stopping: SSE < {sse_cutoff} at iteration {iteration + 1}")
            break
    return weights, biases, errors, iteration + 1



def customized_plot_results_momentum(inputs, targets, weights, biases, final_iteration,beta ,layers, sse_history, sse_cutoff, function_number):
    activations, _ = feedforward(inputs, weights, biases)
    output = activations[-1].flatten()


    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title(f'Function {function_number}: Target vs Network Response\n'
              f'Number of iterations = {final_iteration}\n'
              f'Number of layers = {len(layers)}\n'
              f'Number of neurons = {layers}\n'
              f'Learning ratio = {learning_rate}\n'

              f'SSE error cut off = {sse_cutoff}')
    plt.plot(inputs.flatten(), targets.flatten(), 'g-', label='Target')
    plt.scatter(inputs.flatten(), output, label='Network after training', color='red')
    plt.xlabel('p')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title(f'Sum of squared errors vs Number of iterations\n'
              f'SSE={np.round(sse_history[-1], 3)}\n'
              f'Learning ratio = {learning_rate}\n'
              f'Momentum Term= {beta}\n'
              f'Number of neurons = {layers}\n'
              f'SSE error cut off = {sse_cutoff}')
    plt.plot(range(1, len(sse_history)+1), sse_history, 'b-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of iterations (log scale)')
    plt.ylabel('Sum of squared errors (log scale)')

    plt.tight_layout()
    plt.grid(True)
    plt.show()



choice = int(input("Input function choice (1-4): "))
print(f"For function no {choice}:")

num_hidden_layers = int(input("Enter the number of hidden layers: "))
neurons_per_hidden = []
for j in range(num_hidden_layers):
    neurons = int(input(f"Enter the number of neurons for hidden layer {j + 1}: "))
    neurons_per_hidden.append(neurons)
num_outputs = int(input("Enter the number of output neurons: "))
num_samples = int(input("Enter the number of input samples: "))


learning_rate = float(input("Enter the learning rate: "))
max_iterations = int(input("Enter the maximum number of iterations: "))
sse_cutoff = float(input("Enter the SSE cutoff: "))
beta = float(input("Enter the Momentum Term: "))

layers = neurons_per_hidden + [num_outputs]


func, range_vals = target_funcs[choice-1]
inputs, targets = create_dataset(func, range_vals, num_samples)


weights, biases, sse_history, final_iteration = train_network_momentum(
        inputs, targets,
        neurons_per_hidden=neurons_per_hidden,
        num_outputs=num_outputs,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        sse_cutoff=sse_cutoff,
        beta= beta
    )


customized_plot_results_momentum(inputs, targets, weights, biases, final_iteration,beta ,layers, sse_history,sse_cutoff, i + 1)


#%%

print("------------------Ques6------------------------------")
#momentums = [0.2, 0.4, 0.6, 0.8, 0.9]


results = []



num_hidden_layers = int(input("Enter the number of hidden layers: "))
neurons_per_hidden = []
for j in range(num_hidden_layers):
    neurons = int(input(f"Enter the number of neurons for hidden layer {j + 1}: "))
    neurons_per_hidden.append(neurons)
num_outputs = int(input("Enter the number of output neurons: "))
num_samples = int(input("Enter the number of input samples: "))
input_dim = int(input("Enter the input dimension: "))
max_iterations = int(input("Enter the maximum number of iterations: "))
learning_rate = float(input("Enter the learning rate: "))
beta= float(input("Enter the momentum: "))


layers = neurons_per_hidden + [num_outputs]

func, range_vals= target_funcs[0]


inputs, targets = create_dataset(func, range_vals, num_samples)

weights, biases, sse_history, final_iteration = train_network_momentum(
        inputs,
        targets,
        #num_hidden_layers=len(layers) - 1,
        neurons_per_hidden=neurons_per_hidden,
        num_outputs=num_outputs,
        learning_rate=learning_rate,
        #batch_size=10,
        max_iterations=max_iterations,
        sse_cutoff=0.01,
        beta= beta


    )

results.append({
        'Momentum': beta,
        'SSE': sse_history[-1],
        '# of observations': 100,
        'Learning Rate': 0.02,
        'SSE Cut off': 0.01,
        '# of iterations': final_iteration
    })


results_df = pd.DataFrame(results)

print(results_df)
