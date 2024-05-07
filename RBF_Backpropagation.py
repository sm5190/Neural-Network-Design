import numpy as np
import matplotlib.pyplot as plt

def radbas(n):
    return np.exp(-n**2)

# Parameters
w1 = np.array([-1, 1])
b1 = 2
w2 = np.array([1, 1])
b2 = 0

# Input ranges
p1 = np.arange(-2, 2.5, 0.5)
p2 = np.linspace(-2, 2, 1000)

def compute_output(p, w1, b1, w2, b2):

    n1 = np.sqrt((p - w1)**2) * b1
    a1 = radbas(n1)

    n2 = np.dot(w2, a1) + b2
    a2 = n2
    return a2

# Compute outputs
outputs_p1 = [compute_output(p, w1, b1, w2, b2) for p in p1]
outputs_p2 = [compute_output(p, w1, b1, w2, b2) for p in p2]

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(p1, outputs_p1, 'o-', label='Step 0.5', linewidth=2)
plt.title('Network Output vs. Input (Step Size 0.5)')
plt.xlabel('Input p')
plt.ylabel('Output')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(p2, outputs_p2, '-', label='1000 Samples', linewidth=2)
plt.title('Network Output vs. Input (1000 Samples)')
plt.xlabel('Input p')
plt.ylabel('Output')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
###2



def simulate_rbf_network(mean, variance, p_lower, p_upper, num_samples, num_neurons_hidden, num_neurons_output):

    np.random.seed(5806)


    p = np.linspace(p_lower, p_upper, num_samples)


    w1 = mean + np.sqrt(variance) * np.random.randn(num_neurons_hidden)
    b1 = np.random.rand(num_neurons_hidden)
    w2 = np.random.randn(num_neurons_hidden, num_neurons_output)
    b2 = np.random.randn(num_neurons_output)


    outputs = np.zeros(num_samples)
    for i in range(num_samples):
        n1 = np.sqrt((p[i] - w1)**2) *b1
        a1 = radbas(n1)
        n2 = np.dot(a1, w2) + b2
        outputs[i] = n2

    # Plot the network response
    plt.figure(figsize=(10, 5))
    plt.plot(p, outputs, linewidth=2, color='blue', label='RBF response without training')
    plt.title('RBF Response with 10 neurons in the hidden layer')
    plt.xlabel('Input p')
    plt.ylabel('Mag')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Display the weights and biases
    print("Weights of the first layer (w1):", w1)
    print("Biases of the first layer (b1):", b1)
    print("Weights of the second layer (w2):", w2)
    print("Biases of the second layer (b2):", b2)

# # Parameters for the given task
# mean = 0  # Mean of the normally distributed weights
# variance = 1  # Variance of the normally distributed weights
# p_lower = -2  # Lower bound for the input p
# p_upper = 2  # Upper bound for the input p
# num_samples = 100  # Number of samples
# num_neurons_hidden = 10  # Number of neurons in the hidden layer
# num_neurons_output = 1  # Number of neurons in the output layer


# Asking user for input
mean = float(input("Enter the mean of the normally distributed weights: "))
variance = float(input("Enter the variance of normally distributed weights: "))
p_lower = float(input("Enter the lower bound for the input p: "))
p_upper = float(input("Enter the upper bound for the input p: "))
num_samples = int(input("Enter the number of samples: "))
num_neurons_hidden = int(input("Enter the number of neurons in the hidden layer: "))
num_neurons_output = int(input("Enter the number of neurons in the output layer: "))

# Call the simulation function with specified parameters
simulate_rbf_network(mean, variance, p_lower, p_upper, num_samples, num_neurons_hidden, num_neurons_output)

###q4
# Given parameters
# Given parameters

w21_range = np.linspace(0, 2, 5)
b1_range = np.linspace(0.5, 8, 5)
w11_range = np.linspace(-1, 1, 5)
b2_range = np.linspace(-1, 1, 5)

# Input values
p = np.linspace(-2, 2, 100)

# Initialize subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 7))

# Plot (a): Varying b2
for b1 in b1_range:
    w1 = np.array([-1, 1])
    b1 = np.array([2, b1])
    b1_blue=np.array([2,2])
    w2 = np.array([1, 1])
    b2=0
    outputs = np.dot(radbas(np.sqrt((p[:, None] - w1)**2) * b1), w2) + b2
    output=np.dot(radbas(np.sqrt((p[:, None] - w1)**2) * b1_blue), w2) + b2

    axs[0, 0].plot(p, outputs, linewidth=2, color='red')
    axs[0, 0].plot(p, output, linewidth=2, color='blue')


# Plot (b): Varying w21
for w21 in w21_range:
    w1 = np.array([-1, w21])
    w1_blue = np.array([-1, 1])
    b1= np.array([2, 2])
    w2 = np.array([1, 1])
    b2=0
    outputs = np.dot(radbas(np.sqrt((p[:, None] - w1) ** 2) * b1), w2) + b2
    output = np.dot(radbas(np.sqrt((p[:, None] - w1_blue) ** 2) * b1), w2) + b2

    axs[0, 1].plot(p, outputs, linewidth=2, color='red')
    axs[0, 1].plot(p, output, linewidth=2, color='blue')

# Plot (c): Varying w11
for w11 in w11_range:
    w1 = np.array([-1, 1])
    b1 = np.array([2, 2])
    w2 = np.array([w11, 1])
    w2_blue = np.array([1, 1])
    b2 = 0
    outputs = np.dot(radbas(np.sqrt((p[:, None] - w1) ** 2) * b1), w2) + b2
    output = np.dot(radbas(np.sqrt((p[:, None] - w1) ** 2) * b1), w2_blue) + b2

    axs[1, 0].plot(p, outputs, linewidth=2, color='red')
    axs[1, 0].plot(p, output, linewidth=2, color='blue')

# Plot (d): Varying b1
for b2 in b2_range:
    w1 = np.array([-1, 1])
    b1 = np.array([2, 2])
    w2 = np.array([1, 1])

    b2 = np.array([b2])
    b2_blue=0
    outputs = np.dot(radbas(np.sqrt((p[:, None] - w1) ** 2) * b1), w2) + b2
    output = np.dot(radbas(np.sqrt((p[:, None] - w1) ** 2) * b1), w2) + b2_blue
    axs[1, 1].plot(p, outputs, linewidth=2, color='red')
    axs[1, 1].plot(p, output, linewidth=2, color='blue')

# Set titles and labels
axs[0, 0].set_title('(a)')
axs[0, 1].set_title('(b)')
axs[1, 0].set_title('(c)')
axs[1, 1].set_title('(d)')
for ax in axs.flat:
    ax.set(xlabel='p', ylabel='RBF Response')

# Adjust layout
plt.tight_layout()
plt.show()

# ###Ques 5
#%%

# Define the target function
def target_function(p):
    return np.sin(p)

# Define the radial basis function and its derivative
def radial_basis_function(n):
    return np.exp(-n**2)

def radial_basis_function_derivative(n):
    return -2 * n * radial_basis_function(n)

# Initialize the weights and biases
def initialize_rbf(input_dim, hidden_neurons, output_neurons):
    np.random.seed(5806)

    w1 = np.random.randn(hidden_neurons, input_dim)
    b1 = np.random.rand(hidden_neurons)
    w2 = np.random.randn(hidden_neurons, output_neurons)


    b2 = np.random.randn(output_neurons)
    return w1, b1, w2, b2


def train_rbf(inputs, targets, w1, b1, w2, b2, learning_rate, max_iterations, sse_threshold):
    errors = []

    for iteration in range(max_iterations):
        total_loss = 0
        outputs=[]
        for input_vec, target in zip(inputs, targets):

            input_vec = input_vec.reshape(-1, 1)

            # Forward pass

            n1 = np.linalg.norm(w1-input_vec, axis=1)
            
            a1 = radial_basis_function(n1*b1)
            a1 = a1.reshape(-1, 1)
            n2 = np.dot(a1.T, w2) + b2
            #n2 = np.dot(w2, a1) + b2
            a2 = n2
            error = target - a2
            sse = np.power(error, 2)
            total_loss += sse.sum()

            # Backpropagation

            s2 = -2 * error

            s1 = radial_basis_function_derivative(n1) * (w2.T * s2)

            # Calculate gradients

            grad_w2 = a1 * s2

            grad_b2 = s2

            grad_w1 = np.dot(s1, b1* (w1 - input_vec).flatten()) / n1.reshape(-1, 1)

            grad_b1 = s1 * n1


            w2 -= learning_rate * grad_w2
            b2 -= learning_rate * grad_b2.flatten()
            w1 -= learning_rate * grad_w1
            b1 -= learning_rate * grad_b1.flatten()
            outputs.append(a2)

        errors.append(total_loss)

        if total_loss < sse_threshold:
            print(f"Early stopping: SSE < {sse_threshold} at iteration {iteration + 1}")
            break

    return outputs,w1, b1, w2, b2, errors, iteration+1



# Set up the RBF network
input_dim = 1
hidden_neurons =2
output_neurons = 1
learning_rate = 0.01
max_iterations = 2000
sse_threshold = 0.001

# Initialize network parameters
w1, b1, w2, b2 = initialize_rbf(input_dim, hidden_neurons, output_neurons)
print("initial weights:")
print("weights of the first layer:", w1)
print("weights of the 2nd layer:", w2)
print("bias of the first layer:", b1)
print("bias of the 2nd layer:", b2)

# Generate the inputs and targets
inputs = np.linspace(0, np.pi, 100)
targets = target_function(inputs)

# Train the RBF network
a1, w1, b1, w2, b2, errors, final_iteration= train_rbf(inputs, targets, w1, b1, w2, b2, learning_rate, max_iterations, sse_threshold)


# Ensure that outputs are converted to a flat NumPy array
outputs = np.array(a1).flatten()
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title(f'Function  Target vs Network Response\n'
              f'Number of iterations = {final_iteration}\n'
              f'Number of layers = {2}\n'
              f'Number of neurons = {2}\n'
              f'Learning ratio = {learning_rate}\n'
              f'SSE error cut off = {sse_threshold}')
plt.plot(inputs.flatten(), targets.flatten(), 'g-', label='Target')
plt.scatter(inputs.flatten(), outputs, label='Network after training', color='red')
plt.xlabel('p')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title(f'Sum of squared errors vs Number of iterations\n'
          f'SSE={np.round(errors[-1], 3)}\n'
          #f'SSE={0.0010}\n'
          f'Learning ratio ={ learning_rate}\n'
          f'Number of neurons = {2}\n'
          f'SSE error cut off = {sse_threshold}')
plt.plot(range(len(errors)), errors, color='b')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of iterations (log scale)')
plt.ylabel('Sum of squared errors (log scale)')

plt.tight_layout()
plt.grid(True)
plt.show()

print("finial weights:")

print("weights of the first layer:", w1)
print("weights of the 2nd layer:", w2)
print("bias of the first layer:", b1)
print("bias of the 2nd layer:", b2)

#%%
def train_rbf_with_momentum(inputs, targets, w1, b1, w2, b2, learning_rate, gamma, max_iterations, sse_threshold):
    errors = []

    # Initialize the momentum terms for weights and biases
    w1_update = np.zeros_like(w1)
    b1_update = np.zeros_like(b1)
    w2_update = np.zeros_like(w2)
    b2_update = np.zeros_like(b2)

    for iteration in range(max_iterations):
        total_loss = 0
        outputs = []
        for input_vec, target in zip(inputs, targets):
            input_vec = input_vec.reshape(-1, 1)

            # Forward pass
            n1 = np.linalg.norm(w1 - input_vec, axis=1)
            a1 = radial_basis_function(n1 * b1)
            a1 = a1.reshape(-1, 1)
            n2 = np.dot(a1.T, w2) + b2
            a2 = n2
            error = target - a2
            sse = np.power(error, 2)
            total_loss += sse.sum()

            # Backpropagation
            s2 = -2 * error
            s1 = radial_basis_function_derivative(n1) * (w2.T * s2)

            # Calculate gradients
            grad_w2 = a1 * s2
            grad_b2 = s2
            grad_w1 = np.dot(s1, b1 * (w1 - input_vec).flatten()) / n1.reshape(-1, 1)
            grad_b1 = s1 * n1

            # Update the weights and biases with momentum
            w1_update = gamma * w1_update - (1-gamma)* learning_rate * grad_w1
            b1_update = gamma * b1_update - (1-gamma)*learning_rate * grad_b1.flatten()
            w2_update = gamma * w2_update - (1-gamma)*learning_rate * grad_w2
            b2_update = gamma * b2_update - (1-gamma)*learning_rate * grad_b2.flatten()

            w1 += w1_update
            b1 += b1_update
            w2 += w2_update
            b2 += b2_update

            outputs.append(a2)

        errors.append(total_loss)

        if total_loss < sse_threshold:
            print(f"Early stopping: SSE < {sse_threshold} at iteration {iteration + 1}")
            break

    return outputs, w1, b1, w2, b2, errors, iteration + 1


# Parameters
learning_rate = 0.01
gamma = 0.4# Example momentum term
max_iterations = 2000
sse_threshold = 0.001


# Set up the RBF network
input_dim = 1
hidden_neurons =2
output_neurons = 1


# Initialize network parameters
#w1, b1, w2, b2 = initialize_rbf(input_dim, hidden_neurons, output_neurons)

# Initialize network parameters
w1, b1, w2, b2 = initialize_rbf(input_dim, hidden_neurons, output_neurons)
print("initial weights:")
print("weights of the first layer:", w1)
print("weights of the 2nd layer:", w2)
print("bias of the first layer:", b1)
print("bias of the 2nd layer:", b2)
# Train the RBF network with momentum
a1, w1, b1, w2, b2, errors, final_iteration = train_rbf_with_momentum(inputs, targets, w1, b1, w2, b2,
                                                                           learning_rate, gamma, max_iterations,
                                                                           sse_threshold)

# Ensure that outputs are converted to a flat NumPy array
outputs = np.array(a1).flatten()
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title(
          f'Number of iterations = {final_iteration}\n'
          f'Number of layers = {2}\n'
          f'Number of neurons = {2}\n'
          f'Momentum = {gamma}\n'
          f'Learning ratio = {learning_rate}\n'
          f'SSE error cut off = {sse_threshold}')
plt.plot(inputs.flatten(),targets.flatten(), 'g-', label='Target')
plt.scatter(inputs.flatten(), outputs, label='Network after training', color='red')
plt.xlabel('p')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title(f'Sum of squared errors vs Number of iterations\n'
          f'SSE={np.round(errors[-1], 3)}\n'
          #f'SSE={0.0010}\n'
          f'Learning ratio = {learning_rate}\n'
          f'Number of neurons = {2}\n'
          f'SSE error cut off = {sse_threshold}')
plt.plot(range(len(errors)), errors, color='b')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of iterations (log scale)')
plt.ylabel('Sum of squared errors (log scale)')

plt.tight_layout()
plt.grid(True)
plt.show()

# Final weights and biases after training with momentum
print("Final weights of the first layer (w1):", w1)
print("Final biases of the first layer (b1):", b1)
print("Final weights of the second layer (w2):", w2)
print("Final biases of the second layer (b2):", b2)


