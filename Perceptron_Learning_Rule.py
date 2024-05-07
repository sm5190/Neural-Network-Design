import numpy as np
import matplotlib.pyplot as plt
import numpy as np
print("-------------------Ques 1------------------------------")
# Prototype vectors
prototypes = {
    'I': np.array([[1, 4], [1, 5], [2, 4], [2, 5]]),
    'II': np.array([[3, 1], [3, 2], [4, 1], [4, 2]])
}

font1 = {'family':'serif', 'color':'darkred', 'size':30}
font2 = {'family':'serif', 'color':'blue', 'size':25}
font3 = {'family':'serif', 'color':'darkred', 'size':20}

# Plot prototypes
plt.figure(figsize=(8, 8))

for cls, data in prototypes.items():
    if cls == 'I':
        plt.scatter(data[:, 0], data[:, 1], color='red', marker='s', s=200, label='Class I')
    elif cls == 'II':
        plt.scatter(data[:, 0], data[:, 1], color='blue', marker='D', s=200, label='Class II')

plt.legend()
plt.title('Prototypes', fontdict=font1)
plt.xlabel('Weight', fontdict=font2)
plt.ylabel('Ear Length', fontdict=font2)
plt.xlim(-2, 7)
plt.ylim(-2, 7)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid("True")
plt.show()

print("-------------------Ques 2------------------------------")
def activate(activations):
    return np.where(activations >= 0, 1, 0)
def perceptron_learning_rule(prototypes, initial_weights, initial_bias):
    weights = initial_weights
    bias = initial_bias
    condition = True
    decision_boundaries = []

    while(condition):

        all_errors_zero = 0
        for cls, data in prototypes.items():
            for x in data:
                y = activate(np.dot(weights, x) + bias)
                desired_output = 1 if cls == 'I' else 0
                error = desired_output - y

                if error != 0:
                    weights += (error* x) # Update weights
                    bias += error  # Update bias
                all_errors_zero += np.power(error,2)
        decision_boundaries.append((-weights[0] / weights[1], -bias / weights[1]))
        if all_errors_zero==0:
            condition=False
    return weights, bias, decision_boundaries

# Parameters
initial_weights = np.random.normal(0, 5, 2)
initial_bias = np.random.normal(0, 5)
print("Initial Weights:", np.round(initial_weights,2))
print("Initial Bias:", np.round(initial_bias,2))

# Training perceptron
final_weights, final_bias, decision_boundaries= perceptron_learning_rule(prototypes, initial_weights, initial_bias)

print("Final Weights:", np.round(final_weights,2))
print("Final Bias:", np.round(final_bias,2))


# Plotting decision boundary
x_range = np.linspace(-2, 7, 100)
#decision_boundary = (-final_weights[0] * x_range - final_bias) / final_weights[1]

plt.figure(figsize=(8, 8))

for (m, c) in decision_boundaries:
    x_vals = np.linspace(-2, 7, 100)
    y_vals = c + m * x_vals
    plt.plot(x_vals, y_vals, color='black', linewidth=4)

for cls, data in prototypes.items():
    if cls == 'I':
        plt.scatter(data[:, 0], data[:, 1], color='red', marker='s', s=200, label='Class I')
    elif cls == 'II':
        plt.scatter(data[:, 0], data[:, 1], color='blue', marker='D', s=200, label='Class II')

#plt.plot(x_range, decision_boundary, color='green', linewidth=4)
plt.legend()
plt.title('Prototype Classification', fontdict=font1)
plt.xlabel('Weight', fontdict=font2)
plt.ylabel('Ear Length', fontdict=font2)
plt.xlim(-2, 7)
plt.ylim(-2, 7)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid("True")
plt.show()



print("-------------------Ques 3------------------------------")

# Verification
for cls, data in prototypes.items():
    for x in data:
        y = 1 if np.dot(final_weights, x) + final_bias >= 0 else 0
        print(f"Prototype {x} is classified as Class {cls} with output {y}")



print("-------------------Ques 4------------------------------")

noise_radius = 2.5
num_samples = 800
noise_levels=0.5
def generate_noisy_data(data, radius, num_points, noise_level):
    noisy_data = {}
    np.random.seed(5806)


    for class_label, class_data in data.items():

        center = np.mean(class_data, axis=0)


        angles = np.random.uniform(0, 2*np.pi, num_points)
        distances = noise_level* np.sqrt(np.random.uniform(0, radius**2, num_points))
        noisy_x_coords = center[0] + distances * np.cos(angles)
        noisy_y_coords = center[1] + distances * np.sin(angles)


        noisy_points = np.column_stack((noisy_x_coords, noisy_y_coords))
        noisy_data[class_label] = noisy_points

    return noisy_data
noisy_prototypes= generate_noisy_data(prototypes,radius= noise_radius,num_points=num_samples,noise_level=noise_levels)




final_weights, final_bias, decision_boundaries = perceptron_learning_rule(noisy_prototypes, initial_weights, initial_bias)

print("Final Weights:", np.round(final_weights))
print("Final Bias:", np.round(final_bias))
# print("Final iteration:", iteration)

# Plotting decision boundary
x_range = np.linspace(-2, 7, 100)
decision_boundary = (-final_weights[0] * x_range - final_bias) / final_weights[1]

plt.figure(figsize=(8, 8))

for cls, data in noisy_prototypes.items():
    if cls == 'I':
        plt.scatter(data[:, 0], data[:, 1], color='red', marker='s', s=200, label='Class I')
    elif cls == 'II':
        plt.scatter(data[:, 0], data[:, 1], color='blue', marker='D', s=200, label='Class II')

plt.plot(x_range, decision_boundary, color='green', linewidth=4)
plt.legend()
plt.title('Classification with Optimum Decision Boundary', fontdict=font3)
plt.xlabel('Weight', fontdict=font2)
plt.ylabel('Ear Length', fontdict=font2)
plt.xlim(-2, 7)
plt.ylim(-2, 7)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid("True")
plt.show()

#increase noise to overlap


print("-------------------Ques 5------------------------------")
def perceptron_learning_rule_iterative(prototypes, initial_weights, initial_bias, max_iterations=500):
    weights = initial_weights
    bias = initial_bias


    for _ in range(max_iterations):

        all_errors_zero = 0
        for cls, data in prototypes.items():
            for x in data:
                y = activate(np.dot(weights, x) + bias)
                desired_output = 1 if cls == 'I' else 0
                error = desired_output - y

                if error != 0:
                    weights += (error* x) # Update weights
                    bias += error  # Update bias
                all_errors_zero += np.power(error,2)

        if all_errors_zero==0:
            break;


    return weights, bias


noise_radius = 5
num_samples = 800
noise_levels=0.5
noisy_prototypes_overlap=generate_noisy_data(prototypes, noise_radius,num_samples,noise_levels)

final_weights1, final_bias1 = perceptron_learning_rule_iterative(noisy_prototypes_overlap, initial_weights, initial_bias)

# Plotting decision boundary
x_range = np.linspace(-2, 7, 100)
decision_boundary1 = (-final_weights1[0] * x_range - final_bias1) / final_weights1[1]

plt.figure(figsize=(8, 8))

for cls, data in noisy_prototypes_overlap.items():
    if cls == 'I':
        plt.scatter(data[:, 0], data[:, 1], color='red', marker='s', s=200, label='Class I')
    elif cls == 'II':
        plt.scatter(data[:, 0], data[:, 1], color='blue', marker='D', s=200, label='Class II')

plt.plot(x_range, decision_boundary1, color='black', linewidth=4)
plt.legend()
plt.title('Classification with Overlap of Noise', fontdict=font3)
plt.xlabel('Weight', fontdict=font2)
plt.ylabel('Ear Length', fontdict=font2)
plt.xlim(-2, 7)
plt.ylim(-2, 7)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid("True")
plt.show()

print("-------------------Ques 6------------------------------")

additional_prototypes = {

    'III': np.array([[-1, 1], [-1, 2], [0, 1], [0, 2]]),
    'IV': np.array([[2, 0], [2, -1], [3, 0], [3, -1]])
}

# Combine original and additional prototypes
all_prototypes = {**prototypes, **additional_prototypes}




# Plot all prototypes
plt.figure(figsize=(8, 8))
for cls, data in all_prototypes.items():
    if cls == 'I':
        plt.scatter(data[:, 0], data[:, 1], color='red', marker='s', s=200, label='Class I')
    elif cls == 'II':
        plt.scatter(data[:, 0], data[:, 1], color='blue', marker='D', s=200, label='Class II')

    elif cls == 'III':
        plt.scatter(data[:, 0], data[:, 1], color='green', marker='*', s=200, label='Class III')

    elif cls == 'IV':
        plt.scatter(data[:, 0], data[:, 1], color='cyan', marker='o', s=200, label='Class IV')

data2 = {(0, 0): np.array([[1, 4], [1, 5], [2, 4], [2, 5]]),
        (0, 1): np.array([[3, 1], [3, 2], [4, 1], [4, 2]]),
        (1, 0): np.array([[-1, 1], [-1, 2], [0, 1], [0, 2]]),
        (1, 1): np.array([[2, 0], [2, -1], [3, 0], [3, -1]])}


initial_weights = np.random.normal(loc=0, scale=5, size=(2, 2))


initial_bias = np.random.normal(loc=0, scale=5, size=(1, 2))

def predict(inputs, weights, bias):
    activations = np.dot(weights,inputs) + bias
    activations = np.array([activate(x) for x in activations])
    return activations

def perceptron_learning_rule2(data2, initial_weights, initial_bias):
    weights = initial_weights
    bias = initial_bias
    condition = True
    decision_boundaries1=[]
    decision_boundaries2=[]
    while condition:
        all_errors_zero = np.zeros_like(initial_bias)  # Reset errors for each iteration

        for target_vector, input_vector in data2.items():
            for x in input_vector:
                output = predict(x, weights, bias)
                error = target_vector - output

                if np.any(error != 0):
                    weights += np.outer(error, x)  # Update weights
                    bias += error  # Update bias

                all_errors_zero = np.append(all_errors_zero, np.array([error]))
        #print(all_errors_zero)
        decision_boundaries1.append((-weights[0,0] / weights[0,1], -bias[0,0] / weights[0,1]))
        decision_boundaries2.append((-weights[1,0] / weights[1,1], -bias[0,1] / weights[1,1]))
        if np.all(all_errors_zero == 0):
            condition = False

    return weights, bias, decision_boundaries1, decision_boundaries2


final_weights, final_bias, decision_boundaries1, decision_boundaries2 = perceptron_learning_rule2(data2,initial_weights, initial_bias)

print("Final Weights:", np.round(final_weights,2))
print("Final Bias:", np.round(final_bias,2))


# Show the plot
plt.legend()
plt.title('Prototypes with Four Classes', fontdict=font1)
plt.xlabel('Weight', fontdict=font2)
plt.ylabel('Ear Length', fontdict=font2)
plt.xlim(-2, 7)
plt.ylim(-2, 7)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid("True")
plt.show()






# Plot the prototypes
plt.figure(figsize=(8, 8))
# Plot the decision boundaries
x = np.linspace(-2, 7, 100)
y = np.linspace(-2, 7, 100)
X, Y = np.meshgrid(x, y)



for cls, data in all_prototypes.items():
    if cls == 'I':
        plt.scatter(data[:, 0], data[:, 1], color='red', marker='s', s=200, label='Class I')
    elif cls == 'II':
        plt.scatter(data[:, 0], data[:, 1], color='blue', marker='D', s=200, label='Class II')
    elif cls == 'III':
        plt.scatter(data[:, 0], data[:, 1], color='green', marker='*', s=200, label='Class III')
    elif cls == 'IV':
        plt.scatter(data[:, 0], data[:, 1], color='cyan', marker='o', s=200, label='Class IV')


for (slope, intercept) in decision_boundaries1:
    x_vals = np.linspace(-2, 7, 100)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='magenta', linewidth=4)

for (slope, intercept) in decision_boundaries2:
    x_vals = np.linspace(-2, 7, 100)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='magenta', linewidth=4)
plt.legend()
plt.title('Classification with Decision Boundaries', fontdict=font3)
plt.xlabel('Weight', fontdict=font2)
plt.ylabel('Ear Length', fontdict=font2)
plt.xlim(-2, 7)
plt.ylim(-2, 7)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid("True")
plt.show()



print("-------------------Ques 7------------------------------")
radius = 1.97 # Radius of the disk
num_points = 800  # Number of points to generate for each class
noise_level = 0.5  # Level of noise to add to the points

# Generate noisy data points
noisy_data_points = generate_noisy_data(data2, radius, num_points, noise_level)




final_weights, final_biasd, d1, d2= perceptron_learning_rule2(noisy_data_points, initial_weights, initial_bias)
print("Final Weights:", np.round(final_weights,2))
print("Final Bias:", np.round(final_bias,2))



plt.figure(figsize=(8, 8))
# Plot the decision boundaries
x = np.linspace(-2, 6, 100)
# y = np.linspace(-2, 7, 100)
X, Y = np.meshgrid(x, y)

Z1 = final_weights[0, 0] * X + final_weights[0, 1] * Y + final_bias[0,0]
Z2 = final_weights[1, 0] * X + final_weights[1, 1] * Y + final_bias[0,1]

plt.contour(X, Y, Z1, levels=[0], colors='magenta', linewidths=4)
plt.contour(X, Y, Z2, levels=[0], colors='magenta', linewidths=4)

for cls, data in noisy_data_points.items():
    if cls == (0,0):
        plt.scatter(data[:, 0], data[:, 1], color='red', marker='s', s=200, label='Class I')
    elif cls == (0,1):
        plt.scatter(data[:, 0], data[:, 1], color='blue', marker='D', s=200, label='Class II')
    elif cls == (1,0):
        plt.scatter(data[:, 0], data[:, 1], color='green', marker='*', s=200, label='Class III')
    elif cls == (1,1):
        plt.scatter(data[:, 0], data[:, 1], color='cyan', marker='o', s=200, label='Class IV')


plt.legend()
plt.title('Classification with Optimum Decision Boundary', fontdict=font3)
plt.xlabel('Weight', fontdict=font2)
plt.ylabel('Ear Length', fontdict=font2)
plt.xlim(-2, 7)
plt.ylim(-2, 7)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid("True")
plt.show()


