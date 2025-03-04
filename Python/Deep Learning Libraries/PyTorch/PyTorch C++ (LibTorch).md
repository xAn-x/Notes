
## **Setup**:
1. **Download LibTorch**:
   - Visit [LibTorch](https://pytorch.org/get-started/locally/) and download the C++ distribution.
2. **CMake Configuration**:
   Add the following to your `CMakeLists.txt`:
   ```cmake
   cmake_minimum_required(VERSION 3.10)
   project(MyProject)

   find_package(Torch REQUIRED)

   add_executable(my_executable main.cpp)
   target_link_libraries(my_executable "${TORCH_LIBRARIES}")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
   ```

---

## **Basic Tensor Operations**:

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    // Create a tensor
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    // Perform operations
    auto result = tensor * 2;
    std::cout << "Result of tensor * 2: " << result << std::endl;

    return 0;
}
```

---
## **Basic Neural Network**:

```cpp
struct Net : torch::nn::Module {
    Net() {
        // Define layers
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
    // Create model
    Net net;
    of 1, 784 features
    torch::Tensor output = net.forward(input);
    std::cout << output << std::endl;

    return 0;
}
```

- **Net class**: A simple feedforward network with 2 layers.
- `torch::relu`: Applies ReLU activation.
- `register_module`: Registers layers to track and manage weights automatically.

---

## **Loss and Optimizer**:

```cpp
int main() {
    Net net;

    // Input and target
    torch::Tensor input = torch::randn({1, 784});
    torch::Tensor target = torch::randn({1, 10});

    // Loss function (Mean Squared Error)
    torch::nn::MSELoss loss_fn;

    // Optimizer (SGD)
    torch::optim::SGD optimizer(net.parameters(), /*learning rate=*/0.01);

    // Forward pass
    torch::Tensor output = net.forward(input);
    torch::Tensor loss = loss_fn(output, target);

    std::cout << "Loss: " << loss.item<float>() << std::endl;

    // Backward pass and optimization step
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    return 0;
}
```

- `torch::nn::MSELoss`: Mean Squared Error loss function.
- `torch::optim::SGD`: Stochastic Gradient Descent optimizer.
- `loss.backward()`: Computes gradients.
- `optimizer.step()`: Updates model parameters.

---

## **Training Loop**:

```cpp
for (size_t epoch = 0; epoch < 10; ++epoch) {
    for (auto& batch : *data_loader) {
        // Get data
        auto data = batch.data, target = batch.target;

        // Forward pass
        auto output = net.forward(data);

        // Compute loss
        auto loss = loss_fn(output, target);

        // Backpropagation and update weights
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        std::cout << "Epoch [" << epoch << "] Loss: " << loss.item<float>() << std::endl;
    }
}
```
- **data_loader**: Iterable to get batches of data.
- For each batch, it computes the forward pass, loss, and updates the model.

---

## **Save and Load Model**:

### 1. **Save Model**:
```cpp
torch::save(net, "model.pt");
```

### 2. **Load Model**:
```cpp
Net net;
torch::load(net, "model.pt");
```

