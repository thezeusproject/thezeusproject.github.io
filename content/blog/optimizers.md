---
title: Optimizers
subtitle: A deep dive into the evolution of optimization algorithms in deep learning.
author: neuralnets
---

In deep learning, an **optimizer** is a crucial element that fine-tunes a neural network's parameters during training. Its primary role is to minimize the model's error or loss function, enhancing performance.

Optimizers facilitate the learning process of a neural network by iteratively updating the weights and biases received from the earlier data. We will discovering optimizers by taking a small road trip down the history of optimizers and slowly inventing newer and more advanced optimizers!

## Premise

What we are trying to do here is nothing but to minimize the loss function of any model. Our objective is to find a set of values for which the value of the loss function is to be minimum

$$
objective = min(loss\space function)
$$

We will be employing several methods to do so, while trying to make our optimizers better and better.

## Approach 1 - Random Search

### Logic

First and the most obvious solution is generate random parameters and find loss with that set of values, if the loss is lesser than our minimum loss, we will keep updating it! After few thousand iterations, we will find our optimum set of parameters

### Code

```python
min_loss=float("inf")
for n in range(10000):
	W=np.random.randn(10,2073)*0.0001 #random parameters
	loss=L(X_train,Y_train,W) #calculate the loss
	if loss<min_loss:
		min_loss=loss
		min_W=W
	print(f"Attempt {n} | Loss - {loss} | MinLoss - {min_loss}")
```

If we actually run this, it will give us an accuracy of around 16.3% (which is not actually that bad, considering it's just random numbers.

Our first approach is not that good in terms of actually finding the optimum parameters, let's try a more intuitive approach!

## Approach 2 - Numeric Gradient

### Logic

What if we try to follow the slope of the curve of the loss function? In a 1-D case, the formula of the derivative is nothing but :

$$
\frac{df(x)}{dx}=\lim_{h\rightarrow0}{\frac{f(x+h)-f(x)}{h}}
$$

The slope of any curve is nothing but it's dot product of its direction with the gradient. Let's take an example table of weights and find their slopes.

| **W** | **W+h** | dW |
| --- | --- | --- |
| 0.34 | 0.34+0.0001 | ? |
| -1.11 | . | ? |
| 0.78 | . | ? |
| 0.12 | . | ? |
| 0.55 | . | ? |
| …….. | ……… | …… |
| Loss : 1.25347 | Loss : 1.25322 |  |

$$
dW_1=\frac{1.25322-1.25347}{0.0001}=-2.5
$$

| **W** | **W+h** | dW |
| --- | --- | --- |
| 0.34 | 0.34+0.0001 | -2.5 |
| -1.11 | . | ? |
| 0.78 | . | ? |
| 0.12 | . | ? |
| 0.55 | . | ? |
| …….. | ……… | …… |
| Loss : 1.25347 | Loss : 1.25353 |  |

Let's find the rest of the values like this

| **W** | **W+h** | dW |
| --- | --- | --- |
| 0.34 | 0.34 | -2.5 |
| -1.11 | -1.11 | 0.6 |
| 0.78 | 0.78+0.0001 | 0 |
| 0.12 | . | ? |
| 0.55 | . | ? |
| …….. | ……… | …… |
| Loss : 1.25347 | Loss : 1.25347 |  |

### What's the problem with this method?

We can go on and on with this method, but it's evident - It's too slow, and you can see it's very tedious to calculate even in 1 dimension, imagine doing this over multiple dimensions. It also tends to approximate the descent by a huge amount! So let's try to think more intuitively for this.

## Approach 3 - Analytic Gradient

### Logic

We can realize that loss is nothing but a function of W.

$$
L=\frac{1}{N}\sum_{i=1}^N L_i + \sum_k W^2_k\\
where, L_i=\sum_{j\neq y_i} max(0,s_j-{s_y}_i+1)\space and\space s=Wx
$$

We need to find where the gradient of the loss function is minimum w.r.t to its weights to find out our optimal parameters. Our goal is to minimize $\nabla_W\space L$ .

| **W** | dW |
| --- | --- |
| 0.34 | -2.5 |
| -1.11 | 0.6 |
| 0.78 | 0 |
| 0.12 | 0.2 |
| 0.55 | 0.7 |
| …….. | …… |
| Loss : 1.25347 |  |

### Why is it better?

This method is quite direct and you can find the exact gradient value, as we are avoiding any find of approximation in our calculation. Hence, it's very fast.

> Tip : After doing analytic gradient, be sure to check the results with your results from numeric gradient, just to cross check values. This method is called *Gradient Check.*
> 

## Approach 4 - Gradient Descent

### Logic

Let's say you are standing at A, and we need to get to the minimum position of the curve. We kinda tried to tackle this problem earlier using **Approach 3** and  **Approach 4.**

![image.png](/assets/images/b2i1.png)

But now we try something new, Gradient Descent. We will find the gradient w.r.t to the positions at A and B, and intuitively we can see that B can be reached by :

$$
B=A-\alpha \nabla A
$$

This procedure of repeatedly evaluating the gradient and then updating the parameters essentially is nothing but Gradient Descent!

How much we step and how far we want to go, is decided by $\alpha$, which is coined as learning rate. Larger the value, more step-size we take, indication a further down B.

### Code

We first get the predictions on the whole training data, then we calculate the loss using some loss function. Finally, we update the weights in the direction of the gradients to minimize the loss. We do this repeatedly for some predefined number of epochs.

```python
for epoch in range(no_of_epochs):
	prediction=model(training_data)
	loss=L(prediction,truth_values)
	W_grad=evaluate_gradient(loss)
	weights-=learning_rate*W_grad
```

### What's the problem with this method?

What if we have millions of data? We can see that the loss function will be counted over the whole dataset, just for one parameter update. This will be computationally difficult and to address this challenge we come up with a new method.

## Approach 5 - Stochastic Gradient Descent

### Logic

In Stochastic Gradient Descent(SGD), we divide our training data into sets of batches.
SGD has two main components :

1. We have to divide the training data into batches
2. For each batch of data,
i) Find out the predictions on the data and find out the loss
ii) Calculate the gradients, based on the loss
iii) Take a step in the opposite direction of gradients to minimize loss

![1_laN3aseisIU3T9QTIlob4Q.gif](https://prod-files-secure.s3.us-west-2.amazonaws.com/cc95b4dc-10fa-4b10-a476-92cd53dd254d/091c8752-fea5-43b9-9c98-53ef1bbf6d5c/1_laN3aseisIU3T9QTIlob4Q.gif)

### Code (Simple)

```python
for epoch in range(no_of_epochs):
	for input_data,labels in training_dataloader: #dataloader divides everything into batches
		prediction=model(training_data)
		loss=L(prediction,truth_values)
		W_grad=evaluate_gradient(loss)
		weights-=learning_rate*W_grad
```

### Code (Pytorch)

In Pytorch, you can simply use the `torch.optim.Optimizer` class for all your optimizer needs. It has the necessary step and zero_grad functions needed for your code

```python
#Pytorch Implementation
model = create_model()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for input_data, labels in train_dataloader:
    preds = model(input_data)
    loss  = L(preds, labels) #finding out loss
    loss.backward() #gradient 
    optimizer.step() #updates the weights by taking step
    optimizer.zero_grad() #resets the grad values to 0 for each epoch
```

### Problems with SGD

**Problem 1**

What if loss changes quickly in one direction and slowly in another? What does gradient descent do?

Ans : The loss function will move slowly along the shallow direction (where the gradient value is small), and it will start jittering. You will be able to notice a zig zag behaviour, as the gradients will be large in that direction and we waste a lot of training time hence.

![image.png](/assets/images/b2i2.gif)

**Problem 2**

What if the loss function has a local minima or saddle point?

Ans : Since that point has zero gradient, the gradient descent will stop and the optimizer gets stuck, in the case of local minima.

![image.png](/assets/images/b2i3.png)

In the case of a saddle point too, we can see it has 0 gradient too, so it will definitely get stuck there.

![image.png](/assets/images/b2i4.png)

**Problem 3**

The gradients are calculated from small batches so it has been noticed that they will have a considerable amount of noise in them as well

![image.png](/assets/images/b2i5.png)

## Approach 6 - SGD with Momentum

### Logic

What is momentum and why are we even using it? 

> Gradient descent is a man walking down a hill. He follows the steepest path downwards; his progress is slow, but steady. Momentum is a heavy ball rolling down the same hill. The added inertia acts both as a smoother and an accelerator, dampening oscillations and causing us to barrel through narrow valleys, small humps and local minima.
> 

To us, momentum is something that speeds up our descent, as sometimes even SGD can take a lot of time to give us satisfactory results.

The equations for SGD with momentum are :

$$
v_{t+1}=\rho v_t+\nabla f(x_t)\\
x_{t+1}=x_t-\alpha v_{t+1}
$$

We have an velocity component($v$) now into the equation. Instead of having just $\nabla x_t$ as the step term, now we have a huge term of $\rho v_t+\nabla f(x_t)$, where $\rho$ is nothing but the momentum factor. This $\rho v_t$  term gives us extra momentum so that our SGD goes faster!

### Code

```python
v_x=0
for epoch in range(no_of_epochs):
	for input_data,labels in training_dataloader: #dataloader divides everything into batches
		prediction=model(training_data)
		loss=L(prediction,truth_values)
		W_grad=evaluate_gradient(loss)
		v_x=rho*v_x+W_grad
		weights-=learning_rate*v_x
```

## Approach 7 - Nesterov Momentum

In 1983, Yuri Nesterov suggested Nesterov Accelerate Gradients method, which is similar to momentum method, but with an alteration:

![image.png](/assets/images/b2i6.png)
We want to look ahead to the point where updating using velocity takes us, we will compute the gradient there and add it with velocity to get an update direction!

The new equations become :

$$
v_{t+1}=\rho v_t-\alpha \nabla f(x_t+\rho v_t)\\
x_{t+1}=x_t+v_{t+1}
$$

Let's simplify the equations further. Let $\tilde{x}_t = x_t + \rho v_t$,
$$
\begin{aligned}
v_{t+1} &= \rho v_t - \alpha \nabla f(\tilde{x}_t) \\
\tilde{x}_{t+1} &= \tilde{x}_t - \rho v_t + (1 + \rho)v_{t+1} \\
                &= \tilde{x}_t + v_{t+1} + \rho(v_{t+1} - v_t)
\end{aligned}
$$

It is like that Nesterov momentum looks ahead into future and uses momentum to predict it.

## Approach 8 - AdaGrad

### Logic

This method came up in 2011, which later will give birth to many other approaches that we will discover soon!

During the calculation of gradients, Adagrad stores the history of the outer products of gradients :

$$
G_t = \sum_{\tau=1}^t g_\tau^2
$$
where $g$ is the gradient. Note that the square is element-wise.

Finally this result is used to maintain the essence of some momentum in each coordinate. The final update rule comes out to be :

$$
x_{t+1} = x_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} g_t
$$

### Code

```python
for epoch in range(no_of_epochs):
	for input_data,labels in training_dataloader: #dataloader divides everything into batches
		prediction=model(training_data)
		loss=L(prediction,truth_values)
		W_grad=evaluate_gradient(loss)
		grad_sq+=W_grad**2
		weights-=learning_rate*W_grad/(np.sqrt(grad_sq)+1e-7)
```

## Approach 9 - RMSProp

### Logic

This optimizer brings to us an idea that why should all parameters have the step-size when clearly some parameters should move faster?

> **Fun Fact** : This technique was introduced by Geoffrey Hinton in his class. He won the Turing Award (often called the "Nobel Prize of Computing") in 2018!
> 

During the calculation of gradients RMSprop maintains moving averages of squares of gradients :

$$
v_t = \beta v_{t-1} + (1-\beta)g_t^2
$$
where $g$ is the gradient and the square is element-wise.

We introduce a new constant $\beta$ here. What is it? It is called the decay rate momentum. To calculate the new weighted average, it sets the weight between the average of previous values and the current value.

The final update rule becomes :

$$
x_{t+1}=x_t - \frac{\alpha}{\sqrt{v_t + \epsilon}} g_t
$$

Thus we can see, by tinkering AdaGrad, we got RMSProp, hence it's also famously called Leaky Adagrad.

### Code

```python
for epoch in range(no_of_epochs):
	for input_data,labels in training_dataloader: #dataloader divides everything into batches
		prediction=model(training_data)
		loss=L(prediction,truth_values)
		W_grad=evaluate_gradient(loss)
		grad_sq = decay_rate * grad_sq + (1 - decay_rate) * (W_grad**2)
		weights -= learning_rate * W_grad / (np.sqrt(grad_sq) + 1e-7)
```

## Approach 10 - Adam

### Logic

Adam is a 2014 improvement over AdaGrad and RMSprop. It is an adaptive learning rate algorithm. This means it dynamically adjusts the learning rate for each individual parameter within a model, rather than using a single global learning rate. 

Adam combines the two approaches through hyperparameters $\beta_1\space and \space \beta_2$. We can see how the $x_t$ update looks similar to the RMSProp update, amplified with Momentum, so we get the best of both worlds.

we need to be able to keep a moving average of the first and second moments of the gradients. Finally, based on the bias correction term $1-\beta_1^t$ for the first moment estimate and $1-\beta_2^t$ for the second moment estimate, we compute the biased corrected version and first and second raw moment estimates.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\end{aligned}
$$

The bias-corrected estimates are:

$$
\begin{aligned}
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t}
\end{aligned}
$$

Combining these two, we can write the update rule for Adam to be :

$$
x_{t+1}= x_t- \frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

This gives the essence of the best of both worlds. We are updating according to two moving averages: $m_t$ (the first moment estimate) and $v_t$ (the second moment estimate). These are estimates of the mean and uncentered variance of the gradients, respectively. They are then bias-corrected to $\hat{m}_t$ and $\hat{v}_t$ before being used in the update rule.

### Code

```python
first_moment = 0
second_moment = 0
for t in range(1, num_iterations):
    dx = compute_gradient(x)
    first_moment = beta1 * first_moment + (1 - beta1) * dx
    second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
    first_moment_corrected = first_moment / (1 - beta1 ** t)
    second_moment_corrected = second_moment / (1 - beta2 ** t)
    x -= learning_rate * first_moment_corrected / (np.sqrt(second_moment_corrected) + 1e-7)
```

## Conclusion

In practice, Adam is a default choice, as it can work with okayish learning rates on its own. SGD+Momentum might beat Adam but might require extra fine tuning.
**Use Adam as default!**


