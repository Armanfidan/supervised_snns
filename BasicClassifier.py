import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch

# We will be creating a spiking neural network which classifies input
# patterns as one of two types - hence the two outputs
num_inputs = 100
num_hidden = 4
num_outputs = 2

# As discussed in the "image processing using SNNs" document, we will
# modelling an SNN as an RNN. Hence, we need to approximate each spike
# in discrete-time. This is why we use time steps. Since we use 200 steps
# each 1 ms long, the simulation will run for 200 ms.
time_step = 1e-3
num_steps = 200

# We will run the code on batches of data. This is usually done in neural
# networks being trained in a supervised manner.
batch_size = 256

dtype = torch.float
device = torch.device("cpu")

# We will now generate a spiking dataset to feed into the network.
frequency = 5  # Hz
probability = frequency * time_step  # 5 * 0.001 = 0.005
# We create a Tensor of rank three, so a three-dimensional array.
# The size of this array is (batch_size x num_steps x num_inputs).
# The torch.rand() function fills this array with random numbers between 0 and 1.
mask = torch.rand((batch_size, num_steps, num_inputs), device=device, dtype=dtype)
# We create a tensor of the same dimensions, but this one is initialised with zeros.
x_data = torch.zeros((batch_size, num_steps, num_inputs), device=device, dtype=dtype)
# Writing a condition in the indices of a Torch Tensor returns the indices
# of the Tensor, the value in which satisfies that condition. In this case,
# we are checking every index and setting the ones whose value is greater than
# our set "probability" (0.005 in our case) to 1.
#
# This means that we are iterating through our Tensor of zeros and checking whether
# values in the same index in the Tensor "mask" are greater than our probability.
# If so, we are replacing that zero with a one. This way, we have created a three-dimensional
# array of ones and zeros, with the ones representing the spike times.
x_data[mask < probability] = 1.0

# Now we will plot the first layer of this data.
# data_id = 0
# plt.imshow(x_data[data_id].cpu().t(), cmap=plt.cm.gray_r, aspect="auto")
# plt.xlabel("Time (ms)")
# plt.ylabel("Neuron Index")
# # This just removes two of the lines around the plot.
# sns.despine()
# plt.show()

# Here we create a new rank one tensor, which will carry a label (0 or 1)
# for each of the two-dimensional input patterns.
# The way we do that is that we multiply one with a boolean, which is 0 if
# the random number generated is smaller than 0.5 and 1 otherwise.3
y_data = torch.tensor(1 * (np.random.rand(batch_size) < 0.5), device=device)

# Now we start constructing the spiking neural network model that is
# described by the two incremental discrete-time equations given in the
# aforementioned document.
tau_mem = 10e-3
tau_syn = 5e-3

alpha = float(np.exp(-time_step / tau_syn))
beta = float(np.exp(-time_step / tau_mem))

# It is now time for the weight matrices between the first and hidden (w1)
# and hidden and output (w2) layers

# Weights are initialised according to a normal distribution.
# This weight scale is used to scale the variance of the distribution -
# The more inputs connexions we have, the smaller the standard deviation.
# This is because... Look this up.
weight_scale = 7 * (1.0 - beta)

# We first create an empty tensor for the weights, specifying the dimensions.
# The values will ne initialised later.
w1 = torch.empty((num_inputs, num_hidden), device=device, dtype=dtype, requires_grad=True)
# Here we initialise the weights.
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / np.sqrt(num_inputs))

w2 = torch.empty((num_hidden, num_outputs), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / np.sqrt(num_hidden))

print("Initialisation complete")


# # We now multiply the input spike trains by the weights corresponding to the first layer.
# # In this function, we are specifying an equation in the following format:
# # "indices of first input, indices of second input -> indices of output"
# # So this performs a vector multiplication between a three dimensional vector and
# # a two dimensional vector. This corresponds to multiplying each "layer" of
# # the inputs (corresponding to each individual input sample) with the same weights.
# h1 = torch.einsum("abc, cd -> abd", (x_data, w1))
# # print(x_data)
#
# # print(x_data.size())
# # print(w1.size())
# # print(h1.size())
# # print(x_data)
# # print(w1)
# # As expected, h1 is just x_data, with each spike scaled numerically according to
# # the weights w1. That is, each spike has a floating point amplitude.
# # print(h1)
#
# # No. h1 has size (batch_size, num_steps, num_hidden)
# # Which means that for each hidden neuron, it contains (batch_size)
# # spike patterns of length (time_steps).
#
#
# # # This section explains how h1 is calculated -------------------------------------------------------------
# # # Wait let's test something. This tensor has dimensions [3, 2, 4]
# # a = torch.tensor([[[1, 2, 3, 4],
# #                    [5, 6, 7, 8]],
# #
# #                   [[2, 3, 4, 5],
# #                    [6, 7, 8, 9]],
# #
# #                   [[3, 4, 5, 6],
# #                    [7, 8, 9, 0]]])
# # # This one has dimensions [4, 2]
# # b = torch.tensor([[1, 2],
# #                   [3, 4],
# #                   [5, 6],
# #                   [7, 8]])
# # # If we use einsum on them the same way h1 was created, we get:
# # c = torch.einsum("abc, cd -> abd", (a, b))
# # print(c)
# # # The resulting tensor is:
# # # tensor([[[ 50,  60],
# # #          [114, 140]],
# # #
# # #         [[ 66,  80],
# # #          [130, 160]],
# # #
# # #         [[ 82, 100],
# # #          [ 76, 100]]])
# # # The dimensions of this are [3, 2, 2]. This is an outer product (Tensor product) (?)
# # # In every cell, we have:
# # #   * Let's check the first item: Let's take [1, 2, 3, 4]*[1, 3, 5, 7]T
# # #   * That's 1*1 + 2*3 + 3*5 + 4*7 = 50
# # #   * If we repeat this for the other items in the first layer (first row times second
# # #     column, second row times first column, second row times second column), we will
# # #     get the remaining three entries, 60, 114 and 140. And thus, this product is
# # #     demystified.
# # # Now, let's think about how this could be used in multiplying spike trains by their
# # # corresponding weights: Each layer contains (time_step) vectors of (num_inputs)
# # # boolean values. What we do when using einsum(abc, cd -> abd) is that we multiply each
# # # entry in each vector (each of which corresponds to one neuron) by its corresponding
# # # weight (remember, the weights vector is arranged like this:
# # #                            (Hidden neurons)
# # #                         [[.56, .25, .75, .34]
# # #         (Input neurons)  [.24, .64, .19, .53]
# # #                          ...,
# # #                          [.63, .29, .49, .10]]
# # #         which means that, for example, the first column represents W_i0, so each input
# # #         neuron connected to the first hidden neuron.)
# # # Thus, each entry in the h1 vector is the weighted spike train that activates each hidden
# # # neuron at that time-step. This is also called the "activation" of each hidden neuron.
# # -----------------------------------------------------------------------------------------------------
#
#
# # WAIT NO NO NO now I get it. An array is passed as an argument to this function
# # and the array is supposed to contain the membrane potentials. This function
# # just constitutes the spiking nonlinearity: It just sets the potential to 1
# # once it passes the threshold (resting potential + 1).
# def spike_fn(x):
#     out = torch.zeros_like(x)
#     # What the condition x > 0 does is that it checks every item in x to see whether
#     # it is greater than zero. It then returns an array of booleans that are True
#     # only at indices where that value is greater than 0. out[x > 0] means that we
#     # select indices where this is True. We then set these to 1.
#     out[x > 0] = 1.0
#     return out
#
#
# # We initialise the synaptic currents and membrane potentials to zero.
# # Syn: I[n+1] = alpha*I[n] + sum(W*S[n]) + sum(V*S[n])
# # Where S[n] is approximated using the unit step function u(n)
# syn = torch.zeros((batch_size, num_hidden), device=device, dtype=dtype)
# # Mem: U[n+1] = beta*U[n] + I[n] - S[n]
# # The last term in this equation is S[n](Urest-threshold), but
# # Urest=0 and threshold=1 so it becomes -S[n]
# mem = torch.zeros((batch_size, num_hidden), device=device, dtype=dtype)
#
# # Defining two lists in which the membrane is recorded
# mem_rec = [mem]
# spk_rec = [mem]
#
# # We loop over the time steps, so what is in the loop is what should happen at each time step
# for t in range(num_steps):
#     # We subtract 1 from the membrane potentials,
#     # meaning that we initialise this array to all -1's.
#     # We do this to set the resting potential to -1, since the function spike_fn makes the neurons
#     # spike if their membrane potential is above 0, and since the threshold is supposed to be
#     # (Urest + 1), we set Urest to 0. Not doing this and using mem, and in the spike_fn function
#     # typing out[x > 1] = 1.0 would have been equivalent.
#     mthr = mem - 1.0
#     # print("mthr: ", mthr)
#     # u(mthr) just resets all values of mthr to zero, because the first
#     # index in this vector is 0. Since the unit step function adds
#
#     # Now this array contains a 1 for each neuron that spikes and a 0 for each one that does not.
#     out = spike_fn(mthr)
#     print("Out: ", out)
#
#     # This here is redundant. Out can be removed instead of reset. ----------------------------------------
#     # However, I will keep it because it was included in the original tutorial.
#     reset = torch.zeros_like(mem)
#     # This is an array that is true whenever there is a spike.
#     c = (mthr > 0)
#     print(c)
#     # Here, we use the above spike vector, and set reset to 1 whenever there is a spike.
#     reset[c] = torch.ones_like(mem)[c]
#     # print("Reset:\n", reset)
#     # -----------------------------------------------------------------------------------------------------
#
#     # These implement the equations described above
#     # syn_new = I[n+1], syn = I[n]
#     # h1 consists of the input spike trains multiplied by the weights of
#     # the first layer, so it makes the second term in the above equation.
#     # The last term is not applicable since our network does not have
#     # any recurrent connexions.
#
#     # In a three-dimensional tensor, the square brackets represent
#     # [row, column, index (in column)]
#     # So h1[:, t] would represent every t'th column in h1.
#     # Considering that each layer in h1 represents one spike pattern,
#     # Each column would represent the spike input to one neuron in
#     # a spike pattern. Thus, h1[:, t] just selects the input to each neuron
#     # (taking into account every input pattern in the batch) in order.
#
#     # Each element h1[:, t] is a two-dimensional array in itself,
#     # consisting of (batch_size) different arrays of 0s or spikes
#     # each of length num_inputs.
#     # So basically we are taking the activations of all 4 hidden neurons
#     # when subjected to 256 different input patterns, at instant t.
#     # This is just sum(W*S[n]) but for every hidden neuron.
#     syn_new = alpha * syn + h1[:, t]
#     # mem_new = U[n+1], mem = U[n]
#     # We subtract 1 from the membrane potential anytime there is a spike (-reset)
#     # This is the mechanism to reset the membrane potential to Urest.
#     mem_new = beta * mem + syn - reset
#
#     # We update our variables
#     mem = mem_new
#     syn = syn_new
#
#     #We
#     mem_rec.append(mem)
#     spk_rec.append(out)
#
# # The stack function concatenates dimensions of the input tensor
# # into the dimension number specified. So this downscales the tensors to 1 dimension.
# mem_rec = torch.stack(mem_rec, dim=1)
# spk_rec = torch.stack(spk_rec, dim=1)


def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5):
    # We use this to customise the figure layouts.
    gs = GridSpec(*dim)
    # If a spike array (zeros with ones where spikes are present) is present,
    # plot the normal membrane potential with a shoot up to 5*1=5 at spikes.
    if spk is not None:
        dat = (mem + spike_height * spk).detach().cpu().numpy()
    else:
        # If no spike array is present, just plot the membrane potentials
        dat = mem.detach().cpu().numpy()
    # This one just multiplies dimensions (total number of plots to be drawn)
    for i in range(np.prod(dim)):
        if i == 0:
            # First plot, so set a0 to this one.
            a0 = ax = plt.subplot(gs[i])
        else:
            # Take the first plot (a0) as a reference.
            ax = plt.subplot(gs[i], sharey=a0)
        # the ith element of dat is the ith input pattern. Here we only
        # plot the first 15 as an example.
        ax.plot(dat[i])
        ax.axis("on")


#
# fig = plt.figure(dpi=100)
# plot_voltage_traces(mem_rec, spk_rec)
# plt.show()


def spike_fn(x):
    out = torch.zeros_like(x)
    out[x > 0] = 1.0
    return out


# We will now place the above code plus the output layer in a function
# in order for the SNN to be run more than once.
def run_snn(inputs):
    h1 = torch.einsum("abc, cd -> abd", (inputs, w1))
    syn = torch.zeros((batch_size, num_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, num_hidden), device=device, dtype=dtype)

    mem_rec = [mem]
    spk_rec = [mem]

    # Compute hidden layer activity
    for t in range(num_steps):
        mthr = mem - 1.0
        out = spike_fn(mthr)

        new_syn = alpha * syn + h1[:, t]
        new_mem = beta * mem + syn - out

        mem = new_mem
        syn = new_syn

        mem_rec.append(mem)
        spk_rec.append(out)

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    # Compute readout layer activity
    h2 = torch.einsum("abc, cd -> abd", (spk_rec, w2))
    flt = torch.zeros((batch_size, num_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size, num_outputs), device=device, dtype=dtype)

    out_rec = [out]

    # Remember, out here is mem in the previous layer.
    # The only thing I do not understand is why we do not reset the membrane
    # potentials in the readout layer.
    # Ok, here we go. We never let the output neurons spike. This is
    # to be able to define a smooth objective on their membrane voltages.
    # Since we never let them spike, we do not have to reset them.
    for t in range(num_steps):
        new_flt = alpha * flt + h2[:, t]
        new_out = beta * out + flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec, dim=1)

    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs


# # When this is run, we get two traces on the same graph. This is because
# # we have two output neurons. Why they spike less frequently and appear
# # more stable is still a mystery to me though.
# output_spikes, other_recordings = run_snn(x_data)
# fig = plt.figure(dpi=100)
# plot_voltage_traces(output_spikes)
# # plt.show()


def print_classification_accuracy():
    output, _ = run_snn(x_data)
    # print(output)
    # Max over time - returns 256 lists that each contain the maximum
    # potential for both output neurons over 200 time steps.
    # The first output "maximum" contains the values of the maximum potentials.
    # The second output contains the argmax, or the index of the maximum potentials.
    maximum, _ = torch.max(output, 1)
    # print("Maximum: ", maximum)
    # Argmax over maximum outputs: For each input pattern in the batch, we have two
    # output neurons. "maximum" contains the maximum membrane potential for both
    # output neurons. We compare them and see which one is greater, then retrieve its
    # index; hence "classifying" the input pattern as one of two types. This is based on
    # which output neuron responded to it the most (had the highest potential when shown
    # the pattern). The comparison thus produces an array of length 256 of 1s and 0s.
    _, argmax = torch.max(maximum, 1)
    # print(argmax)
    # We compare this array to the random labels we initially assigned to see whether they match
    # up. Unsurprisingly, the accuracy is about 50%, because there are two options and this is
    # how chance works.
    # print(y_data == argmax)
    accuracy = np.mean((y_data == argmax).detach().cpu().numpy())
    print("Accuracy %.3f" % accuracy)


# # The accuracy, as predicted above, is around 0.5.
# print_classification_accuracy()
#
# # We try to perform gradient descent using the true gradient to see that it does not work. --------------------------
# # The parameters we want to optimise
# params = [w1, w2]
# # This function holds the current state and update the parameters based
# # on the calculated gradients.
# optimizer = torch.optim.Adam(params, lr=2e-3, betas=(0.9, 0.999))
#
# # The log softmax is a function that converts a vector of numbers (any
# # value) into a vector whose sum is 1. This way, the values can
# # be interpreted as probabilities. This is the log of the softmax fn.
# # EDIT: THIS IS AN ACTIVATION FUNCTION, JUST LIKE THE SIGMOID!! Easy lol
# log_softmax_fn = torch.nn.LogSoftmax(dim=1)
# # This is the negative log-likelihood loss function.
# loss_fn = torch.nn.NLLLoss()
#
# # The optimization loop
# loss_hist = []
# for e in range(1000):
#     # run the network and get output
#     output, _ = run_snn(x_data)
#     # Here we compute the maximum membrane potential of each output
#     # neuron, same as before.
#     maximum, _ = torch.max(output, 1)
#     # Now we input this array of maximum potentials into the log softmax
#     # function to obtain log-probabilities.
#     log_p_y = log_softmax_fn(maximum)
#     # # # # For now, I will skip learning how this loss function works.
#     # But essentially, it calculates the difference between the ideal
#     # (y_data) and real (log_p_y) outputs.
#     loss_val = loss_fn(log_p_y, y_data.long())
#
#     # Here we update the weights. Below are standard, inbuilt functions
#     # of PyTorch.
#     # The zero_grad() function clears the gradients of all Tensors.
#     optimizer.zero_grad()
#     # This is the backpropagation function. Previously, when defining
#     # variables, we had set "requires_grad=True" on the ones that we
#     # wish to train. In our case, these were W1 and W2.
#     # This function thus calculates the current d(loss_val)/dw1 and
#     # d(loss_val)/dw2 by backpropagating the errors. When doing this,
#     # it uses the equations that are talked about in Neftci et al.'s
#     # paper (about backpropagation).
#     # This function then adds the computed gradients to the parameters'
#     # ".grad" attribute. Since this attribute is cumulative, we have to
#     # clear the gradients from the previous runs in order to calculate
#     # the correct gradients.
#     loss_val.backward()
#     # The "step" function updates the values of the parameters using this
#     # gradient. This is done according to "W <- W - n*W.grad"
#     # where n is a scalar factor.
#     optimizer.step()
#
#     # Now, the loss value is stored to create the histogram after all the
#     # iterations.
#     loss_hist.append(loss_val.item())
#
# loss_hist_true_grad = loss_hist  # store for later use
#
# plt.plot(loss_hist)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# sns.despine()
# plt.show()
#
# print_classification_accuracy()
# ----------------------------------------------------------------------------------------------------


# When this is plotted, it is evident that a small improvement in loss
# has been made. (Although this is insignificant).
# However, the accuracy is still the same. This is because we encode
# information using a temporal code, which implies that the derivative
# of the output is zero except for where the spikes are, where it is
# infinite. Therefore, the gradient descent algorithm does not modify
# weights at all. In order to fix this, surrogate gradients are
# introduced.

# Now, let's be clear. I still do not have ANY idea what surrogate
# gradients are, but I am hoping to find out!

# Ok. so the tutorial suggests that:
# "The idea behind a surrogate gradient is dead simple. Instead of
# changing the non-linearity itself, we only change the gradient.
# Thus we use a different "surrogate" gradient to optimize parameters
# that would otherwise have a vanishing gradient."
# Let's see that in action.

# The tutorial follows up with:
# "Specifically, we use the partial derivative of a function which
# to some extent approximates the step function  Θ(𝑥) . In what follows,
# chiefly, we will use (up to rescaling) the partial derivative of a
# fast sigmoid function  𝜎(𝑥) . While  Θ  is invariant to
# multiplicative rescaling,  𝜎  isn't. Thus we have to introduce a
# scale parameter."
# Now, what I do not understand is this: We use a function
# whose partial derivative resembles that of a step function. Should this
# not mean that although the gradient is not zero everywhere but at the
# spikes, will it not be so close to zero to the point where it is
# insignificant? How does using this gradient solve our problem?
class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalised negative part of a fast sigmoid as this
    was done in Zenke & Ganguli (2018).
    """

    # This controls the steepness of the surrogate gradient.
    scale = 100.0

    # The foreward method of this class comes into use when we run the network.
    # It is run during every forward pass (as the activations propagate from one
    # layer to the next). In this case, we do the exact same thing that we did
    # the previous time - we define the non-linearity (this class) as a step function.
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context based object that we use to stash
        information which we need to later backpropagate our error signals.
        To achieve this we use the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    # However, the backward method defines what non-linearity to use when computing the
    # gradient to backpropagate the errors. This time, we cannot use the step function,
    # because it has a vanishing gradient (its gradient is zero everywhere except for
    # the reset, at which point it is infinity. Therefore, we must use a fast sigmoid
    # to accomplish this task. This is a "surrogate" or "replacement" gradient, which is
    # useful because it lets the gradients flow, which solves the problem of the errors
    # vanishing as they backpropagate.
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalised negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        # Now we restore the inputs that were saved in the forward pass, because we need
        # to compute their gradient.
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Fast sigmoid - this returns an array whose y-values are proportional to
        # x-values/saved input^2.
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


# Here we overwrite our naive spike function with the "SurrGradSpike" function.
# We use the .apply because this is the way for "new-style" functions that do not
# have an __init__() function.
spike_fn = SurrGradSpike.apply

# We reinitialise the weights.
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / np.sqrt(num_inputs))
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / np.sqrt(num_hidden))
print("Reinitialisation complete")

params = [w1, w2]
optimiser = torch.optim.Adam(params, lr=2e-3, betas=(0.9, 0.999))

log_softmax_fn = torch.nn.LogSoftmax(dim=1)
loss_fn = torch.nn.NLLLoss()

loss_hist = []
for e in range(10):
    output, _ = run_snn(x_data)
    maximum, _ = torch.max(output, 1)
    log_p_y = log_softmax_fn(maximum)
    loss_val = loss_fn(log_p_y, y_data.long())

    optimiser.zero_grad()
    loss_val.backward()
    optimiser.step()
    loss_hist.append(loss_val.item())

plt.figure(figsize=(3.3, 2), dpi=150)
# plt.plot(loss_hist_true_grad, label="True gradient")
plt.plot(loss_hist, label="Surrogate gradient")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
sns.despine()
plt.show()

output, other_recordings = run_snn(x_data)
mem_rec, spk_rec = other_recordings
fig = plt.figure(dpi=100)
plot_voltage_traces(mem_rec, spk_rec)
plt.show()

plot_voltage_traces(output)
plt.show()

print_classification_accuracy()
