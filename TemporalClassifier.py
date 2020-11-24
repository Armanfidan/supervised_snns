import os

import numpy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn
import torchvision

# Numbers of neurons
num_inputs = 28 * 28
num_hidden = 100
num_outputs = 10

time_step = 1e-3
num_steps = 100

batch_size = 256

dtype = torch.float

# Check whether a CPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

# Now the dataset is loaded
root_dir = os.path.expanduser(r"~\OneDrive - Imperial College London\UROP - Neuromorphic Circuits with Memristors")
train_dataset = torchvision.datasets.FashionMNIST(root_dir, train=True, transform=None, target_transform=None,
                                                  download=True)
test_dataset = torchvision.datasets.FashionMNIST(root_dir, train=False, transform=None, target_transform=None,
                                                 download=True)

x_train = numpy.array(train_dataset.data, dtype=numpy.float)
# This converts an array of 60000 2D images (28 by 28) into an array of 60000
# 1D images (784 by 1). Additionally, it normalises all values.
x_train = x_train.reshape(x_train.shape[0], -1) / 255

x_test = numpy.array(test_dataset.data, dtype=numpy.float)
x_test = x_test.reshape(x_test.shape[0], -1) / 255

# 1D arrays containing labels for every image. The labels are integers between
# 0 and 9, which each one representing a different article of clothing.
y_train = numpy.array(train_dataset.targets, dtype=numpy.int)
y_test = numpy.array(test_dataset.targets, dtype=numpy.int)

data_id = 1
plt.imshow(x_train[data_id].reshape(28, 28), cmap=plt.cm.gray_r)
plt.axis("on")
# plt.show()


def current_to_firing_time(x, tau=20, threshold=0.2, t_max=1.0, epsilon=1e-7):
    """
    This function computes first firing time latency for a current input x,
    assuming a charge time of a current=based LIF neuron.

    :param x: Input current (Between 0 and 1)
    :param tau: Membrane time constant of the LIF neuron to be charged
    :param threshold: Firing threshold of neuron
    :param t_max: Maximum time that can be returned
    :param epsilon: A generic (small) value, greater than 0.
    :return: Time to first spike for each input current value
    """
    # Since x is an array of currents, this statement will return an array
    # of "True" or "False" at each index
    image_index = x < threshold
    # This function limits the values in an array between two values
    # (between threshold+epsilon and 10^9)
    x = numpy.clip(x, threshold + epsilon, 1e9)
    # This is the function that maps the currents onto times.
    T = tau * numpy.log(x / (x - threshold))
    # Remember that when a condition is in the square brackets, it means that
    # only values at said indices are modified. Thus, in this case every pixel
    # whose brightness is smaller than the threshold is set to fire as late
    # as possible (at t_max).
    T[image_index] = t_max
    # T is now an array consisting of len(x) firing times, each between 0 and
    # t_max. All currents with x < threshold will fire at T = t_max.
    return T


# Basically, sparse_data_generator is used to create the input spike trains to the network from images.
def sparse_data_generator(x, y, batch_size, num_steps, num_units, shuffle=True):
    """
    This generator takes datasets in analogue format and generates spiking
    network input as sparse tensors.

    :param x: The data (sample * event * 2) the last dim holds (time, neuron), tuples
    :param y: The labels
    :param batch_size: Minibatch size
    :param num_steps: Total number of timesteps
    :param num_units: Number of neurons
    :param shuffle: Whether the input images should be shuffled
    :return:
    """

    # Here, "y" will be "y_test" or "y_train", so the labels.
    labels_ = numpy.array(y, dtype=numpy.int)
    # The "//" operator stands for floor division.
    # We have 60000 training images for example, but we need to train the network using
    # minibatches of 256 images. Therefore we need a certain number of batches.
    # I do not know, however, why a floor division is used instead of a ceiling division -
    # Does this mean that if a few images do not fit in any batches we just discard them?
    number_of_batches = len(x) // batch_size
    # numpy.arange(end) is equivalent to the python range function but
    # returns a numpy array instead of an array. It creates an array of
    # equally spaced integers from 0 up to (end).
    # Thus, sample index literally contains the indices of each image in x.
    # Keep in mind that x is two dimensional - len(x) returns only the length
    # of the first dimension (separate images).
    sample_index = numpy.arange(len(x))

    # Here we compute the discrete firing times
    tau_eff = 20e-3 / time_step
    # firing_times is an array of firing times, with each time being an integer between
    # 0 and 199 to indicate at which time step each neuron has fired.
    # This array contains 28*28 firing times for all 60000 input images. Thus, the 1st dimension (layer)
    # is the number of input sample, and the 2nd one is the number of input neuron.
    firing_times = numpy.array(current_to_firing_time(x, tau=tau_eff, t_max=num_steps), dtype=numpy.int)
    print("Firing times shape:", firing_times.shape)
    # From what I understand, num_units is the number of neurons in a layer.
    # Thus, unit_numbers is an array that contains integers from 0 to the last input neuron -
    # basically neuron indices.
    unit_numbers = numpy.arange(num_units)

    # I do not understand how this would work - we shuffle the images but not their labels,
    # would this not lead to random labels being assigned to each image?
    if shuffle:
        numpy.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter < number_of_batches:
        # batch_index is just a subset of sample_index that contains the indices of neurons
        # in the current batch (so this will always have a size of 256 or less).
        batch_index = sample_index[(batch_size * counter):(batch_size * (counter + 1))]

        # Creates an array of three empty arrays.
        # COO STANDS FOR COORDINATE - DON'T FORGET.
        # So apparently, the "coordinate format" is used to create sparse tensors, tensors whose contents are
        # zero except for certain coordinates at which there exist values. our coo array has three sets of
        # coordinates - one for the row, one for the column and one for the layer (which image out of the
        # batch the spike belongs to). In ours the first one will represent the layer (batch), the second the
        # firing times (row) and the third the neurons that have fired (column).
        coo = [[] for i in range(3)]

        # the enumerate function iterates through the items in an array, returning both the
        # items and their indices. Here, batch counter holds the index and image_index is the "batch index"
        # So we are iterating through images in the batch.
        for batch_counter, image_index in enumerate(batch_index):
            # print("Image index:", image_index)
            # We now iterate through each input neuron [c] for each image [image_index] in the firing times
            # array and check to see whether each one has fired before the maximum time possible. This
            # differentiates between the neurons whose activation is above the threshold (they have fired
            # before t_max) and others, who have fired at t_max (num_steps).
            # Thus, c is an array of size 28*28=784 that contains booleans (True for neurons that have fired
            # and False for ones that have not)
            c = firing_times[image_index] < num_steps

            # times is an array that contains the firing times for ONLY the neurons that HAVE fired - in the
            # current image. units contains the indices of said neurons. Basically like max and argmax.
            times, units = firing_times[image_index][c], unit_numbers[c]

            # This array contains the image number, one for each spike. This is because we will use this as
            # a coordinate to fill the sparse spike Tensor.
            batch = [batch_counter for _ in range(len(times))]

            # The extend() method adds the parameter to the end of the list it is called on - basically
            # merges lists.
            # So we fill the first coordinate (layer) with the image index.
            coo[0].extend(batch)
            # We then fill the second coordinate (row) with the firing times.
            coo[1].extend(times)
            # Finally, the last coordinate (column) is filled with the indices of the neurons that have fired.
            coo[2].extend(units)
            # Keep in mind that on every iteration of the for loop, this coordinate list is only extended, not
            # overwritten. Thus, it keeps expanding until it has EVERY image on which neurons have spiked,
            # EVERY firing time and EVERY neuron that has fired across the 256 images in the mini-batch.

        # i is the coordinate list, but instead of an array it is a Tensor.
        i = torch.LongTensor(coo).to(device)
        # v contains 1s for every spike across all images. This is because we wish to place 1s for every spike
        # in the sparse Tensor.
        v = torch.FloatTensor(numpy.ones(len(coo[0]))).to(device)
        print("i:", i, "\nv:", v)
        # The sparse API is used to create sparse Tensors, so Tensors that contain the values "v"
        # at coordinates "i". All other locations in sparse Tensors are filled with 0s.
        # This is equivalent to creating a Tensor of the desired size, filling it with zeros and
        # manually inserting the required values in the required locations.
        x_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, num_steps, num_units])).to(device)
        # The y values are put in a dense Tensor, because they are labels instead of spike times.
        y_batch = torch.tensor(labels_[batch_index], device=device)

        # The yield keyword is equivalent to return, with the difference
        # being that after the function's execution is suspended and a value
        # is sent back to the caller, the function retains enough state to
        # enable it to resume where it is left off. When resumed, the function
        # continues execution immediately after the last yield run. This way,
        # the code can produce a series of values over time rather than computing
        # them all at once and sending them back as a list.

        # Thus, we receive the output spike trains and labels of a single batch every time this function
        # is run. When it is run again, the counter is incremented and the outputs of the next batch of
        # images is calculated.
        yield x_batch.to(device=device), y_batch.to(device=device)
        counter += 1


# Here we will set up the spiking network model like the previous tutorial.
tau_mem = 10e-3
tau_syn = 5e-3

alpha = float(numpy.exp(-time_step / tau_syn))
beta = float(numpy.exp(-time_step / tau_mem))

weight_scale = 7 * (1.0 - beta)

w1 = torch.empty((num_inputs, num_hidden), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / numpy.sqrt(num_inputs))

w2 = torch.empty((num_hidden, num_outputs), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / numpy.sqrt(num_hidden))

print("Initialisation complete")


# Now we implement the plotting function as before.
def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5):
    gs = GridSpec(*dim)
    if spk is not None:
        dat = (mem + spike_height * spk).detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(numpy.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharey=a0)
        ax.plot(dat[i])
        ax.axis("off")


# Then, the surrogate gradient is configured.
class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn = SurrGradSpike.apply


def run_snn(inputs):
    h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    syn = torch.zeros((batch_size, num_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, num_hidden), device=device, dtype=dtype)

    mem_rec = [mem]
    spk_rec = [mem]

    # Compute hidden layer activity
    for t in range(num_steps):
        mthr = mem - 1.0
        out = spike_fn(mthr)
        rst = torch.zeros_like(mem)
        c = (mthr > 0)
        rst[c] = torch.ones_like(mem)[c]

        new_syn = alpha * syn + h1[:, t]
        new_mem = beta * mem + syn - rst

        mem = new_mem
        syn = new_syn

        mem_rec.append(mem)
        spk_rec.append(out)

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))
    flt = torch.zeros((batch_size, num_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size, num_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(num_steps):
        new_flt = alpha * flt + h2[:, t]
        new_out = beta * out + flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec, dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs


def train(x_data, y_data, lr=2e-3, num_epochs=10):
    params = [w1, w2]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    log_softmax_fn = torch.nn.LogSoftmax(dim=1)
    loss_fn = torch.nn.NLLLoss()

    loss_hist = []
    for e in range(num_epochs):
        local_loss = []
        for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, num_steps, num_inputs):
            output, _ = run_snn(x_local.to_dense())
            m, _ = torch.max(output, 1)
            log_p_y = log_softmax_fn(m)
            loss_val = loss_fn(log_p_y, y_local.long())

            optimizer.zero_grad()
            loss_val.backward()
            print(loss_val.grad_fn)
            optimizer.step()
            local_loss.append(loss_val.item())
        mean_loss = numpy.mean(local_loss)
        print("Epoch %i: loss=%.5f" % (e + 1, mean_loss))
        loss_hist.append(mean_loss)

    return loss_hist


def compute_classification_accuracy(x_data, y_data):
    """ Computes classification accuracy on supplied data in batches. """
    accs = []
    for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, num_steps, num_inputs, shuffle=False):
        output, _ = run_snn(x_local.to_dense())
        m, _ = torch.max(output, 1)  # max over time
        _, am = torch.max(m, 1)  # argmax over output units
        tmp = numpy.mean((y_local == am).detach().cpu().numpy())  # compare to labels
        accs.append(tmp)
    return numpy.mean(accs)


loss_hist = train(x_train, y_train, lr=2e-4, num_epochs=1)

print("Training accuracy: %.3f" % (compute_classification_accuracy(x_train, y_train)))
print("Test accuracy: %.3f" % (compute_classification_accuracy(x_test, y_test)))


def get_mini_batch(x_data, y_data, shuffle=False):
    for ret in sparse_data_generator(x_data, y_data, batch_size, num_steps, num_inputs, shuffle=shuffle):
        return ret


x_batch, y_batch = get_mini_batch(x_test, y_test)
output, other_recordings = run_snn(x_batch.to_dense())
mem_rec, spk_rec = other_recordings

fig = plt.figure(dpi=100)
plot_voltage_traces(mem_rec, spk_rec)
plt.show()
plot_voltage_traces(output)
plt.show()

num_plt = 4
gs = GridSpec(1, num_plt)
fig = plt.figure(figsize=(7, 3), dpi=150)
for i in range(num_plt):
    plt.subplot(gs[i])
    plt.imshow(spk_rec[i].detach().cpu().numpy().T, cmap=plt.cm.gray_r, origin="lower")
    if i == 0:
        plt.xlabel("Time")
        plt.ylabel("Units")

    sns.despine()

plt.show()
