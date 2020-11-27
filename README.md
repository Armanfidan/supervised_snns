# supervised_snns
Training spiking neural networks using supervised learning, and getting around non-differentiable spiking outputs using surrogate gradients.

Supervised training using autodifferentiation library in PyTorch, using surrogate gradients instead of the binary spiking outputs in order to train Spiking Neural Networks. Based on Neftci et al, 2019, a paper about surrogate gradient training methods. Written based on Emre Neftci's description of the method, with added comments to aid understanding.
