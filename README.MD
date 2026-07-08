# Intro

The puzzle is to restore ordering of blocks of deep neural model. It consists of 48 residual block with same dimension size, last layer (fig1). For each layer we have weights and biases matrices. There are also provided sample of data with features, targets and the model’s predictions.

I break down solution into 3 parts - pairing linear layers to get blocks, preordering, swap adjacent blocks. Last 2 parts could be formulated as geometry understanding and functional fixing. My implementation performs this in two stages:

1. maximize trajectory smoothness (greedy similarity),
2. optimize function (adjacent swaps).

## Pairing Residual Blocks

The model contains 97 linear layers. One of them is the final prediction layer with shape [48, 1], while the remaining 96 layers form the 48 residual blocks.

The model is well trained based on given dataset and MSE loss on predictions and truths. So, residual connections create gradient coherence. 

Previous work has shown that residual MLPs often exhibit low-rank autoencoder-like behavior. Motivated by this observation, I assume that the encoder and decoder within a residual block become approximately symmetric. Under this assumption, the encoder weight matrix and the transpose of the decoder weight matrix should be similar.

To quantify this property, a **hidden alignment** score is defined for each candidate input–output layer pair.
For a candidate pair of weight matrices, the statistic $$cos(Win[k,:],Wout[:,k])$$ measures how well the (k)-th hidden neuron satisfies this encoder–decoder symmetry. Aggregating these similarities over all hidden neurons yields a single alignment score for the layer pair. 

Given the sets of candidate input layers, $W_{\text{in}}$, and output layers, $W_{\text{out}}$, the alignment matrix $A \in \mathbb{R}^{N_{\text{in}} \times N_{\text{out}}}$ is computed, where 
$$
A_{ij} = \operatorname{HiddenAlignment}\!\left(
W_{\text{in}}^{(i)},
W_{\text{out}}^{(j)}
\right).
$$

For each input layer, the output layer with the highest alignment score is selected, $j^* = \arg\max_j A_{ij}$, forming a matched input–output layer pair. These pairs are then converted into residual blocks by combining the corresponding metadata from the input and output layer lists.



## Initial Block Ordering
! add about mse here! 
Directly optimizing the reconstruction loss over all possible block permutations is computationally infeasible. Instead, the search is divided into two stages: first, an approximate ordering is recovered, and then it is refined through local optimization.

A residual network can be interpreted as a discretization of a continuous dynamical system. Since each residual block performs the update

$$x_{k+1}=x_k+f_k(x_k),$$

the network approximates the ordinary differential equation


$$\frac{dx}{dt}=v(x,t),$$
where each residual block represents one discretized vector field.

Under this interpretation, adjacent residual blocks correspond to nearby time steps and therefore should induce similar vector fields,

$$f_i \approx f_{i+1}.$$


Consequently, the similarity between blocks is expected to decay smoothly with increasing distance in the network. This behavior is consistent with the Neural ODE interpretation of residual networks.

To quantify this property, a pairwise similarity matrix is computed using the residual updates produced by each block:
$$
S_{ij} = \mathbb{E}_x\left[
\cos\!\left(h_i(x),\,h_j(x)\right)
\right],
$$
where
$$
h_i(x_0) = B_i(x_0) - x_0
$$ 
is the residual update of block $i$ evaluated on the raw input features $x_0$.

The resulting similarity matrix reveals two distinctive outlier blocks. The first block exhibits uniformly low similarity to all other blocks:

pos: 30
top idx:  [5, 2]
top vals: [0.1072, 0.1046, 0.0940]

This block is hypothesized to be the final residual block, immediately preceding the prediction head. In well-trained ResNets, the representation just before the output layer is often highly specialized and therefore substantially different from the preceding hidden representations.

The second outlier displays high similarity only to a small subset of blocks:

pos: 34
top idx:  [21, 36]
top vals: [0.8065, 0.7463, 0.6479]
dist:     [-13, 2]

Because the residual updates are evaluated only on the initial input $x_0$, rather than on the intermediate activations $x_k$, this behavior is consistent with the first residual block, whose update is naturally computed on the raw input distribution.


## Ordering

MSE Loss for prediction of real model and current ordering. I start with pre-ordering.

I use Traveling Salesman heuristics to get final ordering with zero Loss.

Local 2-opt optimization, but my objective is network MSE instead of path length.

Adjacent swaps to avoid 48! combinations.

## Discussion

Preordering gives 0.9+ spearman correlation to original order. Moreover, h_i(x_0) doesn’t measure the true dynamics of the trained network.

How ReLU changes encoder-decoder stability