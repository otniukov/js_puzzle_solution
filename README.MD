# Intro

The puzzle is to restore ordering of blocks of deep neural model. It consists of 48 residual block with same dimension size, last layer (fig1). For each layer we have weights and biases matrices. There are also provided sample of data with features, targets and the model’s predictions.

I break down solution into 3 parts - pairing linear layers to get blocks, preordering, swap adjacent blocks. Last 2 parts could be formulated as geometry understanding and functional fixing. My implementation performs this in two stages:

1. maximize trajectory smoothness (greedy similarity),
2. optimize function (adjacent swaps).

# Pairing

There are 97 layers. The last layer has dimension [48,1]. Input , output. 

The model is well trained based on given dataset and MSE loss on predictions and truths. So, residual connections create gradient coherence. 

Each residual blocks behaves like a small encoder-decoder. It was shown that residual MLPs become close to low-rank autoencoders. I suggest, that blocks of original model converge toward this. In other words, encoder and decoder naturally become transposes and statistic $cos(Win[k,:],Wout[:,k])$ is literally measuring how close one hidden neuron is to this encoder–decoder symmetry. 

So, given two sets of blocks, Win and Wout, I first compute an alignment matrix $A∈RNin×Nout$, where each element

$Aij=HiddenAlignment(Win(i),Wout(j))$

measures the similarity between the hidden representations of the ii-th input block and the jj-th output block. For each input block, I identify the output block with the maximum alignment score,

$j∗=arg⁡max⁡jAij$,

and form the pair (i,j∗)(i,j∗). These matched pairs are then converted into paired residual blocks by retrieving the corresponding block metadata from the input and output lists.

# Preordering

Loss optimization is great, but there are a lot of combinations. So we need pre ordering and then easy swaps.

Since

$x_{k+1}=x_k+f_k(x_k)$

a ResNet can be viewed as an Euler discretization of an ODE

$\frac{dx}{dt}=v(x,t)$

Each residual block is one discretized vector field. 

If adjacent layers correspond to nearby times in the ODE, then

$[f_i \approx f_{i+1}]$

and cosine similarity should decay smoothly with distance. Residual layers approximate continuous dynamics.

That is exactly what Neural ODE theory predicts. So, I measure the similarity of the **vector fields** induced by each residual block.

First, I calculate similarity matrix. (Are two layers producing updates in similar directions?)

$$
S_{ij} = \mathbb{E}_x\left[
\cos\!\left(h_i(x),\,h_j(x)\right)
\right]
$$

where

$$
h_i(x_0) = B_i(x_0) - x_0
$$

is residual update. For each block I input raw feature vector x_0.

Analyzing similarity matrix there are 2 standing out blocks. First one is 

`pos: 30, top idx: [5 2]; top vals: [0.10724244 0.10460617 0.09398133]`

It’s probably block before last layer, well trained ResNets tend to have layer before head is very odd wrt to others.

Second one is 

`pos: 34, top idx: [21 36]; top vals: [0.80651313 0.7463321  0.647884  ]; dist: [-13   2]`

Since I calculate update of k-th block only on x_0, not on x_k-1, It’s probably first block of the model. 

So I have starting block and ending block. I perform straight forward greedy walk. I construct a pre-order traversal using a greedy nearest-neighbor strategy, where each subsequent block is selected as the most similar unvisited block according to the pairwise similarity matrix, beginning from a specified starting block.

# Ordering

MSE Loss for prediction of real model and current ordering. I start with pre-ordering.

I use Traveling Salesman heuristics to get final ordering with zero Loss.

Local 2-opt optimization, but my objective is network MSE instead of path length.

Adjacent swaps to avoid 48! combinations.

# Discussion

Preordering gives 0.9+ spearman correlation to original order. Moreover, h_i(x_0) doesn’t measure the true dynamics of the trained network.

How ReLU changes encoder-decoder stability
