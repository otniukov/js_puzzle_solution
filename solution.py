import pandas as pd
import torch
import torch.nn as nn
from torch import linalg as LA
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np



class Block(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        residual = x
        x = self.inp(x)
        x = self.activation(x)
        x = self.out(x)
        return residual + x


class LastLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.layer(x)

    
def build_block(inp, out):
    residual_block = Block(48, 96)
    residual_block.inp.weight = nn.Parameter(inp["weight"])
    residual_block.inp.bias = nn.Parameter(inp["bias"])
    residual_block.out.weight = nn.Parameter(out["weight"])
    residual_block.out.bias = nn.Parameter(out["bias"])

    return residual_block
    

def build_model(block_list, pieces, last_idx):
    modules = []
    
    for inp_idx, out_idx in block_list:
        block = build_block(pieces[inp_idx], pieces[out_idx])
        modules.append(block)
        
    last_layer = LastLayer(48, 1)
    last_layer.layer.weight = nn.Parameter(pieces[last_idx]["weight"])
    last_layer.layer.bias = nn.Parameter(pieces[last_idx]["bias"])
    modules.append(last_layer)
    
    return nn.Sequential(*modules)


def load_pieces(pieces_path):
    pieces = []
    for i in range(97):
        d = torch.load(pieces_path + f"piece_{i}.pth", map_location=torch.device("cpu"))
        d["pos"] = i
        pieces.append(d)
    return pieces


def get_layers(pieces):
    win_list = []
    wout_list = []
    #head = None
    for i in range(97):
        if pieces[i]["bias"].shape[0] == 96:
            win_list.append(pieces[i])
        elif pieces[i]["bias"].shape[0] == 48:
            wout_list.append(pieces[i])
        elif pieces[i]["bias"].shape[0] == 1:
            #head = pieces[i]
            last_idx = i
    return win_list, wout_list, last_idx


def hidden_alignment(Win, Wout):
    sims = []

    for k in range(Win.shape[0]):

        a = Win[k]
        b = Wout[:, k]

        sim = torch.nn.functional.cosine_similarity(a, b, dim=0)

        sims.append(sim.item())

    return sum(abs(x) for x in sims) / len(sims)


def make_pairs(win_list, wout_list):

    align = torch.zeros(len(win_list), len(wout_list))

    for i, Win in enumerate(win_list):
        for j, Wout in enumerate(wout_list):
            align[i, j] = hidden_alignment(win_list[i]["weight"], wout_list[j]["weight"])

    #plt.imshow(align.numpy())
    #plt.colorbar()

    best_cols = align.argmax(axis=1)

    for row, col in enumerate(best_cols):
        print(f"Win {row:2d} -> Wout {col:2d}  score {align[row, col]}")

    pairs = [(row, col.item()) for row, col in enumerate(best_cols)]

    matched_scores = np.array([align[r, c] for r, c in pairs])

    print("mean matched:", matched_scores.mean())
    print("min matched :", matched_scores.min())
    print("max matched :", matched_scores.max())

    print("global mean:", align.mean().numpy())
    
    # make residual blocks
    paired_blocks = []
    for p_in, p_out in pairs:
        paired_blocks.append([win_list[p_in], wout_list[p_out]])
    
    return pairs, paired_blocks


@torch.no_grad()
def analyze_resnet_raw(pieces, paired_blocks, X, y_pred):
    '''
    Computing the residual-update similarity matrix
    '''
    n_layers = len(paired_blocks)

    n_samples = 0
    
    # S_ij
    sim_matrix = torch.zeros(n_layers, n_layers)
    
    for x, y in zip(X, y_pred):
    
        h_in = x
        residuals = []
        indxs = [[x["pos"], y["pos"]] for x, y in paired_blocks]

        for inp_idx, out_idx in indxs:

            residual_block = build_block(pieces[inp_idx], pieces[out_idx])
            h_out = residual_block(h_in)

            f = h_out - h_in
            residuals.append(f)
            
        # compute S_ij
        for i in range(n_layers):
            Ri = residuals[i].flatten()

            for j in range(i, n_layers):
                Rj = residuals[j].flatten()

                cos = F.cosine_similarity(Ri, Rj, dim=-1).mean()

                sim_matrix[i, j] += cos
                if i != j:
                    sim_matrix[j, i] += cos

        n_samples += 1

    # normalize
    sim_matrix /= n_samples
    
    print("Similarity stats:")
    for pos in range(len(sim_matrix)):
        top_args = np.argsort(sim_matrix.numpy()[pos])[::-1]
        top_val = sim_matrix.numpy()[pos][top_args[1]]
        print(f"pos: {pos},", f"closest idx: {top_args[1:3]};", f"top vals: {sim_matrix.numpy()[pos][top_args[1:4]]};")

    #return {"sim_matrix": sim_matrix.cpu()}
    return sim_matrix


def load_data(path):
    df = pd.read_csv(path)
    rows_X = df[[x for x in df.columns if x not in ["pred", "true"]]].to_numpy()
    rows_pred = df["pred"].to_numpy()
    X = torch.tensor(rows_X, dtype=torch.float32)
    y_pred = torch.tensor(rows_pred, dtype=torch.float32)
    return X, y_pred
                                               
                                               
def get_pre_order(sim_matrix, start_pos, paired_blocks):
    path = []

    while len(path) < 48:
        if not path:
            path.append(start_pos)
        top_closest = np.argsort(sim_matrix[start_pos].numpy())[::-1]
        for next_pos in top_closest:
            if next_pos not in path:
                path.append(next_pos)
                start_pos = next_pos
                break
                
    pre_order = [[paired_blocks[idx][0]['pos'], paired_blocks[idx][1]['pos']] for idx in path]
    print(path)
    return pre_order


@torch.no_grad()
def get_mse_loss(block_list, pieces, last_idx, X, y):
    model = build_model(block_list, pieces, last_idx)
    model.eval()
    
    return ((model(X).squeeze() - y)**2).mean().item()


def swaps(order, pieces, last_idx, X, y_pred):
    
    cur_ordering = list(order)
    num_blocks = len(order)
    cur_ordering_loss = get_mse_loss(cur_ordering, pieces, last_idx, X, y_pred)
    total_swaps = 0
    
    max_iterations = 10

    for itr in range(max_iterations):
        itr_swaps = 0
        for i in range(1, num_blocks - 1):
            swap_ordering = list(cur_ordering)
            swap_ordering[i], swap_ordering[i + 1] = swap_ordering[i + 1], swap_ordering[i]
            loss_swapped = get_mse_loss(swap_ordering, pieces, last_idx, X, y_pred)
            if loss_swapped < cur_ordering_loss:
                cur_ordering = swap_ordering
                cur_ordering_loss = loss_swapped
                itr_swaps += 1

        for i in range(num_blocks - 2, 0, -1):
            swap_ordering = list(cur_ordering)
            swap_ordering[i], swap_ordering[i + 1] = swap_ordering[i + 1], swap_ordering[i]
            loss_swapped = get_mse_loss(swap_ordering, pieces, last_idx, X, y_pred)
            if loss_swapped < cur_ordering_loss:
                cur_ordering = swap_ordering
                cur_ordering_loss = loss_swapped
                itr_swaps += 1

        total_swaps += itr_swaps
        print(f"Iteration: {itr:2d}, swaps: {itr_swaps:3d}, current Loss = {cur_ordering_loss:.10f}")
        if itr_swaps == 0:
            print(f"No swaps")
            break
    print(f"Final Loss = {cur_ordering_loss:.10f}, total_swaps: {total_swaps}")
    return cur_ordering


def main():

    pieces_path = "historical_data_and_pieces/pieces/"
    data_path = "historical_data_and_pieces/historical_data.csv"
    pieces = load_pieces(pieces_path)

    # load pieces
    win_list, wout_list, last_idx = get_layers(pieces)
    print(win_list[0]["weight"].shape, wout_list[0]["weight"].shape)

    # match Win, Wout layers
    pairs, paired_blocks = make_pairs(win_list, wout_list)

    # load data
    X, y_pred = load_data(data_path)

    # calculate blocks similarity
    sim_matrix = analyze_resnet_raw(pieces, paired_blocks, X, y_pred)

    # pre-order, starting block index=21 with the highest similarity values
    start_pos = 21
    pre_order = get_pre_order(sim_matrix, start_pos, paired_blocks)
    print(f"Pre-order: {pre_order}")

    # final ordering
    final_order = swaps(pre_order, pieces, last_idx, X, y_pred)

    answer = []
    for inp_idx, out_idx in final_order:
        answer.append(inp_idx)
        answer.append(out_idx)
    answer.append(last_idx)
    print(f"True ordering: [{', '.join(str(x) for x in answer)}]")


                                            
if __name__ == "__main__":
    main()

