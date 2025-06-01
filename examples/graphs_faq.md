#### A. What is a Graph?

<details>
    <summary> <b><i>🌀 Expand for an short explanations on Graphs</i></b> </summary>
<div style="display: flex; align-items: flex-start;">
<div style="flex: 1; padding-right: 20px;">

Before we continue it might be good to briefly explain what a Graph even in is!

A Graph is a data structure consisting of:
- Nodes: Individual elements in the graph
- Edges: Connections between nodes

The graph is typically represented by:
- [Adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix): Shows connections between nodes
- Node features: Attributes or properties of each node
- Edge features: Attributes of the connections between nodes

The image on the right represents a stylized version of a frame of tracking data in soccer.

In section 6.1 we can see what this looks like in Python.

</div>
<div style="flex: 1;">

![Graph representation](https://github.com/UnravelSports/unravelsports.github.io/blob/main/imgs/what-is-a-graph-4.png?raw=true)

</div>
</div>
</details>

#### B. What are all GraphConverter settings?

<details>
    <summary><b><i> 🌀 ⚽ 🏈  Expand for a full table of additional <u>optional</u> GraphConverter parameters </i></b></summary><br>

| Parameter                           | Type      | Description                                                                                                                                                                                                                                                                                                                                                                               | Default         | Sport |
|-------------------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|-----------------|
| `prediction`                        | bool      | When True use the converter to create Graph dataset to apply a pre-trained model to, no labels required. Defaults to False.                                                                                                                                                                                                                                                               | False           | ⚽ 🏈 |
| `adjacency_matrix_connect_type`     | str       | The type of connection used in the adjacency matrix, typically related to the ball. Choose from 'ball', 'ball_carrier' or 'no_connection'                                                                                                                                                                                                                                                 | 'ball'          | ⚽ 🏈 |
| `adjacency_matrix_type`             | str       | The type of adjacency matrix, indicating how connections are structured, such as split by team. Choose from 'delaunay' , 'split_by_team', 'dense', 'dense_ap' or 'dense_dp'                                                                                                                                                                                                                | 'split_by_team' | ⚽ 🏈 |
| `self_loop_ball`                    | bool      | Flag to indicate if the ball node should have a self-loop, aka be connected with itself and not only player(s)                                                                                                                                                                                                                                                                            | True            | ⚽ 🏈 |
| `label_type`                        | str       | The type of prediction label used. Currently only supports 'binary'                                                                                                                                                                                                                                                                                                                       | 'binary'        | ⚽ 🏈 |
| `random_seed`                       | int, bool | When a random_seed is given, it will randomly shuffle an individual Graph without changing the underlying structure. When set to True, it will shuffle every frame differently; False won't shuffle. Advised to set True when creating an actual dataset to support Permutation Invariance.                                                                                               | False           | ⚽ 🏈 |
| `pad`                               | bool      | True pads to a total amount of 22 players and ball (so 23x23 adjacency matrix). It dynamically changes the edge feature padding size based on the combination of AdjacencyMatrixConnectType and AdjacencyMatrixType, and self_loop_ball. No need to set padding because smaller and larger graphs can all be used in the same dataset.                                                    | False           | ⚽ 🏈 |
| `verbose`                           | bool      | The converter logs warnings / error messages when specific frames have no coordinates, or other missing information. False mutes all of these warnings.                                                                                                                                                                                                                                   | False           | ⚽ 🏈 |
| `defending_team_node_value`         | float     | Value for the node feature when player is on defending team. Should be between 0 and 1 including.                                                                                                                                                                                                                                                                                         | 0.1             | ⚽ 🏈 
| `attacking_non_qb_node_value` | float     | Value for the node feature when player is NOT the QB, but is on the attacking team                                                                                                                                                                                                  | 0.1             | 🏈  |
| `chunk_size` | int     | Set to determine size of conversions from Polars to Graphs. Preferred setting depends on available computing power                                                                                                                                                                                                              | 2_000           | 🏈 |
| `non_potential_receiver_node_value` | float     | Value for the node feature when player is NOT a potential receiver of a pass (when on opposing team or in possession of the ball). Should be between 0 and 1 including.                                                                                                                                                                                                                   | 0.1             | ⚽ |



</details>

#### C. What features does each Graph have?

<details>
    <summary> <b><i> 🌀 ⚽  Expand for a full list of Soccer features (note: `SoccerGraphConverter`, `SoccerGraphConverter` has slightly different features) </b></i></summary>
    
| Variable | Datatype                          | Index | Features                                                                                                                        |
|----------|-----------------------------------|-------|---------------------------------------------------------------------------------------------------------------------------------|
| a        | np.array of shape (players+ball, players+ball) |       | -                                                                                                                               |
| x        | np.array of shape (n_nodes, n_node_features) | 0     | normalized x-coordinate                                                                                                         |
|          |                                   | 1     | normalized y-coordinate                                                                                                         |
|          |                                   | 2     | x component of the velocity unit vector                                                                                         |
|          |                                   | 3     | y component of the velocity unit vector                                                                                         |
|          |                                   | 4     | normalized speed                                                                                                                |
|          |                                   | 5     | normalized angle of velocity vector                                                                                             |
|          |                                   | 6     | normalized distance to goal                                                                                                     |
|          |                                   | 7     | normalized angle to goal                                                                                                        |
|          |                                   | 8     | normalized distance to ball                                                                                                     |
|          |                                   | 9     | normalized angle to ball                                                                                                        |
|          |                                   | 10    | attacking (1) or defending team (`defending_team_node_value`)                                                                   |
|          |                                   | 11    | potential receiver (1) else `non_potential_receiver_node_value`                                                                 |
| e        | np.array of shape (np.non_zero(a), n_edge_features) | 0     | normalized inter-player distance                                                                                                |
|          |                                   | 1     | normalized inter-player speed difference                                                                                        |
|          |                                   | 2     | inter-player angle cosine                                                                                                       |
|          |                                   | 3     | inter-player angle sine                                                                                                         |
|          |                                   | 4     | inter-player velocity vector cosine                                                                                             |
|          |                                   | 5     | inter-player velocity vector sine                                                                                               |
|          |                                   | 6     | optional: 1 if two players are connected else 0 according to delaunay adjacency matrix. Only if adjacency_matrix_type is NOT 'delauney' |
| y        | np.array                          |       | -                                                                                                                               |
| id       | int, str, None                    |       | -                                                                                                                               |

</details>
<br>
<details>
    <summary> <b><i> 🌀 🏈  Expand for a full list of American Football features </b></i></summary>
    
| Variable | Datatype                          | Index | Features                                                                                                                         |
|----------|-----------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------|
| a        | np.array of shape (players+ball, players+ball) |       | -                                                                                                                                |
| x        | np.array of shape (n_nodes, n_node_features) | 0     | normalized x-coordinate                                                                                                          |
|          |                                   | 1     | normalized y-coordinate                                                                                                          |
|          |                                   | 2     | x component of the speed unit vector                                                                                             |
|          |                                   | 3     | y component of the speed unit vector                                                                                             |
|          |                                   | 4     | normalized speed                                                                                                                 |
|          |                                   | 5     | x component of the acceleration unit vector                                                                                      |
|          |                                   | 6     | y component of the acceleration unit vector                                                                                      |
|          |                                   | 7     | normalized acceleration                                                                                                          |
|          |                                   | 8     | sine of the normalized direction (angle)                                                                                         |
|          |                                   | 9     | cosine of the normalized direction (angle)                                                                                       |
|          |                                   | 10    | sine of the normalized orientation (angle)                                                                                       |
|          |                                   | 11    | cosine of the normalized orientation (angle)                                                                                     |
|          |                                   | 12    | normalized distance to goal                                                                                                      |
|          |                                   | 13    | normalized distance to ball                                                                                                      |
|          |                                   | 14    | normalized distance to end zone                                                                                                  |
|          |                                   | 15    | possession team or defending team (`defending_team_node_value`)                                         indicator                                                                                                       |
|          |                                   | 16    | quarterback indicator or `attacking_non_qb_node_value` or 0 (defending team)                                                                                                        |
|          |                                   | 17    | ball indicator                                                                                                                   |
|          |                                   | 18    | normalized weight                                                                                                                |
|          |                                   | 19    | normalized height                                                                                                                |
| e        | np.array of shape (np.non_zero(a), n_edge_features) | 0     | inter-player distance                                                                                                            |
|          |                                   | 1     | inter-player speed difference                                                                                                    |
|          |                                   | 2     | inter-player acceleration difference                                                                                             |
|          |                                   | 3     | cosine of the inter-player positional angle                                                                                      |
|          |                                   | 4     | sine of the inter-player positional angle                                                                                        |
|          |                                   | 5     | cosine of the inter-player direction angle                                                                                       |
|          |                                   | 6     | sine of the inter-player direction angle                                                                                         |
|          |                                   | 7     | cosine of the inter-player orientation angle                                                                                     |
|          |                                   | 8     | sine of the inter-player orientation angle                                                                                       |
| y        | np.array                          |       | -                                                                                                                                |
| id       | int, str, None                    |       | -                                                                                                                                |

</details>

#### D. What is a CustomGraphDataset?

<details>
    <summary><b><i> 🌀 Expand for a short explanation on CustomSpektralDataset<i></b></summary><br>

Let's have a look at the internals of our `CustomSpektralDataset`. This dataset class contains a list of graphs, available through `dataset.graphs`.

The first item in our dataset has 23 nodes, 12 features per node and 7 features per edge.

<div style="border: 2px solid #ddd; border-radius: 5px; padding: 10px; background-color: ##282C34;">

```python
dataset.graphs[0]

>>> Graph(n_nodes=23, n_node_features=12, n_edge_features=7, n_labels=1)
```

The `CustomSpektralDataset` also allows us to split our data into train and test sets (and validation set if required) by using either:
- `dataset.split_test_train_validation()`
- `dataset.split_test_train()`

<br>
</details>
<br>
<details>
    <summary><b><i> 🌀 Expand for a short explanation on the representation of adjacency matrix <i></b></summary><br>

##### Adjacency Matrix
The **adjacency matrix** is represented as a [compressed sparse row matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix), as required by Spektral. A 'normal' version of this same matrix would be of shape 23x23 filled with zero's and one's in places where two players (or ball) are connected. 

Because we set `adjacency_matrix_type='split_by_team'` and `adjacency_matrix_connect_type="ball"` this results in a total of 287 connections (ones), namely between:
- `adjacency_matrix_type='split_by_team'`:
    - All players on team A (11 * 11) 
    - All players on team B (11 * 11)
    - Ball connected to ball (1)
- `adjacency_matrix_connect_type="ball"`
    - All players and the ball (22) 
    - The ball and all players (22)

<div style="border: 2px solid #ddd; border-radius: 5px; padding: 10px; background-color: ##282C34;">

```python
dataset.graphs[0].a
>>> <Compressed Sparse Row sparse matrix of dtype 'float64'
	    with 287 stored elements and shape (23, 23)>
```
<br>
</details>
<br>
<details>
    <summary><b><i> 🌀 Expand for a short explanation on the representation of node feature matrix <i></b></summary><br>

##### Node Features
The **node features** are described using a regular Numpy array. Each column represents one feature and every row represents one player. 

The ball is presented in the last row, unless we set `random_seed=True` then every Graph gets randomly shuffled (while leaving connections in tact).

See the bullet points in **5. Load Kloppy Data, Convert and Store** to learn which column represents which feature.

The rows filled with zero's are 'empty' players created because we set `pad=True`. Graph Neural Networks are flexible enough to deal with all sorts of different graph shapes in the same dataset, normally it's not actually necessary to add these empty players, even for incomplete data with only a couple players in frame.

<div style="border: 2px solid #ddd; border-radius: 5px; padding: 10px; background-color: ##282C34;">

```python
dataset.graphs[0].x
>>> [[-0.163 -0.135  0.245 -0.97   0.007  0.289  0.959  0.191  0.059  0.376  1.     1.   ]
 [-0.332  0.011 -0.061  0.998  0.02   0.76   1.015  0.177  0.029  0.009  1.     0.1  ]
 [ 0.021 -0.072  0.987 -0.162  0.017  0.474  0.88   0.203  0.121  0.468  1.     1.   ]
 [-0.144  0.232  0.343  0.939  0.024  0.694  0.924  0.186  0.077  0.638  1.     1.   ]
 [-0.252  0.302  0.99   0.141  0.032  0.523  0.964  0.176  0.078  0.741  1.     1.   ]
 [ 0.012  0.573  0.834 -0.551  0.035  0.407  0.842  0.191  0.19   0.646  1.     1.   ]
 [-0.293  0.686  0.999 -0.045  0.044  0.493  0.966  0.163  0.182  0.761  1.     1.   ]
 [ 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.   ]
 [ 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.   ]
 [ 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.   ]
 ...
 [ 0.202  0.124 -0.874  0.486  0.024  0.919  0.791  0.214  0.197  0.524  0.1    0.1  ]
 [ 0.404  0.143 -0.997  0.08   0.029  0.987  0.709  0.23   0.281  0.519  0.1    0.1  ]
 [ 0.195 -0.391  0.48  -0.877  0.014  0.33   0.847  0.218  0.222  0.417  0.1    0.1  ]
 [ 0.212 -0.063  0.982 -0.187  0.009  0.47   0.804  0.217  0.2    0.483  0.1    0.1  ]
 [-0.03   0.248 -0.996  0.091  0.021  0.986  0.876  0.194  0.116  0.591  0.1    0.1  ]
 [ 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.   ]
 [ 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.   ]
 [ 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.   ]
 [ 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.   ]
 [-0.262  0.016  0.937 -0.35   0.036  0.443  0.986  0.044  0.     0.     0.     0.   ]]

 
dataset.graphs[0].x.shape
>>> (23, 12)
```
<br>
</details>
<br>
<details>
    <summary><b><i> 🌀 Expand for a short explanation on the representation of edge feature matrix <i></b></summary><br>

##### Edge Features
The **edge features** are also represented in a regular Numpy array. Again, each column represents one feature, and every row decribes the connection between two players, or player and ball.

We saw before how the **adjacency matrix** was presented in a Sparse Row Matrix with 287 rows. It is no coincidence this lines up perfectly with the **edge feature matrix**. 

<div style="border: 2px solid #ddd; border-radius: 5px; padding: 10px; background-color: ##282C34;">

```python
dataset.graphs[0].e
>>> [[ 0.     0.     1.     0.5    0.5    1.     0.   ]
 [ 0.081  0.006  0.936  0.255  0.21   0.907  1.   ]
 [ 0.079  0.004  0.012  0.391  0.     0.515  1.   ]
 [ 0.1    0.007  0.46   0.002  0.005  0.571  1.   ]
 [ 0.125  0.011  0.65   0.023  0.474  0.999  0.   ]
 [ 0.206  0.012  0.322  0.033  0.535  0.999  0.   ]
 [ 0.23   0.016  0.619  0.014  0.567  0.996  0.   ]
 [ 0.     0.     0.     0.     0.     0.     0.   ]
 [ 0.     0.     0.     0.     0.     0.     0.   ]
 [ 0.     0.     0.     0.     0.     0.     0.   ]
 ...
 [ 0.197 -0.025  0.005  0.426  0.929  0.757  1.   ]
 [ 0.281 -0.023  0.004  0.439  0.959  0.699  1.   ]
 [ 0.222 -0.03   0.067  0.75   0.979  0.643  1.   ]
 [ 0.2   -0.032  0.003  0.554  0.982  0.633  1.   ]
 [ 0.116 -0.026  0.08   0.229  0.82   0.884  1.   ]
 [ 0.     0.     0.     0.     0.     0.     1.   ]
 [ 0.     0.     0.     0.     0.     0.     1.   ]
 [ 0.     0.     0.     0.     0.     0.     1.   ]
 [ 0.     0.     0.     0.     0.     0.     1.   ]
 [ 0.     0.     1.     0.5    0.5    1.     1.   ]]

 dataset.graphs[0].e.shape
 (287, 7)
```
<br>
</details>

