# DalQ: Reconciling Accuracy and Efficiency in Vector Quantization

This is the source code of the method proposed in paper: "DalQ: Reconciling Accuracy and Efficiency in Vector Quantization" (accepted by ECML PKDD 2026).

## Dependencies
* g++ version: 13.1.0
* cmake version: 3.25.1
* CPU with AVX-512 support
* BLAS: sudo apt-get install libblas-dev
* LAPACK: sudo apt-get install liblapack-dev



## Compilation
```bash
sh build.sh
```

## Usage
Data files use the standard [fvecs/ivecs format](http://corpus-texmex.irisa.fr/) (`.fvecs` for float vectors, `.ivecs` for integer vectors).

### Create Index

```bash
./build/create_index \
  <base_fvecs> \
  <centroids_fvecs> \
  <cluster_id_ivecs> \
  <output_index> \
  <quantization_bits> \
  [clip_factor]
```

**Example:**

```bash
./build/create_index \
  dataset1_base.fvecs \
  dataset_centroids.fvecs \
  dataset_cluster_id.ivecs \
  dataset1.index \
  8 \
  0.95
```

#### Parameter Description:
* **base_fvecs**: The path to the original base vector data file.
* **centroids_fvecs**: The path to the cluster centroids data file.
* **cluster_id_ivecs**: The path to the file containing the cluster IDs corresponding to the vectors.
* **output_index**: The path where the output index file will be saved after construction.
* **quantization_bits**: The number of quantization bits (e.g., 4 or 8).
* **clip_factor**: The clip factor β ∈ (0, 1] that controls how tightly the quantization range is narrowed (default: 1.0).



### Test Mean Squared Error (MSE)

```bash
./build/test_mse <index_file> <base_fvecs>
```

**Example:**

```bash
./build/test_mse dataset1.index dataset1_base.fvecs
```

#### Parameter Description:
* **index_file**: The path to the index file.
* **base_fvecs**: The path to the original base vector data file.



### Test ANN Performance

```bash
./build/test_qps \
  <index_file> \
  <query_fvecs> \
  <groundtruth_ivecs> \
  [output_file]
```

**Example:**

```bash
./build/test_qps \
  dataset1.index \
  dataset1_query.fvecs \
  dataset1_groundtruth.ivecs \
  dataset1_results.csv
```

#### Parameter Description:
* **index_file**: The path to the index file.
* **query_fvecs**: The path to the query vector data file used for testing.
* **groundtruth_ivecs**: The path to the ground truth file.
* **output_file**: The path to the output CSV file (default: `results.csv`).
