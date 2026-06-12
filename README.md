# DalQ: Reconciling Accuracy and Efficiency in Vector Quantization

This is the source code of the method proposed in paper: "DalQ: Reconciling Accuracy and Efficiency in Vector Quantization" (accepted by ECML PKDD 2026).

## Dependencies
* g++ version: 13.1.0
* cmake version: 3.25.1
* CPU with AVX-512 support
* BLAS: sudo apt-get install libblas-dev
* LAPACK: sudo apt-get install liblapack-dev



## Compilation
```
sh build.sh
```

## Usage
### Create Index 

```bash
./build/create_index \
  "$BASE_FILE" \
  "$CENTROIDS_FILE" \
  "$CLUSTER_ID_FILE" \
  "$INDEX_FILE" \
  "$QUANTIZATION_BITS" \
  "$CLIP_FACTOR"
```

#### Parameter Description:
* **BASE_FILE**: The path to the original base vector data file.
* **CENTROIDS_FILE**: The path to the cluster centroids data file.
* **CLUSTER_ID_FILE**: The path to the file containing the cluster IDs corresponding to the vectors.
* **INDEX_FILE**: The path where the output index file will be saved after construction.
* **QUANTIZATION_BITS**: The number of quantization bits (e.g., 4 or 8).
* **CLIP_FACTOR**: The clip factor.



### Test Mean Squared Error (MSE)

```bash
./build/test_mse "$INDEX_FILE" "$BASE_FILE"
```

#### Parameter Description:
* **INDEX_FILE**: The path to the index file.
* **BASE_FILE**: The path to the original base vector data file.



### Test ANN Performance

```bash
./build/test_qps "$INDEX_FILE" "$QUERY_VECTORS" "$GROUNDTRUTH"
```

#### Parameter Description:
* **INDEX_FILE**: The path to the index file.
* **QUERY_VECTORS**: The path to the query vector data file used for testing.
* **GROUNDTRUTH**: The path to the ground truth file.

