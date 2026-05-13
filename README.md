# GPU-Accelerated Parallel Cholesky Factorization

## Overview
This project provides a parallel implementation of the Cholesky factorization algorithm designed to run efficiently on a GPU using CUDA. Cholesky factorization factors a symmetric positive definite (SPD) matrix A into an upper triangular matrix R and its transpose R^T (such that A = R^T R). This parallel implementation significantly accelerates the process compared to a serial CPU approach, achieving over a 4500x speedup for large matrices (4096 x 4096).

## Implementation Details
The core of the parallelization is handled by the `parallel_cholesky_factorization()` function, which iteratively processes one column (pivot) of the matrix at a time. For each pivot column, the implementation dispatches three specific CUDA kernel functions:

1. **`device_cholesky_normalize`**: Computes the square root of the diagonal element and divides the corresponding row elements by this value. Uses a 1D grid structure.
2. **`device_cholesky_submat_update`**: Updates the remaining submatrix by subtracting the outer product of the pivot row and column. Uses a 2D grid structure.
3. **`device_cholesky_lower_traingale_update`**: Sets elements below the diagonal to zero to maintain the upper triangular structure. Uses a 1D grid structure.

A key optimization in this design is the dynamic calculation of grid dimensions based on the matrix size and block dimensions, ensuring maximum thread occupancy and efficient GPU utilization.

## Getting Started

### Prerequisites
To compile and execute this code on the TAMU Grace cluster (or a similar environment), you need the Intel compiler and CUDA toolkit.

### Compilation
1. Load the required modules:
```bash
module load intel/2023a CUDA/12.2
```

2. Compile the CUDA code using `nvcc`:
```bash
nvcc -o cholesky_gpu_parallel.exe cholesky_gpu_parallel.cu
```

### Execution
Run the compiled executable by providing the matrix size, 1D block size, and 2D block dimensions:

```bash
./cholesky_gpu_parallel.exe <matrix_size> <block_1d_grid> <block_2d_grid_x> <block_2d_grid_y>
```

**Parameters:**
* `<matrix_size>`: Size of the matrix (Maximum value: 4096).
* `<block_1d_grid>`: Range for Block Size in the 1D Grid (1 to 1024).
* `<block_2d_grid_x>` & `<block_2d_grid_y>`: Dimensions for the 2D Grid. Their product must be <= 1024.

**Example:**
```bash
./cholesky_gpu_parallel.exe 1024 1024 32 32
```
*(Note: If executing on a login shell, you may occasionally encounter a CUDA memory error. Try again or submit it as an `sbatch` job).*

---

## Performance & Experimentation
The parallel implementation was heavily benchmarked against a serial execution baseline across three main experiments.

### 1. Matrix Size Scaling
Parallel performance yields significant benefits starting at matrix sizes of 64 (22x speedup) and reaches an optimal 4500x speedup at a matrix size of 4096. Larger matrices benefit from more regular memory access patterns and efficient shared memory reuse.

<img width="696" height="403" alt="image" src="https://github.com/user-attachments/assets/4fe612fd-84bb-4c1f-8ce8-f9ef05b5aedd" />
<br><br>

<img width="696" height="418" alt="image" src="https://github.com/user-attachments/assets/d86c35cd-56b6-4612-a17d-d1915926041d" />

### 2. 1D Grid Block Size Optimization
Varying the 1D block size (with a constant 1024x1024 matrix) showed that a block size of 16 yielded the best performance (3383x speedup), indicating an optimal balance between parallelism and overhead.

> <img width="696" height="411" alt="image" src="https://github.com/user-attachments/assets/5506b1b2-48a3-40b5-83bd-4815fde0fcd7" />

### 3. 2D Grid Dimensions Optimization
Experiments with different x and y dimensions for the 2D grid (maintaining a constant product of 1024 threads) demonstrated that row-oriented memory access patterns are highly efficient. The 64x16 dimension achieved the highest speedup.

> <img width="696" height="411" alt="image" src="https://github.com/user-attachments/assets/71e52634-de1b-47bc-ba08-1e7e9054ac62" />
