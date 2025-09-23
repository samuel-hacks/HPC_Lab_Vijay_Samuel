#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// A helper function to dynamically allocate a 2D matrix
int allocMatrix(int*** mat, int rows, int cols) {
    int* p = (int*)malloc(sizeof(int) * rows * cols);
    if (!p) return -1;
    *mat = (int**)malloc(rows * sizeof(int*));
    if (!mat) {
        free(p);
        return -1;
    }
    for (int i = 0; i < rows; i++) {
        (*mat)[i] = &(p[i * cols]);
    }
    return 0;
}

// A helper function to free a dynamically allocated 2D matrix
void freeMatrix(int ***mat) {
    if (*mat) {
        free(&((*mat)[0][0]));
        free(*mat);
        *mat = NULL;
    }
}

// Standard matrix multiplication for sub-blocks
void matrixMultiply(int **a, int **b, int aRows, int aCols, int bRows, int bCols, int ***c) {
    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bCols; j++) {
            int val = 0;
            for (int k = 0; k < aCols; k++) {
                val += a[i][k] * b[k][j];
            }
            (*c)[i][j] = val;
        }
    }
}

// Prints a matrix
void printMatrix(int **mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    MPI_Comm cartComm;
    int dim[2], period[2], reorder;
    int coord[2], id;
    int worldSize, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int M = 0, K = 0, N = 0;
    int procDim;
    int blockDimA_M, blockDimA_K, blockDimB_K, blockDimB_N;
    int **A = NULL, **B = NULL, **C = NULL;
    int **localA = NULL, **localB = NULL, **localC = NULL;

    if (rank == 0) {
        // Define matrix dimensions and check for valid configuration
        M = 4; // Rows of A, Rows of C
        K = 4; // Columns of A, Rows of B
        N = 4; // Columns of B, Columns of C
        
        double sqroot = sqrt(worldSize);
        if ((sqroot - floor(sqroot)) != 0) {
            printf("[ERROR] Number of processes must be a perfect square!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        procDim = (int)sqroot;
        if (M % procDim != 0 || K % procDim != 0 || N % procDim != 0) {
            printf("[ERROR] Matrix dimensions must be divisible by process grid dimension!\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        blockDimA_M = M / procDim;
        blockDimA_K = K / procDim;
        blockDimB_K = K / procDim;
        blockDimB_N = N / procDim;

        // Allocate and initialize global matrices A and B (example data)
        allocMatrix(&A, M, K);
        allocMatrix(&B, K, N);
        allocMatrix(&C, M, N);

        int val = 1;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                A[i][j] = val++;
            }
        }
        val = 1;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < N; j++) {
                B[i][j] = val++;
            }
        }
    }

    // Broadcast dimensions to all processes
    int dims[5];
    if (rank == 0) {
        dims[0] = M; dims[1] = K; dims[2] = N;
        dims[3] = procDim;
        dims[4] = blockDimA_M;
    }
    MPI_Bcast(dims, 5, MPI_INT, 0, MPI_COMM_WORLD);
    M = dims[0]; K = dims[1]; N = dims[2];
    procDim = dims[3];
    blockDimA_M = M / procDim;
    blockDimA_K = K / procDim;
    blockDimB_K = K / procDim;
    blockDimB_N = N / procDim;

    // Create a 2D Cartesian grid of processes
    dim[0] = procDim; dim[1] = procDim;
    period[0] = 1; period[1] = 1;
    reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartComm);
    MPI_Cart_coords(cartComm, rank, 2, coord);

    // Rank 0 distributes the data using MPI_Send
    if (rank == 0) {
        for (int i = 0; i < procDim; i++) {
            for (int j = 0; j < procDim; j++) {
                int destRank;
                // --- FIX 1: Use a named array for coordinates ---
                int current_coords[2] = {i, j};
                MPI_Cart_rank(cartComm, current_coords, &destRank);
                
                // Distribute blocks of A
                int startRowA = i * blockDimA_M;
                int startColA = j * blockDimA_K;
                for (int row = 0; row < blockDimA_M; row++) {
                    MPI_Send(&A[startRowA + row][startColA], blockDimA_K, MPI_INT, destRank, 0, cartComm);
                }

                // Distribute blocks of B
                int startRowB = i * blockDimB_K;
                int startColB = j * blockDimB_N;
                for (int row = 0; row < blockDimB_K; row++) {
                    MPI_Send(&B[startRowB + row][startColB], blockDimB_N, MPI_INT, destRank, 1, cartComm);
                }
            }
        }
    }

    // All other ranks receive their local blocks
    allocMatrix(&localA, blockDimA_M, blockDimA_K);
    allocMatrix(&localB, blockDimB_K, blockDimB_N);
    for (int row = 0; row < blockDimA_M; row++) {
        MPI_Recv(&(localA[row][0]), blockDimA_K, MPI_INT, 0, 0, cartComm, MPI_STATUS_IGNORE);
    }
    for (int row = 0; row < blockDimB_K; row++) {
        MPI_Recv(&(localB[row][0]), blockDimB_N, MPI_INT, 0, 1, cartComm, MPI_STATUS_IGNORE);
    }
    
    // Initial skewing of matrices A and B using MPI_Sendrecv_replace
    int initialShiftA_left, initialShiftA_right, initialShiftB_up, initialShiftB_down;
    MPI_Cart_shift(cartComm, 1, coord[0], &initialShiftA_left, &initialShiftA_right);
    MPI_Cart_shift(cartComm, 0, coord[1], &initialShiftB_up, &initialShiftB_down);
    MPI_Sendrecv_replace(&(localA[0][0]), blockDimA_M * blockDimA_K, MPI_INT, initialShiftA_left, 1, initialShiftA_right, 1, cartComm, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(&(localB[0][0]), blockDimB_K * blockDimB_N, MPI_INT, initialShiftB_up, 1, initialShiftB_down, 1, cartComm, MPI_STATUS_IGNORE);
    
    // Allocate and initialize local result matrix C
    allocMatrix(&localC, blockDimA_M, blockDimB_N);
    for (int i = 0; i < blockDimA_M; i++) {
        for (int j = 0; j < blockDimB_N; j++) {
            localC[i][j] = 0;
        }
    }
    
    // Main loop for Cannon's algorithm
    int** multiplyRes = NULL;
    allocMatrix(&multiplyRes, blockDimA_M, blockDimB_N);
    for (int k = 0; k < procDim; k++) {
        matrixMultiply(localA, localB, blockDimA_M, blockDimA_K, blockDimB_K, blockDimB_N, &multiplyRes);

        for (int i = 0; i < blockDimA_M; i++) {
            for (int j = 0; j < blockDimB_N; j++) {
                localC[i][j] += multiplyRes[i][j];
            }
        }

        // Cyclic shift of matrices A and B
        int shiftA_left, shiftA_right, shiftB_up, shiftB_down;
        MPI_Cart_shift(cartComm, 1, 1, &shiftA_left, &shiftA_right);
        MPI_Cart_shift(cartComm, 0, 1, &shiftB_up, &shiftB_down);
        MPI_Sendrecv_replace(&(localA[0][0]), blockDimA_M * blockDimA_K, MPI_INT, shiftA_left, 1, shiftA_right, 1, cartComm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(&(localB[0][0]), blockDimB_K * blockDimB_N, MPI_INT, shiftB_up, 1, shiftB_down, 1, cartComm, MPI_STATUS_IGNORE);
    }
    
    freeMatrix(&multiplyRes);

    // Rank 0 gathers results using MPI_Recv
    if (rank != 0) {
        for (int row = 0; row < blockDimA_M; row++) {
            MPI_Send(&(localC[row][0]), blockDimB_N, MPI_INT, 0, 2, cartComm);
        }
    } else {
        for (int i = 0; i < procDim; i++) {
            for (int j = 0; j < procDim; j++) {
                int sourceRank;
                // --- FIX 2: Use a named array for coordinates ---
                int current_coords[2] = {i, j};
                MPI_Cart_rank(cartComm, current_coords, &sourceRank);
                
                int startRowC = i * blockDimA_M;
                int startColC = j * blockDimB_N;

                if (sourceRank == 0) {
                    for(int row = 0; row < blockDimA_M; row++) {
                        for(int col = 0; col < blockDimB_N; col++) {
                            C[startRowC + row][startColC + col] = localC[row][col];
                        }
                    }
                } else {
                    for (int row = 0; row < blockDimA_M; row++) {
                        MPI_Recv(&C[startRowC + row][startColC], blockDimB_N, MPI_INT, sourceRank, 2, cartComm, MPI_STATUS_IGNORE);
                    }
                }
            }
        }
        printf("C matrix (final result):\n");
        printMatrix(C, M, N);
        freeMatrix(&A);
        freeMatrix(&B);
        freeMatrix(&C);
    }
    
    freeMatrix(&localA);
    freeMatrix(&localB);
    freeMatrix(&localC);

    MPI_Finalize();
    return 0;
}
