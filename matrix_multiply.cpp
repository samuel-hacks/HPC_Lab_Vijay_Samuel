#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

// Matrix allocation (contiguous)
int allocMatrix(int*** mat, int rows, int cols) {
    int* p = (int*)malloc(sizeof(int) * rows * cols);
    if (!p) return -1;

    *mat = (int**)malloc(rows * sizeof(int*));
    if (!*mat) {
        free(p);
        return -1;
    }
    for (int i = 0; i < rows; i++) {
        (*mat)[i] = &p[i * cols];
    }
    return 0;
}

int freeMatrix(int*** mat) {
    if (mat && *mat) {
        free(&((*mat)[0][0]));
        free(*mat);
        *mat = NULL;
    }
    return 0;
}

// Multiply local blocks: C_local += A_local * B_local
void matrixMultiply(int** A, int** B, int blockDim, int** C) {
    for (int i = 0; i < blockDim; i++)
        for (int j = 0; j < blockDim; j++) {
            int sum = 0;
            for (int k = 0; k < blockDim; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] += sum;
        }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s <matrix_size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int rows = atoi(argv[1]);
    int columns = rows;

    // Check if worldSize is perfect square and divides rows evenly
    double root = sqrt(worldSize);
    if ((root - floor(root)) != 0) {
        if (rank == 0) printf("[ERROR] Number of processes must be perfect square!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int procDim = (int)root;

    if (rows % procDim != 0) {
        if (rank == 0) printf("[ERROR] Matrix size must be divisible by sqrt(number of processes)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int blockDim = rows / procDim;

    // Broadcast matrix size info to all
    int meta[3];
    if (rank == 0) {
        meta[0] = procDim;
        meta[1] = blockDim;
        meta[2] = rows;
    }
    MPI_Bcast(meta, 3, MPI_INT, 0, MPI_COMM_WORLD);
    procDim = meta[0];
    blockDim = meta[1];
    rows = meta[2];
    columns = rows;

    int **A = NULL, **B = NULL, **C = NULL;         // Global matrices on rank 0
    int **localA = NULL, **localB = NULL, **localC = NULL; // Local blocks

    // Create Cartesian communicator
    MPI_Comm cartComm;
    int dims[2] = {procDim, procDim};
    int periods[2] = {1, 1}; // Periodic for Cannon's algorithm
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartComm);

    int coords[2];
    MPI_Cart_coords(cartComm, rank, 2, coords);

    // Allocate local blocks
    if (allocMatrix(&localA, blockDim, blockDim) != 0 ||
        allocMatrix(&localB, blockDim, blockDim) != 0 ||
        allocMatrix(&localC, blockDim, blockDim) != 0) {
        printf("[ERROR] Local allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Rank 0 generates random global matrices A, B
    if (rank == 0) {
        srand(time(NULL));
        if (allocMatrix(&A, rows, columns) != 0 ||
            allocMatrix(&B, rows, columns) != 0 ||
            allocMatrix(&C, rows, columns) != 0) {
            printf("[ERROR] Allocation failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
                C[i][j] = 0;
            }
    }

    // Scatter A and B manually
    if (rank == 0) {
        for (int p = 0; p < worldSize; p++) {
            int p_coords[2];
            MPI_Cart_coords(cartComm, p, 2, p_coords);
            int row_start = p_coords[0] * blockDim;
            int col_start = p_coords[1] * blockDim;

            int* blockA = (int*)malloc(blockDim * blockDim * sizeof(int));
            int* blockB = (int*)malloc(blockDim * blockDim * sizeof(int));

            for (int i = 0; i < blockDim; i++)
                for (int j = 0; j < blockDim; j++) {
                    blockA[i * blockDim + j] = A[row_start + i][col_start + j];
                    blockB[i * blockDim + j] = B[row_start + i][col_start + j];
                }

            if (p == 0) {
                for (int i = 0; i < blockDim; i++)
                    for (int j = 0; j < blockDim; j++) {
                        localA[i][j] = blockA[i * blockDim + j];
                        localB[i][j] = blockB[i * blockDim + j];
                    }
            }
            else {
                MPI_Send(blockA, blockDim * blockDim, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(blockB, blockDim * blockDim, MPI_INT, p, 1, MPI_COMM_WORLD);
            }
            free(blockA);
            free(blockB);
        }
    }
    else {
        int* blockA = (int*)malloc(blockDim * blockDim * sizeof(int));
        int* blockB = (int*)malloc(blockDim * blockDim * sizeof(int));

        MPI_Recv(blockA, blockDim * blockDim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(blockB, blockDim * blockDim, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < blockDim; i++)
            for (int j = 0; j < blockDim; j++) {
                localA[i][j] = blockA[i * blockDim + j];
                localB[i][j] = blockB[i * blockDim + j];
            }

        free(blockA);
        free(blockB);
    }

    // Initialize localC to zero
    for (int i = 0; i < blockDim; i++)
        for (int j = 0; j < blockDim; j++)
            localC[i][j] = 0;

    MPI_Barrier(cartComm);
    double start_time = MPI_Wtime();

    // Initial skewing for Cannon's algorithm
    int shift_source, shift_dest;

    // Shift A left by row coordinate times
    MPI_Cart_shift(cartComm, 1, -coords[0], &shift_source, &shift_dest);
    MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 10, shift_source, 10, cartComm, MPI_STATUS_IGNORE);

    // Shift B up by column coordinate times
    MPI_Cart_shift(cartComm, 0, -coords[1], &shift_source, &shift_dest);
    MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 20, shift_source, 20, cartComm, MPI_STATUS_IGNORE);

    // Cannon's main loop
    for (int step = 0; step < procDim; step++) {
        matrixMultiply(localA, localB, blockDim, localC);

        MPI_Cart_shift(cartComm, 1, -1, &shift_source, &shift_dest);
        MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 30, shift_source, 30, cartComm, MPI_STATUS_IGNORE);

        MPI_Cart_shift(cartComm, 0, -1, &shift_source, &shift_dest);
        MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 40, shift_source, 40, cartComm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(cartComm);
    double end_time = MPI_Wtime();

    // Gather localC blocks back to rank 0 (optional, but we can skip this if we only want timing)
    // Here we just skip gather for timing-only runs

    if (rank == 0) {
        printf("Matrix Size: %dx%d, Processes: %d, Time taken: %f seconds\n", rows, columns, worldSize, end_time - start_time);
    }

    if (rank == 0) {
        freeMatrix(&A);
        freeMatrix(&B);
        freeMatrix(&C);
    }
    freeMatrix(&localA);
    freeMatrix(&localB);
    freeMatrix(&localC);

    MPI_Comm_free(&cartComm);
    MPI_Finalize();
    return 0;
}
