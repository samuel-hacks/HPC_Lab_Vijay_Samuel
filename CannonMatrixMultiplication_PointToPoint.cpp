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
    if (!mat || !*mat) return -1;
    free(&((*mat)[0][0]));
    free(*mat);
    return 0;
}

// Generate a random square matrix
void generate_matrix(int** mat, int dim) {
    // Seed with a value that is the same for all processes initially
    // but allow rank 0 to control the true seed time.
    srand(1);
    if (mat) { // Ensure matrix is not NULL
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                mat[i][j] = rand() % 10; // Random values 0-9
            }
        }
    }
}

// Serial matrix multiplication: C = A*B
void serial_matmul(int** A, int** B, int** C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < K; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
}

// Validate C_mpi against serial multiplication result
int validate_result(int** A, int** B, int** C_mpi, int M, int K, int N) {
    int** C_expected = NULL;
    if (allocMatrix(&C_expected, M, N) != 0) {
        printf("[ERROR] Allocation failed in validation\n");
        return 0;
    }

    serial_matmul(A, B, C_expected, M, K, N);

    int valid = 1;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (C_expected[i][j] != C_mpi[i][j]) {
                printf("[Validation Failed] C[%d][%d]: expected %d, got %d\n", i, j, C_expected[i][j], C_mpi[i][j]);
                valid = 0;
                break;
            }
        }
        if (!valid) break;
    }

    freeMatrix(&C_expected);
    return valid;
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

    int rows = 0;
    int procDim = 0, blockDim = 0;
    int **A = NULL, **B = NULL, **C = NULL;
    int **localA = NULL, **localB = NULL, **localC = NULL;
    MPI_Comm cartComm;
    int dim[2], period[2], reorder = 1;
    int coords[2];

    if (rank == 0) {
        if (argc < 2) {
            fprintf(stderr, "Usage: mpirun -np <num_procs> %s <matrix_order>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        rows = atoi(argv[1]);
        if (rows <= 0) {
            fprintf(stderr, "[ERROR] Matrix order must be a positive integer.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double root = sqrt(worldSize);
        if ((root - floor(root)) != 0) {
            fprintf(stderr, "[ERROR] Number of processes must be a perfect square!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        procDim = (int)root;
        if (rows % procDim != 0) {
            fprintf(stderr, "[ERROR] Matrix size not divisible by sqrt(processes)\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        blockDim = rows / procDim;

        if (allocMatrix(&A, rows, rows) != 0 || allocMatrix(&B, rows, rows) != 0 || allocMatrix(&C, rows, rows) != 0) {
            fprintf(stderr, "[ERROR] Allocation failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        srand(time(NULL));
        generate_matrix(A, rows);
        generate_matrix(B, rows);

        if (rows < 10) {
            printf("Matrix A:\n");
            for(int i=0; i<rows; i++) { for(int j=0; j<rows; j++) printf("%d ", A[i][j]); printf("\n"); }
            printf("\nMatrix B:\n");
            for(int i=0; i<rows; i++) { for(int j=0; j<rows; j++) printf("%d ", B[i][j]); printf("\n"); }
        }
    }

    int meta[2];
    if (rank == 0) {
        meta[0] = procDim;
        meta[1] = blockDim;
    }
    MPI_Bcast(meta, 2, MPI_INT, 0, MPI_COMM_WORLD);
    procDim = meta[0];
    blockDim = meta[1];
    rows = procDim * blockDim;

    dim[0] = procDim; dim[1] = procDim;
    period[0] = 1; period[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartComm);
    MPI_Cart_coords(cartComm, rank, 2, coords);

    if (allocMatrix(&localA, blockDim, blockDim) != 0 ||
        allocMatrix(&localB, blockDim, blockDim) != 0 ||
        allocMatrix(&localC, blockDim, blockDim) != 0) {
        fprintf(stderr, "[ERROR] Local allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // --- Data distribution using point-to-point communication ---
    #define A_TAG 1
    #define B_TAG 2

    // Create a datatype to describe a non-contiguous block in the full matrix
    MPI_Datatype blocktype;
    MPI_Type_vector(blockDim, blockDim, rows, MPI_INT, &blocktype);
    MPI_Type_commit(&blocktype);

    if (rank == 0) {
        // Rank 0 sends the appropriate block to every other process
        for (int i = 0; i < worldSize; i++) {
            if (i == 0) continue; // Skip sending to self
            int p_coords[2];
            MPI_Cart_coords(cartComm, i, 2, p_coords);
            int* A_start_ptr = &(A[p_coords[0] * blockDim][p_coords[1] * blockDim]);
            int* B_start_ptr = &(B[p_coords[0] * blockDim][p_coords[1] * blockDim]);

            MPI_Send(A_start_ptr, 1, blocktype, i, A_TAG, cartComm);
            MPI_Send(B_start_ptr, 1, blocktype, i, B_TAG, cartComm);
        }
        // Rank 0 copies its own block directly
        for (int i = 0; i < blockDim; i++) {
            for (int j = 0; j < blockDim; j++) {
                localA[i][j] = A[i][j];
                localB[i][j] = B[i][j];
            }
        }
    } else {
        // All other ranks receive their blocks from rank 0
        MPI_Recv(&(localA[0][0]), blockDim * blockDim, MPI_INT, 0, A_TAG, cartComm, MPI_STATUS_IGNORE);
        MPI_Recv(&(localB[0][0]), blockDim * blockDim, MPI_INT, 0, B_TAG, cartComm, MPI_STATUS_IGNORE);
    }

    MPI_Type_free(&blocktype); // Free the datatype after use
    
    for (int i = 0; i < blockDim; i++)
        for (int j = 0; j < blockDim; j++)
            localC[i][j] = 0;

    double start_time, end_time;
    MPI_Barrier(cartComm); 
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    int shift_source, shift_dest;
    MPI_Cart_shift(cartComm, 1, -coords[0], &shift_source, &shift_dest);
    MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 10, shift_source, 10, cartComm, MPI_STATUS_IGNORE);

    MPI_Cart_shift(cartComm, 0, -coords[1], &shift_source, &shift_dest);
    MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 20, shift_source, 20, cartComm, MPI_STATUS_IGNORE);

    for (int step = 0; step < procDim; step++) {
        matrixMultiply(localA, localB, blockDim, localC);

        MPI_Cart_shift(cartComm, 1, -1, &shift_source, &shift_dest);
        MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 30, shift_source, 30, cartComm, MPI_STATUS_IGNORE);

        MPI_Cart_shift(cartComm, 0, -1, &shift_source, &shift_dest);
        MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 40, shift_source, 40, cartComm, MPI_STATUS_IGNORE);
    }
    
    MPI_Barrier(cartComm); 
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Parallel calculation took %f seconds.\n", end_time - start_time);
    }

    // --- Data gathering using point-to-point communication ---
    #define C_TAG 3

    // Create a datatype for the destination blocks on rank 0
    MPI_Datatype gathertype;
    MPI_Type_vector(blockDim, blockDim, rows, MPI_INT, &gathertype);
    MPI_Type_commit(&gathertype);

    if (rank == 0) {
        // Rank 0 receives the result block from every other process
        for (int i = 0; i < worldSize; i++) {
            if (i == 0) continue; // Skip receiving from self
            int p_coords[2];
            MPI_Cart_coords(cartComm, i, 2, p_coords);
            int* C_dest_ptr = &(C[p_coords[0] * blockDim][p_coords[1] * blockDim]);
            
            MPI_Recv(C_dest_ptr, 1, gathertype, i, C_TAG, cartComm, MPI_STATUS_IGNORE);
        }
        // Rank 0 copies its own result block directly
        for (int i = 0; i < blockDim; i++) {
            for (int j = 0; j < blockDim; j++) {
                C[i][j] = localC[i][j];
            }
        }
    } else {
        // All other ranks send their local result to rank 0
        MPI_Send(&(localC[0][0]), blockDim * blockDim, MPI_INT, 0, C_TAG, cartComm);
    }

    MPI_Type_free(&gathertype);

    if (rank == 0) {
        if (rows < 10) {
            printf("\nResult matrix C:\n");
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < rows; j++) {
                    printf("%d ", C[i][j]);
                }
                printf("\n");
            }
        }

        int valid = validate_result(A, B, C, rows, rows, rows);
        if (valid)
            printf("[SUCCESS] Result validated correctly!\n");
        else
            printf("[FAILURE] Result validation failed!\n");

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
