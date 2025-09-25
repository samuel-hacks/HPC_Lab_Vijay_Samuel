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

        // Seed random number generator on rank 0 only
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
    rows = procDim * blockDim; // Recalculate rows for all processes

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

    MPI_Datatype blocktype, blocktype_resized;
    MPI_Type_vector(blockDim, blockDim, rows, MPI_INT, &blocktype);
    MPI_Type_create_resized(blocktype, 0, sizeof(int), &blocktype_resized);
    MPI_Type_commit(&blocktype_resized);

    int* sendcounts = (int*)malloc(worldSize * sizeof(int));
    int* displs = (int*)malloc(worldSize * sizeof(int));

    if (rank == 0) {
        for (int i = 0; i < worldSize; i++) {
            sendcounts[i] = 1;
            int p_coords[2];
            MPI_Cart_coords(cartComm, i, 2, p_coords);
            displs[i] = (p_coords[0] * rows * blockDim) + (p_coords[1] * blockDim);
        }
    }

    int* A_ptr = (rank == 0) ? &(A[0][0]) : NULL;
    int* B_ptr = (rank == 0) ? &(B[0][0]) : NULL;
    MPI_Scatterv(A_ptr, sendcounts, displs, blocktype_resized, &(localA[0][0]), blockDim * blockDim, MPI_INT, 0, cartComm);
    MPI_Scatterv(B_ptr, sendcounts, displs, blocktype_resized, &(localB[0][0]), blockDim * blockDim, MPI_INT, 0, cartComm);

    free(sendcounts);
    free(displs);
    MPI_Type_free(&blocktype);
    MPI_Type_free(&blocktype_resized);

    for (int i = 0; i < blockDim; i++)
        for (int j = 0; j < blockDim; j++)
            localC[i][j] = 0;

    // ----------- TIMING CODE START -----------
    double start_time, end_time;
    MPI_Barrier(cartComm); // Synchronize all processes before starting the timer
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    // ----------- TIMING CODE START -----------

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
    
    // ----------- TIMING CODE END -----------
    MPI_Barrier(cartComm); // Synchronize all processes before stopping the timer
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Parallel calculation took %f seconds.\n", end_time - start_time);
    }
    // ----------- TIMING CODE END -----------


    MPI_Datatype gathertype, gathertype_resized;
    MPI_Type_vector(blockDim, blockDim, rows, MPI_INT, &gathertype);
    MPI_Type_create_resized(gathertype, 0, sizeof(int), &gathertype_resized);
    MPI_Type_commit(&gathertype_resized);

    int* recvcounts = (int*)malloc(worldSize * sizeof(int));
    int* recvdispls = (int*)malloc(worldSize * sizeof(int));

    if (rank == 0) {
        for (int i = 0; i < worldSize; i++) {
            recvcounts[i] = 1;
            int p_coords[2];
            MPI_Cart_coords(cartComm, i, 2, p_coords);
            recvdispls[i] = (p_coords[0] * rows * blockDim) + (p_coords[1] * blockDim);
        }
    }

    int* C_ptr = (rank == 0) ? &(C[0][0]) : NULL;
    MPI_Gatherv(&(localC[0][0]), blockDim * blockDim, MPI_INT, C_ptr, recvcounts, recvdispls, gathertype_resized, 0, cartComm);

    free(recvcounts);
    free(recvdispls);
    MPI_Type_free(&gathertype);
    MPI_Type_free(&gathertype_resized);

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
