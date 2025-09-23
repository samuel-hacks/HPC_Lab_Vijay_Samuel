#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

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
    free(&((*mat)[0][0]));
    free(*mat);
    return 0;
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
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            if (C_expected[i][j] != C_mpi[i][j]) {
                printf("[Validation Failed] C[%d][%d]: expected %d, got %d\n", i, j, C_expected[i][j], C_mpi[i][j]);
                valid = 0;
                break;
            }
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

    int rows = 0, columns = 0;
    int procDim = 0, blockDim = 0;
    int **A = NULL, **B = NULL, **C = NULL;         // Global matrices on rank 0
    int **localA = NULL, **localB = NULL, **localC = NULL; // Local blocks
    MPI_Comm cartComm;
    int dim[2], period[2], reorder = 1;
    int coords[2];

    if (rank == 0) {
        FILE* fp = fopen("A.txt", "r");
        if (!fp) {
            printf("[ERROR] Cannot open A.txt\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Count rows and columns
        int n, count = 0;
        char ch;
        rows = 0;
        while (fscanf(fp, "%d", &n) != EOF) {
            count++;
            ch = fgetc(fp);
            if (ch == '\n' || ch == EOF) rows++;
        }
        columns = count / rows;
        if (rows != columns) {
            printf("[ERROR] Only square matrices supported!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        double root = sqrt(worldSize);
        if ((root - floor(root)) != 0) {
            printf("[ERROR] Number of processes must be perfect square!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        procDim = (int)root;
        if (rows % procDim != 0) {
            printf("[ERROR] Matrix size not divisible by sqrt(processes)\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        blockDim = rows / procDim;

        fseek(fp, 0, SEEK_SET);
        if (allocMatrix(&A, rows, columns) != 0 || allocMatrix(&B, rows, columns) != 0 || allocMatrix(&C, rows, columns) != 0) {
            printf("[ERROR] Allocation failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Read matrix A
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                fscanf(fp, "%d", &A[i][j]);
        fclose(fp);

        fp = fopen("B.txt", "r");
        if (!fp) {
            printf("[ERROR] Cannot open B.txt\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                fscanf(fp, "%d", &B[i][j]);
        fclose(fp);

        // Initialize C to zero
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                C[i][j] = 0;
    }

    // Broadcast matrix info to all
    int meta[3]; // procDim, blockDim, rows (rows=columns)
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

    // Create Cartesian communicator
    dim[0] = procDim; dim[1] = procDim;
    period[0] = 1; period[1] = 1; // Periodic for Cannon's algorithm
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartComm);
    MPI_Cart_coords(cartComm, rank, 2, coords);

    // Allocate local blocks
    if (allocMatrix(&localA, blockDim, blockDim) != 0 ||
        allocMatrix(&localB, blockDim, blockDim) != 0 ||
        allocMatrix(&localC, blockDim, blockDim) != 0) {
        printf("[ERROR] Local allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --------- Manual Scatter of A and B -----------

    if (rank == 0) {
        // Send blocks to other processes
        for (int p = 0; p < worldSize; p++) {
            int p_coords[2];
            MPI_Cart_coords(cartComm, p, 2, p_coords);
            int row_start = p_coords[0] * blockDim;
            int col_start = p_coords[1] * blockDim;

            // Prepare block buffer
            int* blockA = (int*)malloc(blockDim * blockDim * sizeof(int));
            int* blockB = (int*)malloc(blockDim * blockDim * sizeof(int));

            for (int i = 0; i < blockDim; i++)
                for (int j = 0; j < blockDim; j++) {
                    blockA[i * blockDim + j] = A[row_start + i][col_start + j];
                    blockB[i * blockDim + j] = B[row_start + i][col_start + j];
                }

            if (p == 0) {
                // Copy directly to local blocks on rank 0
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
        // Receive local blocks
        int* blockA = (int*)malloc(blockDim * blockDim * sizeof(int));
        int* blockB = (int*)malloc(blockDim * blockDim * sizeof(int));

        MPI_Recv(blockA, blockDim * blockDim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(blockB, blockDim * blockDim, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Copy to local matrices
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

    // -------- Initial skewing for Cannon's algorithm ------------

    // Shift A left by row coordinate times
    int shift_source, shift_dest;
    MPI_Cart_shift(cartComm, 1, -coords[0], &shift_source, &shift_dest);
    MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 10, shift_source, 10, cartComm, MPI_STATUS_IGNORE);

    // Shift B up by column coordinate times
    MPI_Cart_shift(cartComm, 0, -coords[1], &shift_source, &shift_dest);
    MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 20, shift_source, 20, cartComm, MPI_STATUS_IGNORE);

    // --------- Cannon's algorithm main loop -----------
    for (int step = 0; step < procDim; step++) {
        // Multiply local blocks and accumulate in localC
        matrixMultiply(localA, localB, blockDim, localC);

        // Shift A left by 1
        MPI_Cart_shift(cartComm, 1, -1, &shift_source, &shift_dest);
        MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 30, shift_source, 30, cartComm, MPI_STATUS_IGNORE);

        // Shift B up by 1
        MPI_Cart_shift(cartComm, 0, -1, &shift_source, &shift_dest);
        MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, shift_dest, 40, shift_source, 40, cartComm, MPI_STATUS_IGNORE);
    }

    // -------- Manual gather of localC blocks to rank 0 ---------

    if (rank == 0) {
        // Copy own localC into global C
        for (int i = 0; i < blockDim; i++)
            for (int j = 0; j < blockDim; j++) {
                C[i][j] = localC[i][j];
            }

        // Receive other blocks
        for (int p = 1; p < worldSize; p++) {
            int p_coords[2];
            MPI_Cart_coords(cartComm, p, 2, p_coords);
            int row_start = p_coords[0] * blockDim;
            int col_start = p_coords[1] * blockDim;

            int* blockC = (int*)malloc(blockDim * blockDim * sizeof(int));
            MPI_Recv(blockC, blockDim * blockDim, MPI_INT, p, 50, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < blockDim; i++)
                for (int j = 0; j < blockDim; j++) {
                    C[row_start + i][col_start + j] = blockC[i * blockDim + j];
                }

            free(blockC);
        }
    }
    else {
        // Send localC block to rank 0
        int* blockC = (int*)malloc(blockDim * blockDim * sizeof(int));
        for (int i = 0; i < blockDim; i++)
            for (int j = 0; j < blockDim; j++) {
                blockC[i * blockDim + j] = localC[i][j];
            }
        MPI_Send(blockC, blockDim * blockDim, MPI_INT, 0, 50, MPI_COMM_WORLD);
        free(blockC);
    }

    // Rank 0 prints and validates the result
    if (rank == 0) {
        printf("Result matrix C:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                printf("%d ", C[i][j]);
            }
            printf("\n");
        }

        int valid = validate_result(A, B, C, rows, columns, columns);
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
