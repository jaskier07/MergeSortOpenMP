#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm> 
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PROGRAM_STATE 2

#define HARD_DBG 2
#define DBG 1

#define VECTOR_SIZE_STD 4194304*2 ///100 000 000
#define VECTOR_SIZE_DBG 128 // 128

#define VECTOR_MAX_NUMBER_STD 999 //1000000
#define VECTOR_MAX_NUMBER_DBG 100 // 100

#define THREADS_PER_BLOCK_STD 512
#define THREADS_PER_BLOCK_DBG 32 // 32

#define VECTOR_LENGTH_PER_THREAD_STD 32
#define VECTOR_LENGTH_PER_THREAD_DBG 2 //2

#define VECTOR_SIZE ((PROGRAM_STATE >= DBG) ? VECTOR_SIZE_DBG : VECTOR_SIZE_STD)
#define MAX_NUMBER ((PROGRAM_STATE >= DBG) ? VECTOR_MAX_NUMBER_DBG : VECTOR_MAX_NUMBER_STD)
#define THREADS_PER_BLOCK ((PROGRAM_STATE >= DBG) ? THREADS_PER_BLOCK_DBG : THREADS_PER_BLOCK_STD)
#define VECTOR_LENGTH_PER_THREAD ((PROGRAM_STATE >= DBG) ? VECTOR_LENGTH_PER_THREAD_DBG : VECTOR_LENGTH_PER_THREAD_STD)

/* cuda errors */
bool checkForError(const cudaError_t cudaStatus, const char text[], short* dev_input) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\n%s \nError code: %d \nStatus: %s \n\n", text, cudaStatus, cudaGetErrorString(cudaStatus));
		if (dev_input != NULL) {
			cudaFree(dev_input);
		}
		return true;
	}
	return false;
}

bool checkForError(const cudaError_t cudaStatus, const char text[]) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\n%s \nError code: %d \nStatus: %s \n\n", text, cudaStatus, cudaGetErrorString(cudaStatus));
		return true;
	}
	return false;
}

/* info */
void printArray(short* A, int size) {
	printf("\n");
	for (int i = 0; i < size; i++) {
		printf("%d, ", A[i]);
	}
	printf("\n");
	fflush(stdout);
}

void checkIfCorrectlySorted(short* arr) {
	for (int i = 0; i < VECTOR_SIZE - 1; i++) {
		if (arr[i] > arr[i + 1]) {
			printf("\n\n-----------ERROR!-----------%d\n\n ", i);
			return;
		}
	}
	printf("\n----------- OK ------------");
}

/* merge sort */
void fillArrayWithNumbers(short* numbers) {
	int i;
	srand(time(NULL));
	for (i = 0; i < VECTOR_SIZE; i++) {
		numbers[i] = rand() % MAX_NUMBER;
	}

	if (PROGRAM_STATE >= HARD_DBG) {
		printArray(numbers, VECTOR_SIZE);
	}
}

__host__
__device__
int getMid(int start, int end) {
	return start + (end - start) / 2;
}

__host__
__device__
void merge(short* arr, int leftStart, int rightEnd, int mid, int tmpIndexStart) {
	int i, j, k;
	int leftHalfSize = mid - leftStart + 1;
	int rightHalfSize = rightEnd - mid;

	short* L = &arr[tmpIndexStart + leftStart];
	short* R = &arr[tmpIndexStart + mid + 1];
	/* Copy data to temp arrays L[] and R[] */
	for (i = 0; i < leftHalfSize; i++) {
		L[i] = arr[leftStart + i];
	}
	for (j = 0; j < rightHalfSize; j++) {
		R[j] = arr[mid + 1 + j];
	}
	/* Merge the temp arrays back into arr[l..r]*/
	i = 0;
	j = 0;
	k = leftStart;
	while (i < leftHalfSize && j < rightHalfSize) {
		if (L[i] <= R[j]) {
			arr[k] = L[i];
			i++;
		}
		else {
			arr[k] = R[j];
			j++;
		}
		k++;
	}
	/* Copy the remaining elements of L[], if there are any */
	while (i < leftHalfSize) {
		arr[k] = L[i];
		i++;
		k++;
	}

	/* Copy the remaining elements of R[], if there are any */
	while (j < rightHalfSize) {
		arr[k] = R[j];
		j++;
		k++;
	}	
}

__global__
void mergeKernel(short* arr, int vectorLengthPerThread, int vectorLength, int tmpIndexStart) {
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int leftStart = threadId * vectorLengthPerThread;
	int rightEnd = leftStart + vectorLengthPerThread - 1;
	int mid = getMid(leftStart, rightEnd);
	
	if (leftStart < vectorLength) {
		if (PROGRAM_STATE >= HARD_DBG) {
			printf("\n thread: %d, <%d, %d>, mid %d", threadId, leftStart, rightEnd, mid);
		}
		merge(arr, leftStart, rightEnd, mid, tmpIndexStart);
	}
}


void mergeSort(short* arr,int leftStart, int rightEnd, int minVectorLength, int tmpIndexStart) {
	if (leftStart < rightEnd && rightEnd - leftStart >= minVectorLength) {
		if (PROGRAM_STATE >= HARD_DBG) {
			printf("\n<%d,%d> minVec: %d", leftStart, rightEnd, minVectorLength);
		}
		int m = getMid(leftStart, rightEnd);
		mergeSort(arr, leftStart, m, minVectorLength, tmpIndexStart);
		mergeSort(arr, m + 1, rightEnd, minVectorLength, tmpIndexStart);
		merge(arr, leftStart, rightEnd, m, tmpIndexStart);
	}
}

int main() {
	const int vectorMultiplier = 2;
	const int vectorLength = VECTOR_SIZE;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int vectorLengthPerThread = VECTOR_LENGTH_PER_THREAD;
	int numBlocks = ceil(vectorLength / threadsPerBlock);
	const int blockVectorLength = vectorLength / numBlocks;	
	const int vectorSizeInBytes = vectorLength * sizeof(short) * 2;
	int tmpIndexStart = vectorLength;
	
	short* vector = (short*)malloc(vectorSizeInBytes);
	fillArrayWithNumbers(vector);
	short* dev_input = NULL;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (checkForError(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?", dev_input)) {
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_input, vectorSizeInBytes);
	if (checkForError(cudaStatus, "cudaMalloc (dev_input) failed!", dev_input)) {
		return cudaStatus;
	}
	
	cudaStatus = cudaMemcpy(dev_input, vector, vectorSizeInBytes, cudaMemcpyHostToDevice);
	if (checkForError(cudaStatus, "cudaMemcpy (vector -> dev_input) failed!", dev_input)) {
		return cudaStatus;
	}	
	
	printf("\nConfiguration: vector length: %d, threads per block: %d, vector length per thread: %d, num blocks: %d, block vector length: %d\n",
		vectorLength, threadsPerBlock, vectorLengthPerThread, numBlocks, blockVectorLength);

	int i = 0;
	while (vectorLengthPerThread <= blockVectorLength) {		
		if (PROGRAM_STATE >= DBG) {
			printf("\nIter: %d, vector length per thread: %d", i++, vectorLengthPerThread);
		}
		
		mergeKernel<<<numBlocks, threadsPerBlock>>>(dev_input, vectorLengthPerThread, vectorLength, tmpIndexStart);
		
		cudaStatus = cudaGetLastError();
		if (checkForError(cudaStatus, "mergeKernel launch failed!", dev_input)) {
			return cudaStatus;
		}		
		cudaStatus = cudaDeviceSynchronize();
		if (checkForError(cudaStatus, "cudaDeviceSynchronize on \"mergeKernel\" returned error code.", dev_input)) {
			return cudaStatus;
		}
		
		vectorLengthPerThread *= vectorMultiplier;
		
		if (PROGRAM_STATE >= HARD_DBG) {
			cudaStatus = cudaMemcpy(vector, dev_input, vectorSizeInBytes, cudaMemcpyDeviceToHost);
			if (checkForError(cudaStatus, "cudaMemcpy (dev_input -> vector) failed!")) {
				return cudaStatus;
			}
			printArray(vector, vectorLength);
		}
	}	

	if (PROGRAM_STATE < HARD_DBG) {
		cudaStatus = cudaMemcpy(vector, dev_input, vectorSizeInBytes, cudaMemcpyDeviceToHost);
		if (checkForError(cudaStatus, "cudaMemcpy (dev_input -> vector) failed!")) {
			return cudaStatus;
		}
	}
	
	mergeSort(vector, 0, vectorLength - 1, blockVectorLength, tmpIndexStart);
	if (PROGRAM_STATE >= HARD_DBG) {
		printArray(vector, vectorLength);
	}

	cudaFree(dev_input);
	cudaStatus = cudaDeviceReset();
	if (checkForError(cudaStatus, "cudaDeviceReset failed!")) {
		return 1;
	}

	fflush(stdout);
	checkIfCorrectlySorted(vector);

	return 0;
}