#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm> 
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define DBG 1
#define SHOWTIME 0

#define NUMBERS_BIG 100000000 //2000000
#define NUMBERS_DBG 128

#define MAX_BIG 100000 //1000000
#define MAX_DBG 99

#define NUMBERS ((DBG == 1) ? NUMBERS_DBG : NUMBERS_BIG)
#define MAX_NUMBER ((DBG == 1) ? MAX_DBG : MAX_BIG)

/* cuda errors */
bool checkForError(const cudaError_t cudaStatus, const char text[], int* dev_input, int* dev_tmp) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\n%s \nError code: %d \nStatus: %s \n\n", text, cudaStatus, cudaGetErrorString(cudaStatus));
		if (dev_input != NULL) {
			cudaFree(dev_input);
		}
		if (dev_tmp != NULL) {
			cudaFree(dev_tmp);
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
void printArray(int* A, int size) {
	printf("\n");
	for (int i = 0; i < size; i++) {
		printf("%d, ", A[i]);
	}
	printf("\n");
	fflush(stdout);
}

void printTime(time_t t1, time_t t2, const char* solutionType) {
	printf("\nTime in seconds (mergesort %s): %f", solutionType, difftime(t2, t1));
}

void checkIfCorrectlySorted(int* arr) {
	bool correct = true;
	for (int i = 0; i < NUMBERS - 1; i++) {
		if (arr[i] > arr[i + 1]) {
			printf("\n\n-----------ERROR!-----------%d\n\n ", i);
			correct = false;
			break;
		}
	}
	if (correct) {
		printf("\n----------- OK ------------");
	}
}

/* merge sort */
void fillArrayWithNumbers(int* numbers) {
	int i;
	srand(time(NULL));
	for (i = 0; i < NUMBERS; i++) {
		numbers[i] = rand() % MAX_NUMBER;
	}

	if (DBG) {
		printArray(numbers, NUMBERS);
	}
}

__host__
__device__
int getMid(int start, int end) {
	return start + (end - start) / 2;
}

__host__
__device__
void merge(int* arr, int* tmp, int leftStart, int rightEnd, int mid) {
	int i, j, k;
	int leftHalfSize = mid - leftStart + 1;
	int rightHalfSize = rightEnd - mid;

	/* create temp arrays */
	int* L = &tmp[leftStart];
	int* R = &tmp[mid + 1];

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

/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
__global__
void mergeKernel(int* arr, int* tmp, int vectorLengthPerThread, int vectorLength) {
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int leftStart = threadId * vectorLengthPerThread;
	int rightEnd = leftStart + vectorLengthPerThread - 1;
	int mid = getMid(leftStart, rightEnd);
	
	if (leftStart < vectorLength) {
		printf("\n thread: %d, <%d, %d>, mid %d", threadId, leftStart, rightEnd, mid);
		merge(arr, tmp, leftStart, rightEnd, mid);
	}
}


int main() {
	const int vectorLength = NUMBERS;
	const int threadsPerBlock = 64; // FIXME // 128
	int vectorLengthPerThread = 2; // FIXME
	const int vectorMultiplier = 2; // FIXME
	const int numBlocks = ceil(vectorLength / threadsPerBlock);
	const int blockVectorLength = vectorLength / numBlocks;
	const int vectorSizeInBytes = NUMBERS * sizeof(int);
	
	printf("\nConfiguration: vector length: %d, threads per block: %d, vector length per thread: %d, num blocks: %d",
		NUMBERS, threadsPerBlock, vectorLengthPerThread, numBlocks);

	int* tmp = (int*)malloc(vectorSizeInBytes);
	int* vector = (int*)malloc(vectorSizeInBytes);

	fillArrayWithNumbers(vector);

	int* dev_input = NULL;
	int* dev_tmp = NULL;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (checkForError(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?", dev_input, dev_tmp)) {
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_input, vectorSizeInBytes);
	if (checkForError(cudaStatus, "cudaMalloc (dev_input) failed!", dev_input, dev_tmp)) {
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_tmp, vectorSizeInBytes);
	if (checkForError(cudaStatus, "cudaMalloc (dev_tmp) failed!", dev_input, dev_tmp)) {
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_input, vector, vectorSizeInBytes, cudaMemcpyHostToDevice);
	if (checkForError(cudaStatus, "cudaMemcpy (vector -> dev_input) failed!", dev_input, dev_tmp)) {
		return cudaStatus;
	}

	for (int i = 0; vectorLengthPerThread < blockVectorLength; i++,vectorLengthPerThread *= vectorMultiplier)  {		
		printf("\nIter: %d, vector length per thread: %d", i, vectorLengthPerThread);
		
		mergeKernel<<<numBlocks, threadsPerBlock>>>(dev_input, dev_tmp, vectorLengthPerThread, vectorLength);
		
		cudaStatus = cudaGetLastError();
		if (checkForError(cudaStatus, "mergeSortGlobal launch failed!", dev_input, dev_tmp)) {
			return cudaStatus;
		}		
		cudaStatus = cudaDeviceSynchronize();
		if (checkForError(cudaStatus, "cudaDeviceSynchronize on \"mergeSortGlobal\" returned error code.", dev_input, dev_tmp)) {
			return cudaStatus;
		}
		
		if (DBG) {
			cudaStatus = cudaMemcpy(vector, dev_input, vectorSizeInBytes, cudaMemcpyDeviceToHost);
			if (checkForError(cudaStatus, "cudaMemcpy (dev_input -> vector) failed!")) {
				return cudaStatus;
			}
			printArray(vector, vectorLength);
		}
	}

	if (!DBG) {
		cudaStatus = cudaMemcpy(vector, dev_input, vectorSizeInBytes, cudaMemcpyDeviceToHost);
		if (checkForError(cudaStatus, "cudaMemcpy (dev_input -> vector) failed!")) {
			return cudaStatus;
		}
	}
	
	merge(vector, tmp, 0, NUMBERS - 1, getMid(0, NUMBERS - 1));

	cudaFree(dev_input);
	cudaFree(dev_tmp);
	cudaStatus = cudaDeviceReset();
	if (checkForError(cudaStatus, "cudaDeviceReset failed!")) {
		return 1;
	}

	if (DBG) {
		printArray(vector, NUMBERS);
	}
	fflush(stdout);
	checkIfCorrectlySorted(vector);

	return 0;
}