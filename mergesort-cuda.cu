#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm> 
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define HARD_DBG 0
#define DBG 1
#define SHOWTIME 0

#define NUMBERS_BIG 4194304*2//4 194 304 //100 000 000 //2000000
#define NUMBERS_DBG 4194304 //128

#define MAX_BIG 100000 //1000000
#define MAX_DBG 999

#define NUMBERS ((DBG == 1) ? NUMBERS_DBG : NUMBERS_BIG)
#define MAX_NUMBER ((DBG == 1) ? MAX_DBG : MAX_BIG)
#define THREADS_PER_BLOCK 512
#define VECTOR_LENGTH_PER_THREAD 2
#define VECTOR_MULTIPLIER 2

/* cuda errors */
bool checkForError(const cudaError_t cudaStatus, const char text[], short* dev_input, short* dev_tmp=NULL) {
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
void printArray(short* A, int size) {
	printf("\n");
	for (int i = 0; i < size; i++) {
		printf("%d, ", A[i]);
	}
	printf("\n");
	fflush(stdout);
}

void checkIfCorrectlySorted(short* arr) {
	for (int i = 0; i < NUMBERS - 1; i++) {
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
	for (i = 0; i < NUMBERS; i++) {
		numbers[i] = rand() % MAX_NUMBER;
	}

	if (HARD_DBG) {
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
		if (HARD_DBG) {
			printf("\n thread: %d, <%d, %d>, mid %d", threadId, leftStart, rightEnd, mid);
		}
		merge(arr, leftStart, rightEnd, mid, tmpIndexStart);
	}
}


void mergeSort(short* arr,int leftStart, int rightEnd, int minVectorLength, int tmpIndexStart) {
	if (leftStart < rightEnd && rightEnd - leftStart > minVectorLength) {
		if (HARD_DBG) {
			printf("\n<%d,%d> minVec: %d", leftStart, rightEnd, minVectorLength);
		}
		int m = getMid(leftStart, rightEnd);
		mergeSort(arr, leftStart, m, minVectorLength, tmpIndexStart);
		mergeSort(arr, m + 1, rightEnd, minVectorLength, tmpIndexStart);
		merge(arr, leftStart, rightEnd, m, tmpIndexStart);
	}
}

int main() {
	const int vectorSizeInBytes = NUMBERS * sizeof(short) * 2;
	int tmpIndexStart = NUMBERS;
	//short* tmp = (short*)malloc(vectorSizeInBytes);
	short* vector = (short*)malloc(vectorSizeInBytes);

	fillArrayWithNumbers(vector);

	short* dev_input = NULL;
	//short* dev_tmp = NULL;
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
	
	const int vectorLength = NUMBERS;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int vectorLengthPerThread = VECTOR_LENGTH_PER_THREAD;
	const int vectorMultiplier = VECTOR_MULTIPLIER;
	int numBlocks = ceil(vectorLength / threadsPerBlock);
	const int blockVectorLength = vectorLength / numBlocks;
	
	printf("\nConfiguration: vector length: %d, threads per block: %d, vector length per thread: %d, num blocks: %d, block vector length: %d\n",
		NUMBERS, threadsPerBlock, vectorLengthPerThread, numBlocks, blockVectorLength);

	int i = 0;
	while (vectorLengthPerThread <= blockVectorLength) {		
		if (DBG) {
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
		
		if (HARD_DBG) {
			cudaStatus = cudaMemcpy(vector, dev_input, vectorSizeInBytes, cudaMemcpyDeviceToHost);
			if (checkForError(cudaStatus, "cudaMemcpy (dev_input -> vector) failed!")) {
				return cudaStatus;
			}
			printArray(vector, vectorLength);
		}
	}	

	if (!HARD_DBG) {
		cudaStatus = cudaMemcpy(vector, dev_input, vectorSizeInBytes, cudaMemcpyDeviceToHost);
		if (checkForError(cudaStatus, "cudaMemcpy (dev_input -> vector) failed!")) {
			return cudaStatus;
		}
	}
	
	mergeSort(vector, 0, NUMBERS - 1, blockVectorLength, tmpIndexStart);
	if (HARD_DBG) {
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