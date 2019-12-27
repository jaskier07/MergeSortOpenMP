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
#define NUMBERS_DBG 4096

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
__device__
void inlinePrintArray(int* A, int from, int to) {
	printf("[ "); 
	for (int i = from; i <= to; i++) {
		printf("%[%d]%d, ", i, A[i]);
	}
	printf(" ]");
}

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

/* debug */
__device__
void debugPrintMergeSort(int* arr, int left_start, int right_start, const char c[]) {
	//if (DBG) {
		printf("\n%c%c <%d, %d> ",  c[0], c[1], left_start, right_start);
		inlinePrintArray(arr, left_start, right_start);
		//fflush(stdout);
	//}
}

void checkIfCorrectlySorted(int* arr, int from, int to) {
	bool correct = true;
	for (int i = from; i < to; i++) {
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

/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
__device__
void merge(int arr[], int left_start, int mid, int right_start, int* tmp) {
	//if (DBG) {
		/*printf("\nmerge: [ ");
		inlinePrintArray(arr, left_start, mid);
		printf(" ] with [ ");
		inlinePrintArray(arr, mid, right_start);
		printf(" ]");
		fflush(stdout);*/
	//}

	int i, j, k;
	int left_half_size = mid - left_start + 1;
	int right_half_size = right_start - mid;

	/* create temp arrays */
	int* L = &tmp[left_start];
	int* R = &tmp[mid + 1];

	/* Copy data to temp arrays L[] and R[] */
	for (i = 0; i < left_half_size; i++) {
		L[i] = arr[left_start + i];
	}
	for (j = 0; j < right_half_size; j++) {
		R[j] = arr[mid + 1 + j];
	}

	/* Merge the temp arrays back into arr[l..r]*/
	i = 0;
	j = 0;
	k = left_start;
	while (i < left_half_size && j < right_half_size) {
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
	while (i < left_half_size) {
		arr[k] = L[i];
		i++;
		k++;
	}

	/* Copy the remaining elements of R[], if there are any */
	while (j < right_half_size) {
		arr[k] = R[j];
		j++;
		k++;
	}
}

__device__
void mergeSort(int* arr, int left_start, int right_start, int* tmp) {
	if (left_start < right_start)
	{
		int m = left_start + (right_start - left_start) / 2;

		//debugPrintMergeSort(arr, left_start, m, "l ");
		mergeSort(arr, left_start, m, tmp);

		//debugPrintMergeSort(arr, m + 1, right_start, "r ");
		mergeSort(arr, m + 1, right_start, tmp);

		merge(arr, left_start, m, right_start, tmp);
		//debugPrintMergeSort(arr, left_start, right_start, "m ");
	}
}

__global__
void mergeSortGlobal(int* arr, int* tmp, int vectorLengthPerThread) {
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int left_start = threadId * vectorLengthPerThread;
	int right_end = left_start + vectorLengthPerThread - 1;

	if (left_start < right_end)
	{
		// Same as (l+r)/2, but avoids overflow for large l and h 
		int m = left_start + (right_end - left_start) / 2;

		// Sort first and second halves 
		//debugPrintMergeSort(arr, left_start, m, "gl");
		mergeSort(arr, left_start, m, tmp);

		//debugPrintMergeSort(arr, m + 1, right_end, "gr");
		mergeSort(arr, m + 1, right_end, tmp);

		merge(arr, left_start, m, right_end, tmp);
		//debugPrintMergeSort(arr, left_start, right_end, "gm");
	}
}

int main() {
	const int threadsInBlock = 64; // 256
	const int vectorLengthPerThread = 64;
	const int numBlocks = ceil(NUMBERS / threadsInBlock / vectorLengthPerThread);
	printf("\nConfiguration: vector length: %d, threads per block: %d, vector length per thread: %d, num blocks: %d",
		NUMBERS, threadsInBlock, vectorLengthPerThread, numBlocks);

	const int vectorSizeInBytes = NUMBERS * sizeof(int);

	//time_t time1, time2;
	int* tmp = (int*)malloc(vectorSizeInBytes);
	int* vector = (int*)malloc(vectorSizeInBytes);

	//if (!SHOWTIME) {
		/*	int* numbersSeq = (int*)malloc(numbersSize);

			fillArrayWithNumbers(numbersSeq);
			memcpy(numbersPar, numbersSeq, numbersSize);

			time(&time1);
			mergeSortSequential(numbersSeq, 0, NUMBERS - 1, tmp);
			time(&time2);
			printTime(time1, time2, "sequential");

			free(tmp);
			tmp = (int*)malloc(numbersSize);
			*/
			/*}
			else {*/
	fillArrayWithNumbers(vector);
	//}

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

	mergeSortGlobal <<<numBlocks, threadsInBlock >>> (dev_input, dev_tmp, vectorLengthPerThread);
	cudaStatus = cudaGetLastError();
	if (checkForError(cudaStatus, "mergeSortGlobal launch failed!", dev_input, dev_tmp)) {
		return cudaStatus;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (checkForError(cudaStatus, "cudaDeviceSynchronize on \"mergeSortGlobal\" returned error code.", dev_input, dev_tmp)) {
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(vector, dev_input, vectorSizeInBytes, cudaMemcpyDeviceToHost);
	if (checkForError(cudaStatus, "cudaMemcpy (dev_input -> vector) failed!")) {
		return cudaStatus;
	}
	//printTime(time1, time2, "parallel");

	cudaFree(dev_input);
	cudaFree(dev_tmp);
	cudaStatus = cudaDeviceReset();
	if (checkForError(cudaStatus, "cudaDeviceReset failed!")) {
		return 1;
	}
	
	
	int currNumber = 0;
	for (int i = 0; i < numBlocks * threadsInBlock; i++) {
		checkIfCorrectlySorted(vector, currNumber, currNumber + vectorLengthPerThread - 1);
		currNumber += vectorLengthPerThread;
	}

	if (DBG) {
		printArray(vector, NUMBERS);
	}
	fflush(stdout);
	//checkIfCorrectlySorted(vector);

	return 0;
}