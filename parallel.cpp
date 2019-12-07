#include <cstdio>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm> 


#define DBG 0

#define NUMBERS_BIG 100000000 //2000000
#define NUMBERS_DBG 10

#define MAX_BIG 100000 //1000000
#define MAX_DBG 50

#define NUM_THREADS 4
#define NUMBERS ((DBG == 1) ? NUMBERS_DBG : NUMBERS_BIG)
#define MAX_NUMBER ((DBG == 1) ? MAX_DBG : MAX_BIG)

void printArrayFlat(int* A, int from, int to)
{
	int i;
	for (i = from; i < to; i++) {
		printf("%d, ", A[i]);
	}
}

void printArray(int* A, int size)
{
	int i;
	printf("\n");
	for (i = 0; i < size; i++) {
		printf("%d, ", A[i]);
	}
	printf("\n");
}

void printArrayDiff(int* A, int* B, int size)
{
	int i;
	printf("\n");
	for (i = 0; i < size; i++) {
		printf("\n%d \t %d ", A[i], B[i]);
		if (A[i] != B[i]) {
			printf("diff!!!!");
		}
	}
	printf("\n");
}

void fillArrayWithNumbersParallel(int* numbers) {
	int i, from, to;
	srand(time(NULL));
//#pragma omp parallel for schedule(static, NUMBERS / NUM_THREADS) num_threads(NUM_THREADS)
	for (i = 0; i < NUMBERS; i++) {
		numbers[i] = rand() % MAX_NUMBER;
	}
	//printf("\nTime in seconds (fill array parallel): %f", difftime(time2, time1));
}

/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
void merge(int arr[], int left_start, int mid, int right_start, int* alloc) {
	if (DBG) {
		printf("\nmerge: ");
		printArrayFlat(arr, left_start, mid);
		printf(" with ");
		printArrayFlat(arr, mid, right_start);
		fflush(stdout);
	}
	int i, j, k;

	int left_half_size = mid - left_start + 1;
	int right_half_size = right_start - mid;

	/* create temp arrays */

	int* L = &alloc[left_start];
	int* R = &alloc[mid + 1];

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
		arr[k] =L[i];
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

void mergeSort(int* arr, int left_start, int right_start, int* alloc) {
	if (left_start < right_start)
	{
		// Same as (l+r)/2, but avoids overflow for large l and h 
		int m = left_start + (right_start - left_start) / 2;

		// Sort first and second halves 
		if (DBG) {
			printf("\nl <%d, %d> ", left_start, m);
			printArrayFlat(arr, left_start, m);
			fflush(stdout);
		}
		mergeSort(arr, left_start, m, alloc);
		if (DBG) {
			printf("\nr <%d, %d>", m + 1, right_start);
			printArrayFlat(arr, m + 1, right_start);
			fflush(stdout);
		}
		mergeSort(arr, m + 1, right_start, alloc);
		merge(arr, left_start, m, right_start, alloc);
		if (DBG) {
			printf("\nm <%d, %d>", left_start, right_start);
			printArrayFlat(arr, left_start, right_start);
			fflush(stdout);
		}
	}
}

void mergeSortParallel(int* arr, int left_start, int right_start, int* alloc) {
	if (left_start < right_start)
	{
		if ((right_start - left_start) > (10000)) {
			int num = omp_get_thread_num();
			int threads = omp_get_num_threads();

			int m = left_start + (right_start - left_start) / 2;
#pragma omp task		
			{
				if (DBG) {
					printf("\nthread=%d, num_threads=%d <%d, %d>", num, threads, left_start, m);
					fflush(stdout);
				}
				mergeSortParallel(arr, left_start, m, alloc);
			}

#pragma omp task
			{
				if (DBG) {
					printf("\nthread=%d, num_threads=%d <%d, %d>", num, threads, m + 1, right_start);
					fflush(stdout);
				}
				mergeSortParallel(arr, m + 1, right_start, alloc);
			}
#pragma omp taskwait
			if (DBG) {
				printf("\nthread=%d, num_threads=%d skonczylem", num, threads);
				fflush(stdout);
			}
			merge(arr, left_start, m, right_start, alloc);
		}
		else {
			if (DBG) {
				printf("\n<%d, %d> licze dalej sekwencyjnie", left_start, right_start);
				fflush(stdout);
			}
			mergeSort(arr, left_start, right_start, alloc);
		}
	}
}



int main() {
	printf("\nThreads: %d, Numbers: %d\n", NUM_THREADS, NUMBERS);
	time_t time1, time2;
	omp_set_nested(1);

	int* alloc = (int*)malloc(NUMBERS * sizeof(int));
	int* numbersSeq = (int*)malloc(NUMBERS * sizeof(int));
	int* numbersPar = (int*)malloc(NUMBERS * sizeof(int));
	fillArrayWithNumbersParallel(numbersSeq);
	if (DBG) {
		printArray(numbersSeq, NUMBERS);
	}
	memcpy(numbersPar, numbersSeq, NUMBERS * sizeof(int));
	//std::sort(numbersSeq, numbersSeq + NUMBERS);
	
	time(&time1);
	mergeSort(numbersSeq, 0, NUMBERS - 1, alloc);
	time(&time2);
	printf("\nTime in seconds (mergesort sequential): %f", difftime(time2, time1));
	if (DBG) {
		printArray(numbersSeq, NUMBERS);
	}
	///*
	free(alloc);
	alloc = (int*)malloc(NUMBERS * sizeof(int));
	
	time(&time1);
#pragma omp parallel num_threads(NUM_THREADS)
		//num_threads(NUM_THREADS)
#pragma omp single
		{
			mergeSortParallel(numbersPar, 0, NUMBERS - 1, alloc);
		}printf("!!!!WYJATEK!!!!");
	
	time(&time2);
	printf("\nTime in seconds (mergesort parallel): %f", difftime(time2, time1));


	for (int i = 0; i < NUMBERS - 1; i++) {
		if (numbersPar[i] > numbersPar[i + 1]) {
			printf("BLAD!!!!");
		}
	}
	if (DBG) {
		printArray(numbersPar, NUMBERS);
	}
	
	return 0;
}

// optimizations:
// 1. generating numbers in parallel
// 2. prealokowanie pamiêci raz zamiast alokowania przy ka¿dym merge'owaniu (szybciej + wiêcej liczb)
// 3. sections -> tasks
// 4. parallel & sequential with threshold