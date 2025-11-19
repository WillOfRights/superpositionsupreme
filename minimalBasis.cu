#include <iostream>
#include <chrono>
#include <ctime>
#include <random>

using namespace std;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
// Take the orthogonal complement of a vector of dim N (v) onto M unit vectors of dim N (P) and output to O
__global__ void project(int N, int M, const float *v, const float *P, float *O) {

  // This block computes a BLOCKSIZE x BLOCKSIZE chunk of the output
  const uint outputVecStart = blockIdx.x;
  const uint outputDimStart = blockIdx.y;

  // Cached partial values of vectors in shared mem
  __shared__ float vCache[BLOCKSIZE];
  __shared__ float PCache[BLOCKSIZE * BLOCKSIZE];

  // This thread computes a specific entry of that chunk
  const uint threadVecOffset = threadIdx.x % BLOCKSIZE;
  const uint threadDimOffset = threadIdx.x / BLOCKSIZE;

  // Set pointers to starting positions
  v += 0;
  P += outputVecStart * BLOCKSIZE * N;
  O += outputVecStart * BLOCKSIZE * N + outputDimStart * BLOCKSIZE;

  // Floats that we want to remember for final output as we read through the data
  float v_thread;
  float p_thread;

  float dotProduct = 0.0;
  for (uint blockIdx = 0; blockIdx < CEIL_DIV(N, BLOCKSIZE); blockIdx++) {
    // Populate cached memory per block on a per thread basis
    PCache[threadVecOffset * BLOCKSIZE + threadDimOffset] = P[threadVecOffset * N + threadDimOffset];
    if (threadVecOffset == 0) {
      vCache[threadDimOffset] = v[threadDimOffset];
    }

    // Store to local memory if needed
    if (blockIdx == outputDimStart) {
      v_thread = v[threadDimOffset];
      p_thread = P[threadVecOffset * N + threadDimOffset];
    }

    __syncthreads();
    v += BLOCKSIZE;
    P += BLOCKSIZE;

    // Increment dot product
    for (int i = 0; i < BLOCKSIZE; i++) {
      dotProduct += PCache[threadVecOffset * BLOCKSIZE + i] * vCache[i];
    }

    // Wait until cache is free across threads
    __syncthreads();
  }

  // o = v - (v*p)p
  O[threadVecOffset * N + threadDimOffset] = v_thread - dotProduct * p_thread;
}

// Generate a test vector of size N
void generateTestVector(int N, float* vector) {
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);
  for(int i = 0; i < N; i++) {
    vector[i] = dis(gen);
  }
}

// Normalize a vector
void normalizeVector(int N, float* vector) {
  float square_norm = 0.0;
  for(int i = 0; i < N; i++) {
    square_norm += vector[i] * vector[i];
  }
  
  float norm = sqrt(square_norm);
  for(int i = 0; i < N; i++) {
    vector[i] = vector[i] / norm;
  }
}

// Compare for equality in double checking correctness
static bool AreEqual(float f1,float f2) { 
  float epsilon = 1e-06;
  return abs(f1 - f2) <= epsilon;
}

int main(int argc, char *argv[]) {
  cout << "Initializing main function." << endl;

  // --- Compute sizes ---
  // Vector dimension
  int N = 4096;
  int vec_size = N * sizeof(float);

  // Number of vectors
  int M = 4096;
  int vec_num_size = M * vec_size;

  // Vector to project
  float* h_v = (float*)malloc(vec_size);
  generateTestVector(N, h_v);

  // Projection vectors
  float* h_P = (float*)malloc(vec_num_size);
  for (int i = 0; i < M; i++) {
    generateTestVector(N, &h_P[i*N]);
    normalizeVector(N, &h_P[i*N]);
  }

  // Output
  float* h_O = (float*)malloc(vec_num_size);

  // Allocate vectors in GPU memory
  float* d_v;
  cudaMalloc(&d_v, vec_size);
  float* d_P;
  cudaMalloc(&d_P, vec_num_size);
  float* d_O;
  cudaMalloc(&d_O, vec_num_size);

  // Create dimensions
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  dim3 blockDim(32 * 32);

  // Copy host to GPU memory
  cudaMemcpy(d_v, h_v, vec_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_P, h_P, vec_num_size, cudaMemcpyHostToDevice);

  // Measure start of kernel
  cudaDeviceSynchronize();
  auto start = chrono::system_clock::now();
  time_t start_time = chrono::system_clock::to_time_t(start);
  cout << "Projecting vectors started at " << ctime(&start_time) << endl;

  // Run projection
  project<32><<<gridDim, blockDim>>>(N, M, d_v, d_P, d_O);

  // Measure end of kernel
  cudaDeviceSynchronize();
  auto end = chrono::system_clock::now();
  time_t end_time = chrono::system_clock::to_time_t(end);
  cout << "Projecting vectors finished at " << ctime(&end_time) << endl;
  chrono::duration<double> duration = end-start;
  cout << "TOTAL TIME ELAPSED: " << duration.count() << endl;

  // Copy GPU to host memory
  cudaMemcpy(h_O, d_O, vec_num_size, cudaMemcpyDeviceToHost);
  
  // Print original vector
  cout << "Original vector: {";
  for(int i = 0; i < N && i < 10; i++) {
    cout << h_v[i] << ",";
  }
  cout << "}" << endl;

  // Print our dictionary vectors
  cout << "Dictionary vectors: " << endl;
  for(int i = 0; i < M && i < 10; i++) {
    cout << "Vector " << i << ": {";
    for (int j = 0; j < N && j < 10; j++) {
      cout << h_P[i*N + j] << ",";
    }
    cout << "}" << endl;
  }

  // Print orthogonal complements
  cout << "Complement vectors: " << endl;
  for(int i = 0; i < M && i < 10; i++) {
    cout << "Vector " << i << ": {";
    for (int j = 0; j < N && j < 10; j++) {
      cout << h_O[i*N + j] << ",";
    }
    cout << "}" << endl;
  }

  // QA
  int randIndex = rand() % M;
  cout << "Spot checking vector " << randIndex << endl;
  float dotProduct = 0.0;
  bool pass = true;
  for (int i = 0; i < N; i++) {
    dotProduct += h_v[i] * h_P[randIndex * N + i];
  }
  for (int i = 0; i < N && pass; i++) {
    pass = AreEqual(h_v[i] - dotProduct * h_P[randIndex * N + i], h_O[randIndex * N + i])
      && pass;
    if (!pass) {
      cout << "FAILED!" << endl;
      cout << "Difference in dimension " << i << ": " << h_v[i] - dotProduct * h_P[randIndex * N + i] - h_O[randIndex * N + i] << endl;
    }
  }
  if (pass) {
    cout << "PASS!" << endl;
  }

  // GC
  free(h_v);
  free(h_P);
  free(h_O);
  cudaFree(d_v);
  cudaFree(d_P);
  cudaFree(d_O);

  return 0;
}



