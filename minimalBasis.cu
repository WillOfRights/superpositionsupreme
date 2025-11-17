#include <iostream>
#include <random>

using namespace std;
// Take the orthogonal complement of a vector of dim N (v) onto M unit vectors of dim N (P) and output to O
__global__ void project(int N, int M, const float *v, const float *P, float *O) {

  // Take the complement from the projection of vector i
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  // In index j
  const uint j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < M && j < N) {
    // Compute the dot product
    float dotProduct = 0.0;
    for (int dotProductIndex = 0; dotProductIndex < N; dotProductIndex++) {
      dotProduct += v[dotProductIndex] * P[i * N + dotProductIndex];
    }

    // o = v - (v*p)p
    O[i * N + j] = v[j] - dotProduct * P[i * N + j];
  }
}

int ceilDiv(int x, int y) {
  return 1 + ((x - 1) / y);
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

int main(int argc, char *argv[]) {
  cout << "Initializing main function." << endl;

  // --- Compute sizes ---
  // Vector dimension
  int N = 3;
  int vec_size = N * sizeof(float);

  // Number of vectors
  int M = 4;
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
  
  // Copy host to GPU memory
  cudaMemcpy(d_v, h_v, vec_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_P, h_P, vec_num_size, cudaMemcpyHostToDevice);

  // Create dimensions
  dim3 gridDim(ceilDiv(M, 32), ceilDiv(N, 32), 1);
  dim3 blockDim(32, 32, 1);

  // Run projection
  cout << "Projecting vectors" << endl;
  project<<<gridDim, blockDim>>>(N, M, d_v, d_P, d_O);

  // Copy GPU to host memory
  cudaMemcpy(h_O, d_O, vec_num_size, cudaMemcpyDeviceToHost);
  
  // Print original vector
  cout << "Original vector: {";
  for(int i = 0; i < N; i++) {
    cout << h_v[i] << ",";
  }
  cout << "}" << endl;

  // Print our dictionary vectors
  cout << "Dictionary vectors: " << endl;
  for(int i = 0; i < M; i++) {
    cout << "Vector " << i << ": {";
    for (int j = 0; j < N; j++) {
      cout << h_P[i*N + j] << ",";
    }
    cout << "}" << endl;
  }

  // Print orthogonal complements
  cout << "Complement vectors: " << endl;
  for(int i = 0; i < M; i++) {
    cout << "Vector " << i << ": {";
    for (int j = 0; j < N; j++) {
      cout << h_O[i*N + j] << ",";
    }
    cout << "}" << endl;
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



