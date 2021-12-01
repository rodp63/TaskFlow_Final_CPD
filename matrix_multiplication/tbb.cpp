#include "matrix_multiplication.hpp"
#include <tbb/global_control.h>
#include <tbb/flow_graph.h>
#include <tbb/parallel_for.h>

// matrix_multiplication_tbb
void matrix_multiplication_tbb(unsigned num_threads) {

  using namespace tbb;
  using namespace tbb::flow;
  
  tbb::global_control control(
    tbb::global_control::max_allowed_parallelism, num_threads
  );
 
  tbb::parallel_for(0, N, 1, [=](int i){
    for(int j=0; j<N; ++j) {
      a[i][j] = i + j;
    }
  });
  
  tbb::parallel_for(0, N, 1, [=](int i){
    for(int j=0; j<N; ++j) {
      b[i][j] = i * j;
    }
  });
  
  tbb::parallel_for(0, N, 1, [=](int i){
    for(int j=0; j<N; ++j) {
      c[i][j] = 0;
    }
  });
  
  tbb::parallel_for(0, N, 1, [=](int i){
    for(int j=0; j<N; ++j) {
      for(int k=0; k<N; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  });
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  matrix_multiplication_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
