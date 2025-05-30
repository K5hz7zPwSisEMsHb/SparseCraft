/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <iostream>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "KokkosBlas_dot_perf_test.hpp"
#include <benchmark/benchmark.h>

///////////////////////////////////////////////////////////////////////////////////////////////////
// The Level 1 BLAS perform scalar, vector and vector-vector operations;
//
// https://github.com/kokkos/kokkos-kernels/wiki/BLAS-1%3A%3Adot
//
// Usage: result = KokkosBlas::dot(x,y); KokkosBlas::dot(r,x,y);
// Multiplies each value of x(i) [x(i,j)] with y(i) or [y(i,j)] and computes the
// sum. (If x and y have scalar type Kokkos::complex, the complex conjugate of
// x(i) or x(i,j) will be used.) VectorX: A rank-1 Kokkos::View VectorY: A
// rank-1 Kokkos::View ReturnVector: A rank-0 or rank-1 Kokkos::View
//
// REQUIREMENTS:
// Y.rank == 1 or X.rank == 1
// Y.extent(0) == X.extent(0)

// Dot Test design:
// 1) create 1D View containing 1D matrix, aka a vector; this will be your X
// input matrix; 2) create 1D View containing 1D matrix, aka a vector; this will
// be your Y input matrix; 3) perform the dot operation on the two inputs, and
// capture result in "result"

// Here, m represents the desired length for each 1D matrix;
// "m" is used here, because code from another test was adapted for this test.
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class ExecSpace>
static void run(benchmark::State& state) {
  const auto m = state.range(0);
  const auto n = state.range(1);
  // Declare type aliases
  using Scalar   = double;
  using MemSpace = typename ExecSpace::memory_space;
  using Device   = Kokkos::Device<ExecSpace, MemSpace>;

  Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device> x(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x"), m, n);

  Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device> y(Kokkos::view_alloc(Kokkos::WithoutInitializing, "y"), m, n);

  Kokkos::View<Scalar*, Device> result(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x dot y"), n);

  // Declaring variable pool w/ a seeded random number;
  // a parallel random number generator, so you
  // won't get the same number with a given seed each time
  Kokkos::Random_XorShift64_Pool<ExecSpace> pool(123);

  Kokkos::fill_random(x, pool, 10.0);
  Kokkos::fill_random(y, pool, 10.0);
  ExecSpace space;

  Kokkos::fence();
  for (auto _ : state) {
    KokkosBlas::dot(space, result, x, y);
    space.fence();
  }

  const size_t iterFlop   = (size_t)2 * m * n;
  const size_t totalFlop  = iterFlop * state.iterations();
  state.counters["FLOP"]  = benchmark::Counter(iterFlop);
  state.counters["FLOPS"] = benchmark::Counter(totalFlop, benchmark::Counter::kIsRate);
}

BENCHMARK(run<Kokkos::DefaultHostExecutionSpace>)
    ->Name("KokkosBlas_dot_mv<DefaultHostExecutionSpace>")
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->ArgNames({"m", "n"})
    ->ArgsProduct({benchmark::CreateRange(100000, 100000000, 10), benchmark::CreateRange(5, 5, 1)});

BENCHMARK(run<Kokkos::DefaultExecutionSpace>)
    ->Name("KokkosBlas_dot_mv<DefaultExecutionSpace>")
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->ArgNames({"m", "n"})
    ->ArgsProduct({benchmark::CreateRange(100000, 100000000, 10), benchmark::CreateRange(5, 5, 1)});
