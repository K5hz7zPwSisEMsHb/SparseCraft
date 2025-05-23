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
#include <KokkosBlas1_team_dot.hpp>
#include <Kokkos_Random.hpp>
#include "KokkosKernels_TestStringUtils.hpp"

#include <benchmark/benchmark.h>

// Functor to handle the case of a "without Cuda" build
template <class Vector, class ExecSpace>
struct teamDotFunctor {
  // Compile - time check to see if your data type is a Kokkos::View:
  static_assert(Kokkos::is_view<Vector>::value, "Vector is not a Kokkos::View.");

  using Scalar = typename Vector::non_const_value_type;
  // Vector is templated on memory space
  using execution_space = ExecSpace;  // Kokkos Execution Space
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;

  // Declare Kokkos::View Vectors, x and y
  Vector x;
  Vector y;

  // Functor instead of KOKKOS_LAMBDA expression

  KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const { KokkosBlas::Experimental::dot(team, x, y); }
  // Constructor
  teamDotFunctor(Vector X_, Vector Y_) {
    x = X_;
    y = Y_;
  }
};

template <class ExecSpace>
static void run(benchmark::State& state) {
  const auto m      = state.range(0);
  const auto repeat = state.range(1);
  // Declare type aliases
  using Scalar   = double;
  using MemSpace = typename ExecSpace::memory_space;

  // For the Team implementation of dot; ExecSpace is implicit;
  using policy = Kokkos::TeamPolicy<ExecSpace>;

  // Create 1D view w/ Device as the ExecSpace; this is an input vector
  Kokkos::View<Scalar*, MemSpace> x("X", m);
  // Create 1D view w/ Device as the ExecSpace; this is the output vector
  Kokkos::View<Scalar*, MemSpace> y("Y", m);

  // Here, deep_copy is filling / copying values into Host memory from Views X
  // and Y
  Kokkos::deep_copy(x, 3.0);
  Kokkos::deep_copy(y, 2.0);

  Kokkos::fence();
  for (auto _ : state) {
    teamDotFunctor<Kokkos::View<Scalar*, MemSpace>, ExecSpace> teamDotFunctorLiveTestInstance(x, y);
    Kokkos::parallel_for("TeamDotUsage -- Live Test", policy(1, Kokkos::AUTO), teamDotFunctorLiveTestInstance);
    Kokkos::fence();
  }
  const size_t iterFlop   = (size_t)2 * m;
  const size_t totalFlop  = iterFlop * state.iterations();
  state.counters["FLOP"]  = benchmark::Counter(iterFlop);
  state.counters["FLOPS"] = benchmark::Counter(totalFlop, benchmark::Counter::kIsRate);
}

BENCHMARK(run<Kokkos::DefaultExecutionSpace>)
    ->Name("KokkosBlas_team_dot/run<Kokkos::DefaultExecutionSpace>")
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->ArgName("m")
    ->RangeMultiplier(10)
    ->Range(100000, 100000000);
