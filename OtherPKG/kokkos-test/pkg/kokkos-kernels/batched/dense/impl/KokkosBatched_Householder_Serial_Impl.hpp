//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#ifndef KOKKOSBATCHED_HOUSEHOLDER_SERIAL_IMPL_HPP
#define KOKKOSBATCHED_HOUSEHOLDER_SERIAL_IMPL_HPP

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Householder_Serial_Internal.hpp"

namespace KokkosBatched {

///
/// Serial Impl
/// ===========

template <>
template <typename aViewType, typename tauViewType>
KOKKOS_INLINE_FUNCTION int SerialHouseholder<Side::Left>::invoke(const aViewType &a, const tauViewType &tau) {
  return SerialLeftHouseholderInternal::invoke(a.extent(0) - 1, a.data(), a.data() + a.stride(0), a.stride(0),
                                               tau.data());
}

}  // namespace KokkosBatched

#endif
