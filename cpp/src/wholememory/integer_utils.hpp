/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/util/integer_utils.hpp>

namespace wholememory {

//! Utility functions
/**
 * Finds the smallest integer not less than `number_to_round` and modulo `S` is
 * zero. This function assumes that `number_to_round` is non-negative and
 * `modulus` is positive.
 */
template <typename S>
inline S round_up_unsafe(S number_to_round, S modulus) noexcept
{
  auto remainder = number_to_round % modulus;
  if (remainder == 0) { return number_to_round; }
  auto rounded_up = number_to_round - remainder + modulus;
  return rounded_up;
}

/**
 * Divides the left-hand-side by the right-hand-side, rounding up
 * to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
 *
 * @param dividend the number to divide
 * @param divisor the number by which to divide
 * @return The least integer multiple of divisor which is greater than or equal to
 * the non-integral division dividend/divisor.
 *
 * @note sensitive to overflow, i.e. if dividend > std::numeric_limits<S>::max() - divisor,
 * the result will be incorrect
 */
template <typename S, typename T>
constexpr inline S div_rounding_up_unsafe(const S& dividend, const T& divisor) noexcept
{
  return raft::div_rounding_up_unsafe(dividend, divisor);
}

/**
 * Divides the left-hand-side by the right-hand-side, rounding up
 * to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
 *
 * @param dividend the number to divide
 * @param divisor the number of by which to divide
 * @return The least integer multiple of divisor which is greater than or equal to
 * the non-integral division dividend/divisor.
 *
 * @note will not overflow, and may _or may not_ be slower than the intuitive
 * approach of using (dividend + divisor - 1) / divisor
 */
template <typename I>
constexpr inline I div_rounding_up_safe(I dividend, I divisor) noexcept
{
  return raft::div_rounding_up_safe<I>(dividend, divisor);
}

}  // namespace wholememory
