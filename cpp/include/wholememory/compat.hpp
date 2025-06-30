/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <iterator>
#include <random>
#include <vector>

namespace wholememory {
namespace compat {

// Detect g++ 14 compiler
#if defined(__GNUC__) && __GNUC__ >= 14
#define WHOLEMEMORY_GCC_14_OR_HIGHER 1
#else
#define WHOLEMEMORY_GCC_14_OR_HIGHER 0
#endif

// Compatibility implementation of std::sample
// Mainly used to solve g++ 14 compatibility issues with std::sample
template<typename InputIt, typename OutputIt, typename Distance, typename UniformRandomBitGenerator>
OutputIt sample(InputIt first, InputIt last, OutputIt out, Distance n, UniformRandomBitGenerator&& g) {
#if WHOLEMEMORY_GCC_14_OR_HIGHER
    // For g++ 14+, use custom implementation
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    std::vector<value_type> temp(first, last);
    
    if (temp.empty() || n <= 0) {
        return out;
    }
    
    // Fisher-Yates shuffle algorithm
    for (size_t i = 0; i < std::min(static_cast<size_t>(n), temp.size()); ++i) {
        std::uniform_int_distribution<size_t> dist(i, temp.size() - 1);
        size_t j = dist(g);
        if (i != j) {
            std::swap(temp[i], temp[j]);
        }
    }
    
    // Copy first n elements to output
    return std::copy(temp.begin(), temp.begin() + std::min(static_cast<size_t>(n), temp.size()), out);
#else
    // For other compilers, use standard library implementation
    return std::sample(first, last, out, n, std::forward<UniformRandomBitGenerator>(g));
#endif
}

} // namespace compat
} // namespace wholememory

// For g++ 14+, provide macro definitions to redirect std::sample calls
// to our compatibility implementation
#if WHOLEMEMORY_GCC_14_OR_HIGHER

// Provide compatibility macro for std::sample
#define std_sample_impl(first, last, out, n, g) wholememory::compat::sample(first, last, out, n, g)

#else

// For other compilers, use standard library function
#define std_sample_impl(first, last, out, n, g) std::sample(first, last, out, n, g)

#endif // WHOLEMEMORY_GCC_14_OR_HIGHER 