/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "logger.hpp"

#include <cstdio>
#include <memory>
#include <string>

namespace wholememory {

LogLevel& get_log_level()
{
  static LogLevel log_level = LEVEL_INFO;
  return log_level;
}

void set_log_level(LogLevel lev) { get_log_level() = lev; }

bool will_log_for(LogLevel lev) { return lev <= get_log_level(); }

}  // namespace wholememory
