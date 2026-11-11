# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
# project() includes this after loading the inherited toolchain. Apply only to
# matching dependencies, before any targets compile their source files.
function(triton_apply_dependency_patch patch sentinel)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/${sentinel}")
    return()
  endif()
  file(LOCK "${PROJECT_SOURCE_DIR}/.triton-dependency-fixes.lock"
       GUARD FUNCTION TIMEOUT 120)
  set(patch_file "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches/${patch}")
  # Stop repository discovery at this source root: installed source copies may
  # have no .git directory while living inside the enclosing Triton checkout.
  set(git_command "${CMAKE_COMMAND}" -E env
      "GIT_CEILING_DIRECTORIES=${PROJECT_SOURCE_DIR}" git apply)
  execute_process(COMMAND ${git_command} --reverse --check "${patch_file}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE applied OUTPUT_QUIET ERROR_QUIET)
  if(applied EQUAL 0)
    return()
  endif()
  execute_process(COMMAND ${git_command} --check "${patch_file}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE compatible OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT compatible EQUAL 0)
    message(FATAL_ERROR "Dependency patch ${patch} needs review for ${PROJECT_SOURCE_DIR}: ${output}${error}")
  endif()
  execute_process(COMMAND ${git_command} "${patch_file}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Dependency patch ${patch} failed: ${output}${error}")
  endif()
  message(STATUS "Applied dependency fix ${patch} to ${PROJECT_SOURCE_DIR}")
endfunction()
triton_apply_dependency_patch(grpc-state-returns.patch src/core/call/call_state.h)
triton_apply_dependency_patch(protobuf-bounded-access.patch upb/message/internal/accessors.h)
