# Applies a patch to the current working directory, skipping it if it is already applied.
#
# Usage: cmake -P onnxruntime_apply_patch.cmake -- <patch_executable> [patch_args...] -i <patch_file>
#
# ExternalProject re-runs a PATCH_COMMAND on the existing, already patched source tree whenever the
# patch step is invalidated without a re-download (e.g. the PATCH_COMMAND line changes). A plain
# `patch` call then fails with "Reversed (or previously applied) patch detected", so check first.

set(_patch_command)
set(_after_separator FALSE)
math(EXPR _last_arg_index "${CMAKE_ARGC} - 1")
foreach(_i RANGE ${_last_arg_index})
  if(_after_separator)
    list(APPEND _patch_command "${CMAKE_ARGV${_i}}")
  elseif(CMAKE_ARGV${_i} STREQUAL "--")
    set(_after_separator TRUE)
  endif()
endforeach()

list(LENGTH _patch_command _patch_command_length)
list(JOIN _patch_command " " _patch_command_string)
if(_patch_command_length LESS 2)
  message(FATAL_ERROR "Usage: cmake -P onnxruntime_apply_patch.cmake -- <patch_executable> [patch_args...]")
endif()

# --force/--forward keep GNU patch from prompting about (or auto-reversing) reversed hunks.
execute_process(
  COMMAND ${_patch_command} --forward --dry-run
  RESULT_VARIABLE _forward_result
  OUTPUT_VARIABLE _forward_output
  ERROR_VARIABLE _forward_output
)
if(NOT _forward_result EQUAL 0)
  execute_process(
    COMMAND ${_patch_command} --reverse --force --dry-run
    RESULT_VARIABLE _reverse_result
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(_reverse_result EQUAL 0)
    message(STATUS "Patch already applied, skipping: ${_patch_command_string}")
    return()
  endif()
  message(FATAL_ERROR "Patch does not apply: ${_patch_command_string}\n${_forward_output}")
endif()

execute_process(
  COMMAND ${_patch_command} --forward
  RESULT_VARIABLE _apply_result
)
if(NOT _apply_result EQUAL 0)
  message(FATAL_ERROR "Failed to apply patch: ${_patch_command_string}")
endif()
