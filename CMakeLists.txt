cmake_minimum_required(VERSION 3.25)

project(superinfer VERSION 0.1.0 LANGUAGES CXX)

option(SUPERINFER_ENABLE_CUDA "Build CUDA-targeted components when a toolkit is available" OFF)
option(SUPERINFER_BUILD_TESTS "Build the CPU test suite" ON)
option(SUPERINFER_ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SuperInferOptions)

add_library(superinfer_base INTERFACE)
target_compile_features(superinfer_base INTERFACE cxx_std_20)
target_include_directories(superinfer_base INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
superinfer_apply_warnings(superinfer_base)

add_library(superinfer_ir INTERFACE)
target_link_libraries(superinfer_ir INTERFACE superinfer_base)

add_library(superinfer_compiler INTERFACE)
target_link_libraries(superinfer_compiler INTERFACE superinfer_ir)

add_library(superinfer_artifact INTERFACE)
target_link_libraries(superinfer_artifact INTERFACE superinfer_base superinfer_ir)

add_library(superinfer_kernels INTERFACE)
target_link_libraries(superinfer_kernels INTERFACE superinfer_base superinfer_ir)

add_library(superinfer_decode INTERFACE)
target_link_libraries(superinfer_decode INTERFACE superinfer_base superinfer_ir superinfer_kernels)

add_library(superinfer_runtime INTERFACE)
target_link_libraries(superinfer_runtime INTERFACE
  superinfer_base superinfer_artifact superinfer_kernels superinfer_decode)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/S00")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/artifacts/S00/architecture-targets.json" CONTENT [=[
{
  "superinfer_base": [],
  "superinfer_ir": ["superinfer_base"],
  "superinfer_compiler": ["superinfer_ir"],
  "superinfer_artifact": ["superinfer_base", "superinfer_ir"],
  "superinfer_kernels": ["superinfer_base", "superinfer_ir"],
  "superinfer_decode": ["superinfer_base", "superinfer_ir", "superinfer_kernels"],
  "superinfer_runtime": ["superinfer_base", "superinfer_artifact", "superinfer_kernels", "superinfer_decode"]
}
]=])

if(SUPERINFER_CUDA_AVAILABLE)
  add_library(superinfer_cuda INTERFACE)
  target_compile_features(superinfer_cuda INTERFACE cuda_std_20)
  target_compile_definitions(superinfer_cuda INTERFACE SUPERINFER_TARGET_SM120A=1)
  set_target_properties(superinfer_cuda PROPERTIES CUDA_ARCHITECTURES "120a")
endif()

if(SUPERINFER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS
  superinfer_base superinfer_ir superinfer_compiler superinfer_artifact
  superinfer_kernels superinfer_decode superinfer_runtime
  EXPORT SuperInferTargets)
install(EXPORT SuperInferTargets
  FILE SuperInferTargets.cmake
  NAMESPACE SuperInfer::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/SuperInfer")
