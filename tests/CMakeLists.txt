add_executable(superinfer_base_contracts unit/base/contracts_test.cpp)
target_link_libraries(superinfer_base_contracts PRIVATE superinfer_base)
add_test(NAME superinfer.base.contracts COMMAND superinfer_base_contracts)

add_executable(superinfer_architecture architecture/architecture_test.cpp)
target_link_libraries(superinfer_architecture PRIVATE
  superinfer_compiler superinfer_artifact superinfer_runtime)
target_compile_definitions(superinfer_architecture PRIVATE
  SUPERINFER_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME superinfer.architecture COMMAND superinfer_architecture)
add_test(NAME superinfer.architecture.dependencies
  COMMAND ${CMAKE_COMMAND}
    -DSUPERINFER_SOURCE_DIR=${CMAKE_SOURCE_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/architecture/dependency_check.cmake)

if(SUPERINFER_ENABLE_SANITIZERS)
  set_tests_properties(
    superinfer.base.contracts superinfer.architecture
    PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endif()
