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

add_executable(superinfer_semantic_ir unit/ir/semantic_ir_test.cpp)
target_link_libraries(superinfer_semantic_ir PRIVATE superinfer_ir)
target_compile_definitions(superinfer_semantic_ir PRIVATE
  SUPERINFER_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME superinfer.ir.semantic COMMAND superinfer_semantic_ir)

add_executable(superinfer_semantic_ir_properties property/ir/semantic_property_test.cpp)
target_link_libraries(superinfer_semantic_ir_properties PRIVATE superinfer_ir)
add_test(NAME superinfer.ir.semantic.properties COMMAND superinfer_semantic_ir_properties)

add_executable(superinfer_lowered_plan unit/ir/lowered_plan_test.cpp)
target_link_libraries(superinfer_lowered_plan PRIVATE superinfer_ir superinfer_runtime)
target_compile_definitions(superinfer_lowered_plan PRIVATE
  SUPERINFER_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME superinfer.ir.lowered_plan COMMAND superinfer_lowered_plan)

add_executable(superinfer_pass_manager unit/compiler/pass_manager_test.cpp)
target_link_libraries(superinfer_pass_manager PRIVATE superinfer_compiler)
add_test(NAME superinfer.compiler.pass_manager COMMAND superinfer_pass_manager)

add_executable(superinfer_artifact_test integration/artifact/artifact_test.cpp)
target_link_libraries(superinfer_artifact_test PRIVATE superinfer_artifact)
add_test(NAME superinfer.artifact.binary COMMAND superinfer_artifact_test)

if(SUPERINFER_ENABLE_SANITIZERS)
  set_tests_properties(
    superinfer.base.contracts superinfer.architecture superinfer.ir.semantic
    superinfer.ir.semantic.properties superinfer.ir.lowered_plan
    superinfer.compiler.pass_manager superinfer.artifact.binary
    PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endif()
