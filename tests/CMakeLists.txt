# The test suite uses assert as its compact assertion primitive. Keep assertions active in
# release CUDA builds so side-effecting fixture setup is not compiled away by NDEBUG.
if(NOT MSVC)
  add_compile_options(-UNDEBUG)
endif()

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

add_executable(superinfer_sm120_target_profile unit/sm120/compiler/target_profile_test.cpp)
target_link_libraries(superinfer_sm120_target_profile PRIVATE superinfer_compiler)
target_include_directories(superinfer_sm120_target_profile PRIVATE
  "${CMAKE_SOURCE_DIR}/backends")
add_test(NAME superinfer.sm120.target_profile COMMAND superinfer_sm120_target_profile)

add_executable(superinfer_sm120_memory_planner unit/sm120/compiler/memory_planner_test.cpp)
target_link_libraries(superinfer_sm120_memory_planner PRIVATE superinfer_compiler)
add_test(NAME superinfer.sm120.memory_planner COMMAND superinfer_sm120_memory_planner)

add_executable(superinfer_sm120_memory_property property/sm120/compiler/memory_planner_property_test.cpp)
target_link_libraries(superinfer_sm120_memory_property PRIVATE superinfer_compiler)
add_test(NAME superinfer.sm120.memory_property COMMAND superinfer_sm120_memory_property)

add_executable(superinfer_sm120_memory_golden golden/sm120/compiler/memory_plan_golden_test.cpp)
target_link_libraries(superinfer_sm120_memory_golden PRIVATE superinfer_compiler)
target_compile_definitions(superinfer_sm120_memory_golden PRIVATE
  SUPERINFER_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME superinfer.sm120.memory_golden COMMAND superinfer_sm120_memory_golden)

add_executable(superinfer_sm120_specializer unit/sm120/compiler/specializer_test.cpp)
target_link_libraries(superinfer_sm120_specializer PRIVATE superinfer_compiler)
target_include_directories(superinfer_sm120_specializer PRIVATE
  "${CMAKE_SOURCE_DIR}/backends")
add_test(NAME superinfer.sm120.specializer COMMAND superinfer_sm120_specializer)

add_executable(superinfer_sm120_baseline_provider unit/sm120/kernels/baseline_provider_test.cpp)
target_link_libraries(superinfer_sm120_baseline_provider PRIVATE superinfer_kernels)
target_include_directories(superinfer_sm120_baseline_provider PRIVATE
  "${CMAKE_SOURCE_DIR}/backends")
add_test(NAME superinfer.sm120.baseline_provider COMMAND superinfer_sm120_baseline_provider)

add_executable(superinfer_sm120_plan_binding unit/sm120/runtime/plan_binding_test.cpp)
target_link_libraries(superinfer_sm120_plan_binding PRIVATE superinfer_runtime)
add_test(NAME superinfer.sm120.plan_binding COMMAND superinfer_sm120_plan_binding)

add_executable(superinfer_sm120_session_ownership unit/sm120/runtime/session_ownership_test.cpp)
target_link_libraries(superinfer_sm120_session_ownership PRIVATE superinfer_runtime)
add_test(NAME superinfer.sm120.session_ownership COMMAND superinfer_sm120_session_ownership)

add_executable(superinfer_sm120_reference_executor integration/sm120/reference_executor_test.cpp)
target_link_libraries(superinfer_sm120_reference_executor PRIVATE superinfer_ir)
target_include_directories(superinfer_sm120_reference_executor PRIVATE
  "${CMAKE_SOURCE_DIR}/backends")
add_test(NAME superinfer.sm120.reference_executor COMMAND superinfer_sm120_reference_executor)

add_executable(superinfer_sm120_gated_delta_reference integration/sm120/gated_delta_reference_test.cpp)
target_link_libraries(superinfer_sm120_gated_delta_reference PRIVATE superinfer_ir)
target_include_directories(superinfer_sm120_gated_delta_reference PRIVATE
  "${CMAKE_SOURCE_DIR}/backends")
add_test(NAME superinfer.sm120.gated_delta_reference COMMAND superinfer_sm120_gated_delta_reference)

add_executable(superinfer_sm120_reference_primitives integration/sm120/reference_primitives_test.cpp)
target_link_libraries(superinfer_sm120_reference_primitives PRIVATE superinfer_ir)
target_include_directories(superinfer_sm120_reference_primitives PRIVATE
  "${CMAKE_SOURCE_DIR}/backends")
add_test(NAME superinfer.sm120.reference_primitives COMMAND superinfer_sm120_reference_primitives)

add_executable(superinfer_qwen38_frontend unit/frontends/qwen38_frontend_test.cpp)
target_link_libraries(superinfer_qwen38_frontend PRIVATE superinfer_ir superinfer_compiler)
target_include_directories(superinfer_qwen38_frontend PRIVATE
  "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/backends")
add_test(NAME superinfer.qwen38.frontend COMMAND superinfer_qwen38_frontend)

add_executable(superinfer_decode_kv_cache unit/decode/kv_cache_test.cpp)
target_link_libraries(superinfer_decode_kv_cache PRIVATE superinfer_runtime)
add_test(NAME superinfer.decode.kv_cache COMMAND superinfer_decode_kv_cache)

add_executable(superinfer_decode_greedy unit/decode/greedy_strategy_test.cpp)
target_link_libraries(superinfer_decode_greedy PRIVATE superinfer_runtime)
add_test(NAME superinfer.decode.greedy COMMAND superinfer_decode_greedy)

add_executable(superinfer_artifact_test integration/artifact/artifact_test.cpp)
target_link_libraries(superinfer_artifact_test PRIVATE superinfer_artifact)
add_test(NAME superinfer.artifact.binary COMMAND superinfer_artifact_test)

if(SUPERINFER_CUDA_AVAILABLE)
  add_executable(superinfer_sm120_cuda_ownership gpu/sm120/cuda_ownership_test.cu)
  target_compile_features(superinfer_sm120_cuda_ownership PRIVATE cuda_std_20)
  target_include_directories(superinfer_sm120_cuda_ownership PRIVATE
    "${CMAKE_SOURCE_DIR}/backends")
  target_link_libraries(superinfer_sm120_cuda_ownership PRIVATE
    superinfer_sm120_cuda_runtime CUDA::cudart)
  set_target_properties(superinfer_sm120_cuda_ownership PROPERTIES CUDA_ARCHITECTURES "120a")
  add_test(NAME superinfer.sm120.cuda_ownership COMMAND superinfer_sm120_cuda_ownership)
  set_tests_properties(superinfer.sm120.cuda_ownership PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(superinfer_sm120_cuda_plan_executor gpu/sm120/cuda_plan_executor_test.cu)
  target_compile_features(superinfer_sm120_cuda_plan_executor PRIVATE cuda_std_20)
  target_include_directories(superinfer_sm120_cuda_plan_executor PRIVATE
    "${CMAKE_SOURCE_DIR}/backends")
  target_link_libraries(superinfer_sm120_cuda_plan_executor PRIVATE
    superinfer_sm120_cuda_runtime superinfer_compiler CUDA::cudart)
  set_target_properties(superinfer_sm120_cuda_plan_executor PROPERTIES CUDA_ARCHITECTURES "120a")
  add_test(NAME superinfer.sm120.cuda_plan_executor COMMAND superinfer_sm120_cuda_plan_executor)
  set_tests_properties(superinfer.sm120.cuda_plan_executor PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(superinfer_sm120_qwen38_layer_artifact
    gpu/sm120/qwen38_layer_artifact_test.cu)
  target_compile_features(superinfer_sm120_qwen38_layer_artifact PRIVATE cuda_std_20)
  target_include_directories(superinfer_sm120_qwen38_layer_artifact PRIVATE
    "${CMAKE_SOURCE_DIR}/backends")
  target_link_libraries(superinfer_sm120_qwen38_layer_artifact PRIVATE
    superinfer_sm120_cuda_runtime CUDA::cudart)
  set_target_properties(superinfer_sm120_qwen38_layer_artifact PROPERTIES
    CUDA_ARCHITECTURES "120a")
  add_test(NAME superinfer.sm120.qwen38.layer_artifact
    COMMAND superinfer_sm120_qwen38_layer_artifact)
  set_tests_properties(superinfer.sm120.qwen38.layer_artifact PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(SUPERINFER_ENABLE_SANITIZERS)
  set_tests_properties(
    superinfer.base.contracts superinfer.architecture superinfer.ir.semantic
    superinfer.ir.semantic.properties superinfer.ir.lowered_plan
    superinfer.compiler.pass_manager superinfer.sm120.target_profile
    superinfer.sm120.memory_planner superinfer.sm120.memory_property
    superinfer.sm120.memory_golden superinfer.sm120.specializer
    superinfer.sm120.baseline_provider superinfer.sm120.plan_binding
    superinfer.sm120.session_ownership
    superinfer.sm120.reference_executor superinfer.sm120.reference_primitives
    superinfer.artifact.binary
    PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endif()
