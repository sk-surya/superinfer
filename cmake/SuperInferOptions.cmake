function(superinfer_apply_warnings target)
  if(MSVC)
    target_compile_options(${target} INTERFACE
      "$<$<COMPILE_LANGUAGE:CXX>:/W4;/WX>")
  else()
    target_compile_options(${target} INTERFACE
      "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>")
  endif()

  if(SUPERINFER_ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(${target} INTERFACE
      "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-fno-omit-frame-pointer>")
    target_link_options(${target} INTERFACE
      "$<$<LINK_LANGUAGE:CXX>:-fsanitize=address,undefined>")
  endif()
endfunction()

set(SUPERINFER_CUDA_AVAILABLE FALSE)
if(SUPERINFER_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(SUPERINFER_CUDA_AVAILABLE TRUE)
    message(STATUS "SuperInfer CUDA mode enabled for sm_120a")
  else()
    message(WARNING "SUPERINFER_ENABLE_CUDA=ON but no CUDA compiler was found; CUDA targets are disabled")
  endif()
endif()
