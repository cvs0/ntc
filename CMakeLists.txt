cmake_minimum_required(VERSION 3.20)
project(ntc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(ntc STATIC cpp/src/ntc.cpp)
target_include_directories(ntc PUBLIC cpp/include ${CMAKE_SOURCE_DIR}/third_party)
target_compile_options(ntc PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_executable(ntc_decode cpp/src/decode_main.cpp)
target_link_libraries(ntc_decode PRIVATE ntc)
target_compile_options(ntc_decode PRIVATE -Wall -Wextra)

add_executable(test_ntc cpp/tests/test_ntc.cpp)
target_link_libraries(test_ntc PRIVATE ntc)
target_compile_options(test_ntc PRIVATE -Wall -Wextra)

add_executable(expcode_bytes cpp/src/expcode_bytes.cpp)
target_link_libraries(expcode_bytes PRIVATE ntc)
target_include_directories(expcode_bytes PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_compile_options(expcode_bytes PRIVATE -Wall -Wextra)

enable_testing()
add_test(NAME test_ntc COMMAND test_ntc ${CMAKE_SOURCE_DIR}/tests/fixtures)

find_program(GLSLANG glslangValidator)
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SPV_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPV_DIR})

function(ntc_spirv src env out_name)
    if(NOT GLSLANG)
        return()
    endif()
    add_custom_command(
        OUTPUT ${SPV_DIR}/${out_name}
        COMMAND ${GLSLANG} -V --target-env ${env} "-I${SHADER_DIR}"
                ${SHADER_DIR}/${src} -o ${SPV_DIR}/${out_name}
        DEPENDS ${SHADER_DIR}/${src} ${SHADER_DIR}/ntc_decode.glsl
        VERBATIM)
    set(NTC_SPV_OUTPUTS ${NTC_SPV_OUTPUTS} ${SPV_DIR}/${out_name} PARENT_SCOPE)
endfunction()

if(GLSLANG)
    ntc_spirv(ntc_decompress.comp vulkan1.3 ntc_decompress.comp.spv)
    ntc_spirv(ntc_decompress_fp16.comp vulkan1.3 ntc_decompress_fp16.comp.spv)
    ntc_spirv(ntc_sample.frag vulkan1.3 ntc_sample.frag.spv)
    ntc_spirv(fullscreen.vert vulkan1.3 fullscreen.vert.spv)
    ntc_spirv(ntc_subgroup_probe.comp vulkan1.3 ntc_subgroup_probe.comp.spv)
    ntc_spirv(ntc_decompress_gl.comp opengl ntc_decompress_gl.comp.spv)
    ntc_spirv(ntc_sample_gl.frag opengl ntc_sample_gl.frag.spv)
    add_custom_target(ntc_shaders ALL DEPENDS ${NTC_SPV_OUTPUTS})
    add_test(NAME shader_spv COMMAND ${CMAKE_COMMAND} -E echo "spir-v built")
endif()

find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    add_executable(ntc_vk examples/vulkan/main.cpp)
    target_link_libraries(ntc_vk PRIVATE ntc Vulkan::Vulkan)
    target_compile_options(ntc_vk PRIVATE -Wall -Wextra)
    add_dependencies(ntc_vk ntc_shaders)
endif()

find_library(EGL_LIB EGL)
find_library(OPENGL_LIB GL)
if(EGL_LIB AND OPENGL_LIB)
    add_executable(ntc_gl examples/opengl/main.cpp)
    target_link_libraries(ntc_gl PRIVATE ntc ${EGL_LIB} ${OPENGL_LIB})
    target_compile_options(ntc_gl PRIVATE -Wall -Wextra)
endif()
