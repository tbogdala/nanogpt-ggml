cmake_minimum_required(VERSION 3.14)
project("NanogptGgml" C)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# include upstream ggml options
set(GGML_BUILD_TESTS Off)
set(GGML_BUILD_EXAMPLES Off)
add_subdirectory("ggml")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)

add_executable(nanogpt_ggml
    main.c
    dataset.c
    model.c
    utility.c
)
target_link_libraries(nanogpt_ggml PUBLIC ggml)

# Unit testing

add_subdirectory(Unity)

add_executable(test_dataset 
    tests/test_dataset.c
    dataset.c
    )
target_link_libraries(test_dataset PRIVATE unity)

add_executable(test_model_stage1 
    tests/test_model_stage1.c
    dataset.c
    model.c
    utility.c
    )
target_link_libraries(test_model_stage1 PRIVATE unity ggml)

add_executable(test_model_prediction
    tests/test_model_prediction.c
    dataset.c
    model.c
    utility.c
    )
target_link_libraries(test_model_prediction PRIVATE unity ggml)