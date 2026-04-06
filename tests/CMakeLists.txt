add_executable(compiler_ut 
    test_graph.cpp 
    test_onnx.cpp 
    test_mlir.cpp
)

target_compile_options(compiler_ut PRIVATE)

target_link_libraries(compiler_ut PRIVATE
    GTest::GTest
    GTest::gtest_main
    compiler_core 
)

target_include_directories(compiler_ut PUBLIC ${PROJECT_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(compiler_ut)