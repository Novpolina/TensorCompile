cmake_minimum_required(VERSION 3.20)

project(TensorCompile LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})


find_package(Protobuf REQUIRED)

set(ONNX_PROTO_FILE "${PROJECT_SOURCE_DIR}/onnx.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${ONNX_PROTO_FILE})

add_library(compiler_core STATIC
    src/graph.cpp
    src/onnx_parser.cpp
    src/mlir_codegen.cpp
    ${PROTO_SRCS}
)

target_compile_options(compiler_core PRIVATE)

target_include_directories(compiler_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR} 
)

target_link_libraries(compiler_core PUBLIC
    protobuf::libprotobuf
    
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFuncDialect
    MLIRArithDialect
    MLIRAffineToStandard
    MLIRMemRefTransforms
    MLIRTensorDialect
    MLIRTosaDialect
    MLIRSupport

    MLIRTosaToLinalg  
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    MLIRBufferizationTransforms
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRFuncTransforms
    MLIRArithTransforms
    MLIRLinalgTransforms
    MLIRReconcileUnrealizedCasts
    MLIRTensorTransforms
    MLIRMemRefToLLVM
    MLIRFuncToLLVM
    MLIRArithToLLVM

    LLVMSupport
    LLVMCore
    LLVMExecutionEngine
    LLVMTarget
    
    LLVMX86CodeGen
    LLVMX86AsmParser
)

add_executable(etc_compiler src/main.cpp)
target_link_libraries(etc_compiler PRIVATE compiler_core)

find_package(GTest REQUIRED)
enable_testing()
add_subdirectory(tests)