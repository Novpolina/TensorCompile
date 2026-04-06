cmake_minimum_required(VERSION 3.20)

# Название проекта берем из оригинального, но теперь это настоящий компилятор
project(TensorCompile LANGUAGES C CXX)

# Требование C++23 оставим, как было в вашем репозитории
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==========================================
# 1. Настройка LLVM и MLIR
# ==========================================
# Ищем пакет MLIR (он автоматически подтянет зависимости LLVM)
find_package(MLIR REQUIRED CONFIG)

# Добавляем пути к CMake-модулям LLVM и MLIR для использования их встроенных макросов
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

# Подключаем заголовочные файлы и определения LLVM/MLIR
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# ==========================================
# 2. Настройка Protobuf (для ONNX)
# ==========================================
find_package(Protobuf REQUIRED)

# Предполагается, что вы скачаете onnx.proto и положите его в корень проекта.
# CMake сам сгенерирует из него onnx.pb.h и onnx.pb.cc
set(ONNX_PROTO_FILE "${PROJECT_SOURCE_DIR}/onnx.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${ONNX_PROTO_FILE})

# ==========================================
# 3. Сборка ядра компилятора (Статическая библиотека)
# ==========================================
add_library(compiler_core STATIC
    src/graph.cpp
    src/onnx_parser.cpp
    src/mlir_codegen.cpp
    ${PROTO_SRCS} # Сгенерированный код ONNX
)

# Отключаем предупреждения о несовместимости старых заголовков LLVM с C++23
target_compile_options(compiler_core PRIVATE)

target_include_directories(compiler_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR} 
)

# Линкуем LLVM, MLIR и Protobuf к нашей библиотеке
target_link_libraries(compiler_core PUBLIC
    protobuf::libprotobuf
    
    # Базовые диалекты и инфраструктура
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRTosaDialect        # Диалект TOSA для тензорных операций
    MLIRSupport

    # Библиотеки для конвертации MLIR в LLVM IR
    MLIRTosaToLinalg
    MLIRLinalgTransforms
    MLIRTargetLLVMIRExport # Экспорт из MLIR в LLVM IR
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    MLIRFuncToLLVM   # <--- ДОБАВИТЬ
    MLIRArithToLLVM  # <--- ДОБАВИТЬ

    # Базовые библиотеки LLVM
    LLVMSupport
    LLVMCore
    LLVMExecutionEngine
    LLVMTarget
    
    # Кодогенератор ассемблера (включаем для x86 архитектуры)
    LLVMX86CodeGen
    LLVMX86AsmParser
)

# ==========================================
# 4. Сборка главного исполняемого файла
# ==========================================
add_executable(etc_compiler src/main.cpp)
target_link_libraries(etc_compiler PRIVATE compiler_core)

# ==========================================
# 5. Сборка тестов
# ==========================================
find_package(GTest REQUIRED)
enable_testing()
add_subdirectory(tests)