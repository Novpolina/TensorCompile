#include <iostream>
#include <string>
#include <exception>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include <optional>
#include "llvm/TargetParser/Host.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"

#include "graph.hpp"
#include "onnx_parser.hpp"
#include "mlir_codegen.hpp"

#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

static cl::opt<std::string> InputFilename(
    cl::Positional, 
    cl::desc("<input onnx file>"), 
    cl::Required);

static cl::opt<std::string> OutputDot(
    "dot", 
    cl::desc("Output file for GraphViz dump"), 
    cl::init("graph.dot"));

static cl::opt<bool> DumpMLIR(
    "dump-mlir", 
    cl::desc("Print generated MLIR to stdout"), 
    cl::init(false));

int main(int argc, char** argv) {
    InitLLVM X(argc, argv);

    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    cl::ParseCommandLineOptions(argc, argv, "Easy Tensor Compiler (AOT)\n");

    std::cout << "Loading ONNX model from: " << InputFilename << "\n";

    try {
        compiler::ComputationGraph graph = compiler::ONNXParser::parse(InputFilename);
        std::cout << "Successfully parsed ONNX model. Nodes count: " << graph.nodes.size() << "\n";

        graph.dumpGraphViz(OutputDot);
        std::cout << "GraphViz representation saved to: " << OutputDot << "\n";

        mlir::MLIRContext context;
        compiler::MLIRCodegen codegen(context);
        mlir::OwningOpRef<mlir::ModuleOp> module = codegen.generate(graph);

        if (!module) {
            std::cerr << "Error: Failed to generate MLIR module.\n";
            return EXIT_FAILURE;
        }

        if (DumpMLIR) {
            std::cout << "\n--- Generated MLIR ---\n";
            module->dump();
            std::cout << "----------------------\n";
        }

        mlir::PassManager pm(module->getContext());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());

        if (mlir::failed(pm.run(*module))) {
            std::cerr << "Error: Failed to lower MLIR to LLVM dialect.\n";
            return EXIT_FAILURE;
        }

        std::cout << "\n--- Generated MLIR (LLVM Dialect) ---\n";
        module->dump();
        std::cout << "----------------------\n";

        mlir::DialectRegistry registry;
        mlir::registerLLVMDialectTranslation(registry);
        mlir::registerBuiltinDialectTranslation(registry);
        context.appendDialectRegistry(registry);
        llvm::LLVMContext llvmContext;
        std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
        if (!llvmModule) {
            std::cerr << "Error: Failed to translate MLIR to LLVM IR.\n";
            return EXIT_FAILURE;
        }

        std::cout << "Successfully translated to LLVM IR.\n";

        auto TargetTriple = llvm::sys::getDefaultTargetTriple();
        llvmModule->setTargetTriple(TargetTriple);

        std::string Error;
        auto Target = llvm::TargetRegistry::lookupTarget(TargetTriple, Error);
        if (!Target) {
            std::cerr << "Error matching target: " << Error << "\n";
            return EXIT_FAILURE;
        }

        llvm::TargetOptions opt;
        auto RM = std::optional<llvm::Reloc::Model>();
        auto TargetMachine = Target->createTargetMachine(TargetTriple, "generic", "", opt, RM);

        llvmModule->setDataLayout(TargetMachine->createDataLayout());

        std::string OutputFilename = "output_asm.s";
        std::error_code EC;
        llvm::raw_fd_ostream dest(OutputFilename, EC, llvm::sys::fs::OF_None);
        if (EC) {
            std::cerr << "Could not open file: " << EC.message() << "\n";
            return EXIT_FAILURE;
        }

        llvm::legacy::PassManager pass;
        auto FileType = llvm::CGFT_AssemblyFile;

        if (TargetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType)) {
            std::cerr << "TargetMachine can't emit a file of this type.\n";
            return EXIT_FAILURE;
        }

        pass.run(*llvmModule);
        dest.flush();

        std::cout << "Assembly successfully generated in: " << OutputFilename << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}