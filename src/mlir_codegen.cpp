#include "mlir_codegen.hpp"

#include <iostream>
#include <unordered_map>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace compiler {

MLIRCodegen::MLIRCodegen(mlir::MLIRContext& context) : context_(context) {
    context_.loadDialect<mlir::func::FuncDialect>();
    context_.loadDialect<mlir::arith::ArithDialect>();
}

mlir::OwningOpRef<mlir::ModuleOp> MLIRCodegen::generate(const ComputationGraph& graph) {
    mlir::OpBuilder builder(&context_);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());

    auto elementType = builder.getF32Type();

    llvm::SmallVector<mlir::Type, 4> argTypes(graph.graph_inputs.size(), elementType);
    llvm::SmallVector<mlir::Type, 4> retTypes(graph.graph_outputs.size(), elementType);
    auto funcType = builder.getFunctionType(argTypes, retTypes);
    
    auto mainFunc = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", funcType);
    mlir::Block* entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);

    std::unordered_map<std::string, mlir::Value> valueMap;

    for (size_t i = 0; i < graph.graph_inputs.size(); ++i) {
        valueMap[graph.graph_inputs[i]] = entryBlock->getArgument(i);
    }

    for (const auto& node : graph.nodes) {
        llvm::SmallVector<mlir::Value, 4> opInputs;
        for (const auto& in_name : node->inputs) {
            if (valueMap.find(in_name) == valueMap.end()) {
                auto constOp = builder.create<mlir::arith::ConstantFloatOp>(
                    builder.getUnknownLoc(), llvm::APFloat(1.0f), elementType);
                valueMap[in_name] = constOp.getResult();
            }
            opInputs.push_back(valueMap[in_name]);
        }

        if (node->op_type == "Add") {
            auto addOp = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), opInputs[0], opInputs[1]);
            valueMap[node->outputs[0]] = addOp.getResult();
        } 
        else if (node->op_type == "Mul") {
            auto mulOp = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), opInputs[0], opInputs[1]);
            valueMap[node->outputs[0]] = mulOp.getResult();
        }
        else if (node->op_type == "Relu") {
            auto zeroOp = builder.create<mlir::arith::ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(0.0f), elementType);
            auto maxOp = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), opInputs[0], zeroOp.getResult());
            valueMap[node->outputs[0]] = maxOp.getResult();
        }
        else {
            valueMap[node->outputs[0]] = opInputs[0];
        }
    }

    llvm::SmallVector<mlir::Value, 4> returnValues;
    for (const auto& out_name : graph.graph_outputs) {
        if (valueMap.find(out_name) != valueMap.end()) {
            returnValues.push_back(valueMap[out_name]);
        }
    }

    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), returnValues);
    
    return module;
}

} // namespace compiler