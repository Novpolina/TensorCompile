#include "mlir_codegen.hpp"

#include <iostream>
#include <unordered_map>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace compiler {

MLIRCodegen::MLIRCodegen(mlir::MLIRContext& context) : context_(context) {
    context_.loadDialect<mlir::func::FuncDialect>();
    context_.loadDialect<mlir::arith::ArithDialect>();
    context_.loadDialect<mlir::linalg::LinalgDialect>();
    context_.loadDialect<mlir::tensor::TensorDialect>();
}

mlir::OwningOpRef<mlir::ModuleOp> MLIRCodegen::generate(const ComputationGraph& graph) {
    mlir::OpBuilder builder(&context_);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());

    auto elementType = builder.getF32Type();

    auto getMLIRType = [&](const std::string& name) -> mlir::Type {
        if (graph.tensor_infos.count(name) && !graph.tensor_infos.at(name).shape.empty()) {
            return mlir::RankedTensorType::get(graph.tensor_infos.at(name).shape, elementType);
        }
        return mlir::UnrankedTensorType::get(elementType); 
    };

    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (const auto& in : graph.graph_inputs) {
        argTypes.push_back(getMLIRType(in));
    }

    llvm::SmallVector<mlir::Type, 4> retTypes;
    for (const auto& out : graph.graph_outputs) {
        retTypes.push_back(getMLIRType(out));
    }

    auto funcType = builder.getFunctionType(argTypes, retTypes);
    
    
    auto mainFunc = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "forward", funcType);
    mainFunc->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
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
                mlir::Type tensorType = getMLIRType(in_name);
                
                if (auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(tensorType)) {
                    auto attr = mlir::DenseElementsAttr::get(rankedType, 1.0f);
                    auto constOp = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), attr);
                    valueMap[in_name] = constOp.getResult();
                } else {
                    auto constOp = builder.create<mlir::arith::ConstantFloatOp>(
                        builder.getUnknownLoc(), llvm::APFloat(1.0f), elementType);
                    valueMap[in_name] = constOp.getResult();
                }
            }
            opInputs.push_back(valueMap[in_name]);
        }

 if (node->op_type == "Add" || node->op_type == "Mul") {
            auto resultType = getMLIRType(node->outputs[0]);
            int64_t rank = resultType.cast<mlir::RankedTensorType>().getRank();
            
            auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
                builder.getUnknownLoc(), resultType.cast<mlir::RankedTensorType>().getShape(), elementType);

            llvm::SmallVector<mlir::AffineMap, 3> maps(3, builder.getMultiDimIdentityMap(rank));
            llvm::SmallVector<mlir::utils::IteratorType, 3> iterTypes(rank, mlir::utils::IteratorType::parallel);

            auto genericOp = builder.create<mlir::linalg::GenericOp>(
                builder.getUnknownLoc(),
                mlir::TypeRange{resultType},
                mlir::ValueRange{opInputs[0], opInputs[1]},
                mlir::ValueRange{emptyTensor.getResult()},
                maps, iterTypes,
                [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
                    mlir::Value res = (node->op_type == "Add") 
                        ? b.create<mlir::arith::AddFOp>(loc, args[0], args[1]).getResult()
                        : b.create<mlir::arith::MulFOp>(loc, args[0], args[1]).getResult();
                    b.create<mlir::linalg::YieldOp>(loc, res);
                }
            );
            valueMap[node->outputs[0]] = genericOp.getResult(0);
        }
        else if (node->op_type == "Relu") {
            auto resultType = getMLIRType(node->outputs[0]);
            int64_t rank = resultType.cast<mlir::RankedTensorType>().getRank();
            
            auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
                builder.getUnknownLoc(), resultType.cast<mlir::RankedTensorType>().getShape(), elementType);

            llvm::SmallVector<mlir::AffineMap, 2> maps(2, builder.getMultiDimIdentityMap(rank));
            llvm::SmallVector<mlir::utils::IteratorType, 2> iterTypes(rank, mlir::utils::IteratorType::parallel);

            auto genericOp = builder.create<mlir::linalg::GenericOp>(
                builder.getUnknownLoc(),
                mlir::TypeRange{resultType},
                mlir::ValueRange{opInputs[0]},
                mlir::ValueRange{emptyTensor.getResult()},
                maps, iterTypes,
                [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
                    mlir::Value zero = b.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(0.0f), elementType);
                    mlir::Value res = b.create<mlir::arith::MaxFOp>(loc, args[0], zero);
                    b.create<mlir::linalg::YieldOp>(loc, res);
                }
            );
            valueMap[node->outputs[0]] = genericOp.getResult(0);
        }

        else if (node->op_type == "Flatten" || node->op_type == "Reshape") {
            auto resultType = getMLIRType(node->outputs[0]);
            
            llvm::SmallVector<mlir::ReassociationIndices, 2> reassoc(2);
            reassoc[0].push_back(0); // Batch
            reassoc[1].push_back(1); // Channels
            reassoc[1].push_back(2); // Height
            reassoc[1].push_back(3); // Width
            
            auto collapseOp = builder.create<mlir::tensor::CollapseShapeOp>(
                builder.getUnknownLoc(), resultType, opInputs[0], reassoc);
                
            valueMap[node->outputs[0]] = collapseOp.getResult();
        }
else if (node->op_type == "Conv") {
            auto resultType = getMLIRType(node->outputs[0]);
            
            std::vector<int64_t> strides = {1, 1};
            if (node->attributes.count("strides")) strides = node->attributes.at("strides").ints;
            
            std::vector<int64_t> dilations = {1, 1};
            if (node->attributes.count("dilations")) dilations = node->attributes.at("dilations").ints;

            auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
                builder.getUnknownLoc(), resultType.cast<mlir::RankedTensorType>().getShape(), elementType);
            
            auto convOp = builder.create<mlir::linalg::Conv2DNchwFchwOp>(
                builder.getUnknownLoc(),
                mlir::TypeRange{resultType},
                mlir::ValueRange{opInputs[0], opInputs[1]}, // Вход и Веса
                mlir::ValueRange{emptyTensor.getResult()},
                builder.getDenseI64ArrayAttr(strides),
                builder.getDenseI64ArrayAttr(dilations)
            );
            
            mlir::Value finalResult = convOp.getResult(0);
            valueMap[node->outputs[0]] = finalResult;
        }
        else if (node->op_type == "MatMul") {
            auto resultType = getMLIRType(node->outputs[0]);
            auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
                builder.getUnknownLoc(), resultType.cast<mlir::RankedTensorType>().getShape(), elementType);
                
            auto matmulOp = builder.create<mlir::linalg::MatmulOp>(
                builder.getUnknownLoc(),
                mlir::TypeRange{resultType},
                mlir::ValueRange{opInputs[0], opInputs[1]},
                mlir::ValueRange{emptyTensor.getResult()}
            );
            valueMap[node->outputs[0]] = matmulOp.getResult(0);
        }
        else if (node->op_type == "Gemm") {
            auto resultType = getMLIRType(node->outputs[0]);
            
            int64_t transA = 0, transB = 0;
            if (node->attributes.count("transA")) transA = node->attributes.at("transA").i_val;
            if (node->attributes.count("transB")) transB = node->attributes.at("transB").i_val;

            mlir::Value aVal = opInputs[0];
            mlir::Value bVal = opInputs[1];
            
            if (transA) {
                auto typeA = aVal.getType().cast<mlir::RankedTensorType>();
                auto shapeA = typeA.getShape();
                auto transTypeA = mlir::RankedTensorType::get({shapeA[1], shapeA[0]}, elementType);
                auto emptyA = builder.create<mlir::tensor::EmptyOp>(builder.getUnknownLoc(), transTypeA.getShape(), elementType);
                
                aVal = builder.create<mlir::linalg::TransposeOp>(
                    builder.getUnknownLoc(), aVal, emptyA.getResult(), builder.getDenseI64ArrayAttr({1, 0})).getResult()[0];
            }
            if (transB) {
                auto typeB = bVal.getType().cast<mlir::RankedTensorType>();
                auto shapeB = typeB.getShape();
                auto transTypeB = mlir::RankedTensorType::get({shapeB[1], shapeB[0]}, elementType);
                auto emptyB = builder.create<mlir::tensor::EmptyOp>(builder.getUnknownLoc(), transTypeB.getShape(), elementType);
                
                bVal = builder.create<mlir::linalg::TransposeOp>(
                    builder.getUnknownLoc(), bVal, emptyB.getResult(), builder.getDenseI64ArrayAttr({1, 0})).getResult()[0];
            }

            auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
                builder.getUnknownLoc(), resultType.cast<mlir::RankedTensorType>().getShape(), elementType);
                
            auto matmulOp = builder.create<mlir::linalg::MatmulOp>(
                builder.getUnknownLoc(),
                mlir::TypeRange{resultType},
                mlir::ValueRange{aVal, bVal},
                mlir::ValueRange{emptyTensor.getResult()}
            );
            
            valueMap[node->outputs[0]] = matmulOp.getResult(0);
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