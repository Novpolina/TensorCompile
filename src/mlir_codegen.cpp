#include "mlir_codegen.hpp"

#include <iostream>
#include <unordered_map>
#include <limits>

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
                    auto attr = mlir::DenseElementsAttr::get(rankedType, 0.01f);
                    auto constOp = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), attr);
                    valueMap[in_name] = constOp.getResult();
                } else {
                    auto constOp = builder.create<mlir::arith::ConstantFloatOp>(
                        builder.getUnknownLoc(), llvm::APFloat(0.01f), elementType);
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

        else if (node->op_type == "Flatten" || node->op_type == "Reshape" || node->op_type == "Squeeze") {
            auto resultType = getMLIRType(node->outputs[0]).cast<mlir::RankedTensorType>();
            auto inputType = opInputs[0].getType().cast<mlir::RankedTensorType>();
            
            if (inputType.getRank() == resultType.getRank()) {
                valueMap[node->outputs[0]] = opInputs[0];
            } else {
                llvm::SmallVector<mlir::ReassociationIndices, 2> reassoc(resultType.getRank());
                reassoc[0].push_back(0);
                for (int i = 1; i < inputType.getRank(); ++i) {
                    reassoc[1].push_back(i);
                }
                auto collapseOp = builder.create<mlir::tensor::CollapseShapeOp>(
                    builder.getUnknownLoc(), resultType, opInputs[0], reassoc);
                    
                valueMap[node->outputs[0]] = collapseOp.getResult();
            }
        }
else if (node->op_type == "Conv") {
            auto resultType = getMLIRType(node->outputs[0]);
            
            std::vector<int64_t> strides = {1, 1};
            if (node->attributes.count("strides")) strides = node->attributes.at("strides").ints;
            
            std::vector<int64_t> dilations = {1, 1};
            if (node->attributes.count("dilations")) dilations = node->attributes.at("dilations").ints;

            mlir::Value convInput = opInputs[0];
            if (node->attributes.count("pads")) {
                std::vector<int64_t> pads = node->attributes.at("pads").ints;
                if (pads.size() >= 4 && (pads[0] > 0 || pads[1] > 0 || pads[2] > 0 || pads[3] > 0)) {
                    auto inputType = convInput.getType().cast<mlir::RankedTensorType>();
                    auto inShape = inputType.getShape();
                    
                    llvm::SmallVector<int64_t, 4> padShape = {
                        inShape[0], inShape[1], 
                        inShape[2] + pads[0] + pads[2], 
                        inShape[3] + pads[1] + pads[3]
                    };
                    auto paddedType = mlir::RankedTensorType::get(padShape, elementType);
                    
                    llvm::SmallVector<mlir::OpFoldResult, 4> lowPads = {
                        builder.getIndexAttr(0), builder.getIndexAttr(0),
                        builder.getIndexAttr(pads[0]), builder.getIndexAttr(pads[1])
                    };
                    llvm::SmallVector<mlir::OpFoldResult, 4> highPads = {
                        builder.getIndexAttr(0), builder.getIndexAttr(0),
                        builder.getIndexAttr(pads[2]), builder.getIndexAttr(pads[3])
                    };
                    
                    mlir::Value zeroVal = builder.create<mlir::arith::ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(0.0f), elementType);
                    auto padOp = builder.create<mlir::tensor::PadOp>(
                        builder.getUnknownLoc(), paddedType, convInput, lowPads, highPads, zeroVal);
                    convInput = padOp.getResult();
                }
            }

            auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
                builder.getUnknownLoc(), resultType.cast<mlir::RankedTensorType>().getShape(), elementType);
            
            auto zeroValConv = builder.create<mlir::arith::ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(0.0f), elementType);
            auto initTensorConv = builder.create<mlir::linalg::FillOp>(
                builder.getUnknownLoc(),
                mlir::ValueRange{zeroValConv.getResult()},
                mlir::ValueRange{emptyTensor.getResult()}
            ).getResult(0);

            auto convOp = builder.create<mlir::linalg::Conv2DNchwFchwOp>(
                builder.getUnknownLoc(),
                mlir::TypeRange{resultType},
                mlir::ValueRange{convInput, opInputs[1]}, 
                mlir::ValueRange{initTensorConv},
                builder.getDenseI64ArrayAttr(strides),
                builder.getDenseI64ArrayAttr(dilations)
            );
            
            valueMap[node->outputs[0]] = convOp.getResult(0);
        }
        else if (node->op_type == "Concat") {
            auto resultType = getMLIRType(node->outputs[0]).cast<mlir::RankedTensorType>();
            int64_t axis = 1;
            if (node->attributes.count("axis")) axis = node->attributes.at("axis").i_val;
            
            auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
                builder.getUnknownLoc(), resultType.getShape(), elementType);
            
            mlir::Value result = emptyTensor.getResult();
            int64_t current_offset = 0;
            
            for (mlir::Value in_val : opInputs) {
                auto inType = in_val.getType().cast<mlir::RankedTensorType>();
                auto inShape = inType.getShape();
                
                llvm::SmallVector<mlir::OpFoldResult, 4> offsets, sizes, strides;
                for (size_t i = 0; i < inShape.size(); ++i) {
                    if (static_cast<int64_t>(i) == axis) {
                        offsets.push_back(builder.getIndexAttr(current_offset));
                        sizes.push_back(builder.getIndexAttr(inShape[i]));
                    } else {
                        offsets.push_back(builder.getIndexAttr(0));
                        sizes.push_back(builder.getIndexAttr(inShape[i]));
                    }
                    strides.push_back(builder.getIndexAttr(1));
                }
                
                result = builder.create<mlir::tensor::InsertSliceOp>(
                    builder.getUnknownLoc(), in_val, result, offsets, sizes, strides);
                    
                current_offset += inShape[axis];
            }
            valueMap[node->outputs[0]] = result;
        }
        else if (node->op_type == "MaxPool") {
            auto resultType = getMLIRType(node->outputs[0]);
            
            std::vector<int64_t> kernel_shape = {3, 3};
            if (node->attributes.count("kernel_shape")) kernel_shape = node->attributes.at("kernel_shape").ints;
            
            std::vector<int64_t> strides = {1, 1};
            if (node->attributes.count("strides")) strides = node->attributes.at("strides").ints;
            
            std::vector<int64_t> dilations = {1, 1};
            if (node->attributes.count("dilations")) dilations = node->attributes.at("dilations").ints;
            
            auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
                builder.getUnknownLoc(), resultType.cast<mlir::RankedTensorType>().getShape(), elementType);

            auto windowShape = mlir::RankedTensorType::get({kernel_shape[0], kernel_shape[1]}, elementType);
            auto windowEmpty = builder.create<mlir::tensor::EmptyOp>(builder.getUnknownLoc(), windowShape.getShape(), elementType);

            auto minAttr = builder.getFloatAttr(elementType, -std::numeric_limits<float>::infinity());
            auto minOp = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), minAttr);
            auto initTensor = builder.create<mlir::linalg::FillOp>(
                builder.getUnknownLoc(), 
                mlir::ValueRange{minOp.getResult()}, 
                mlir::ValueRange{emptyTensor.getResult()}
            ).getResult(0);
            auto poolOp = builder.create<mlir::linalg::PoolingNchwMaxOp>(
                builder.getUnknownLoc(), 
                mlir::TypeRange{resultType},
                mlir::ValueRange{opInputs[0], windowEmpty},
                mlir::ValueRange{initTensor},
                builder.getDenseI64ArrayAttr(strides),
                builder.getDenseI64ArrayAttr(dilations)
            );
            valueMap[node->outputs[0]] = poolOp.getResult(0);
        }
else if (node->op_type == "GlobalAveragePool" || node->op_type == "AveragePool" || node->op_type == "ReduceMean") {
            auto inputType = opInputs[0].getType().cast<mlir::RankedTensorType>();
            auto resultType = getMLIRType(node->outputs[0]).cast<mlir::RankedTensorType>();
            
            int64_t H = inputType.getShape()[2];
            int64_t W = inputType.getShape()[3];
            float hw_float = static_cast<float>(H * W);
            
            auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
                builder.getUnknownLoc(), resultType.getShape(), elementType);
            
            auto zeroAttr = builder.getFloatAttr(elementType, 0.0f);
            auto zeroOp = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), zeroAttr);
            auto zeroTensor = builder.create<mlir::linalg::FillOp>(
                builder.getUnknownLoc(), 
                mlir::ValueRange{zeroOp.getResult()}, 
                mlir::ValueRange{emptyTensor.getResult()}
            ).getResult(0);

            llvm::SmallVector<mlir::AffineMap, 2> maps;
            maps.push_back(builder.getMultiDimIdentityMap(4)); 
            
            llvm::SmallVector<mlir::AffineExpr, 4> outExprs;
            outExprs.push_back(builder.getAffineDimExpr(0)); // Batch
            outExprs.push_back(builder.getAffineDimExpr(1)); // Channels
            if (resultType.getRank() == 4) {
                outExprs.push_back(builder.getAffineConstantExpr(0)); // Height = 1
                outExprs.push_back(builder.getAffineConstantExpr(0)); // Width = 1
            }
            maps.push_back(mlir::AffineMap::get(4, 0, outExprs, &context_));
            
            llvm::SmallVector<mlir::utils::IteratorType, 4> iterTypes = {
                mlir::utils::IteratorType::parallel,
                mlir::utils::IteratorType::parallel,
                mlir::utils::IteratorType::reduction,
                mlir::utils::IteratorType::reduction
            };
            
            auto genericOp = builder.create<mlir::linalg::GenericOp>(
                builder.getUnknownLoc(),
                mlir::TypeRange{resultType},
                mlir::ValueRange{opInputs[0]},
                mlir::ValueRange{zeroTensor},
                maps, iterTypes,
                [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
                    mlir::Value sum = b.create<mlir::arith::AddFOp>(loc, args[0], args[1]);
                    b.create<mlir::linalg::YieldOp>(loc, sum);
                }
            );
            
            auto hwConst = builder.create<mlir::arith::ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(hw_float), elementType);
            llvm::SmallVector<mlir::AffineMap, 2> divMaps(2, builder.getMultiDimIdentityMap(resultType.getRank()));
            llvm::SmallVector<mlir::utils::IteratorType, 4> divIter(resultType.getRank(), mlir::utils::IteratorType::parallel);
            
            auto divEmpty = builder.create<mlir::tensor::EmptyOp>(builder.getUnknownLoc(), resultType.getShape(), elementType);
            
            auto divOp = builder.create<mlir::linalg::GenericOp>(
                builder.getUnknownLoc(),
                mlir::TypeRange{resultType},
                mlir::ValueRange{genericOp.getResult(0)},
                mlir::ValueRange{divEmpty},
                divMaps, divIter,
                [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
                    mlir::Value div = b.create<mlir::arith::DivFOp>(loc, args[0], hwConst);
                    b.create<mlir::linalg::YieldOp>(loc, div);
                }
            );
            valueMap[node->outputs[0]] = divOp.getResult(0);
        }
        else if (node->op_type == "Dropout") {
            valueMap[node->outputs[0]] = opInputs[0]; 
        }
        else {
            std::cerr << "WARNING: Unsupported ONNX operation '" << node->op_type << "'! Falling back to input tensor.\n";
            valueMap[node->outputs[0]] = opInputs[0];
        }
    } // <-- ЗДЕСЬ ЗАКРЫВАЕТСЯ ЦИКЛ FOR

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