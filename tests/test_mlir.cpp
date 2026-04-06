#include "mlir_codegen.hpp"
#include "graph.hpp"
#include <gtest/gtest.h>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace compiler;

TEST(MLIRCodegen, GenerateEmptyGraph) {
    mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
    context.loadDialect<mlir::func::FuncDialect>();
    ComputationGraph graph;
    graph.graph_inputs.push_back("input_1");
    graph.graph_outputs.push_back("input_1");

    MLIRCodegen codegen(context);
    mlir::OwningOpRef<mlir::ModuleOp> module = codegen.generate(graph);

    EXPECT_TRUE(module);

    EXPECT_FALSE(module->getBody()->empty());
}

TEST(MLIRCodegen, GenerateWithOneNode) {
    mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
    context.loadDialect<mlir::func::FuncDialect>();

    ComputationGraph graph;
    graph.graph_inputs = {"in_A", "in_B"};
    graph.graph_outputs = {"out_C"};

    auto node = std::make_shared<Node>("add_op", "Add");
    node->inputs = {"in_A", "in_B"};
    node->outputs = {"out_C"};
    graph.addNode(node);

    MLIRCodegen codegen(context);
    mlir::OwningOpRef<mlir::ModuleOp> module = codegen.generate(graph);

    EXPECT_TRUE(module);
    
}