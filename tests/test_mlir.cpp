#include <gtest/gtest.h>
#include "mlir_codegen.hpp"
#include "graph.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"

TEST(MLIRCodegenTest, GenerateModuleFromGraph) {
    compiler::ComputationGraph graph;
    graph.graph_inputs.push_back("input_A");
    graph.graph_inputs.push_back("input_B");
    graph.graph_outputs.push_back("output_C");

    graph.tensor_infos["input_A"] = {{1, 16, 16, 16}};
    graph.tensor_infos["input_B"] = {{1, 16, 16, 16}};
    graph.tensor_infos["output_C"] = {{1, 16, 16, 16}};

    auto add_node = std::make_shared<compiler::Node>("add_node_1", "Add");
    add_node->inputs = {"input_A", "input_B"};
    add_node->outputs = {"output_C"};
    graph.addNode(std::move(add_node));

    mlir::MLIRContext context;
    compiler::MLIRCodegen codegen(context);
    
    mlir::OwningOpRef<mlir::ModuleOp> module = codegen.generate(graph);

    EXPECT_TRUE(module.get() != nullptr);
}