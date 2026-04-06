#include <gtest/gtest.h>
#include <fstream>
#include <stdexcept>
#include <algorithm>

#include "graph.hpp"
#include "onnx_parser.hpp" 

TEST(ONNXParser, FileNotFound) {
    EXPECT_THROW({
        compiler::ONNXParser::parse("some_random_missing_file_123.onnx");
    }, std::runtime_error);
}

TEST(ONNXParser, ParseFullModel) {
    const std::string model_path = "data/full_model.onnx"; 
    
    std::ifstream f(model_path);
    if (!f.good()) {
        GTEST_SKIP() << "Model file " << model_path << " not found. Please run generate_onnx.py first.";
    }
    f.close();

    compiler::ComputationGraph graph;
    EXPECT_NO_THROW({
        graph = compiler::ONNXParser::parse(model_path);
    });

    EXPECT_EQ(graph.nodes.size(), 7);

    EXPECT_EQ(graph.graph_inputs.size(), 3);
   
    EXPECT_EQ(graph.graph_outputs.size(), 1);

    bool has_conv = false;
    bool has_relu = false;
    bool has_matmul = false;

    for (const auto& node : graph.nodes) {
        if (node->op_type == "Conv") has_conv = true;
        if (node->op_type == "Relu") has_relu = true;
        if (node->op_type == "MatMul") has_matmul = true;
    }

    EXPECT_TRUE(has_conv) << "Graph should contain a Conv node";
    EXPECT_TRUE(has_relu) << "Graph should contain a Relu node";
    EXPECT_TRUE(has_matmul) << "Graph should contain a MatMul node";
}