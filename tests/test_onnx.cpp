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

TEST(ONNXParser, ParseSqueezeNet) {
    std::string model_path = "data/squeezenet.onnx"; 
    std::ifstream f(model_path);

    if (!f.good()) {
        model_path = "../data/squeezenet.onnx";
        f.open(model_path);
    }
    if (!f.good()) {
        model_path = "../../data/squeezenet.onnx";
        f.open(model_path);
    }

    if (!f.good()) {
        GTEST_SKIP() << "SqueezeNet file not found. Please run export_squeezenet.py first.";
    }
    f.close();

    compiler::ComputationGraph graph;
    EXPECT_NO_THROW({
        graph = compiler::ONNXParser::parse(model_path);
    });

    EXPECT_GT(graph.nodes.size(), 50) << "Graph should contain more than 50 nodes";


    EXPECT_EQ(graph.graph_inputs.size(), 1);
    EXPECT_EQ(graph.graph_outputs.size(), 1);

    bool has_conv = false;
    bool has_relu = false;
    bool has_maxpool = false;
    bool has_concat = false;
    bool has_avgpool = false;
    bool has_flatten = false;

    for (const auto& node : graph.nodes) {
        if (node->op_type == "Conv") has_conv = true;
        if (node->op_type == "Relu") has_relu = true;
        if (node->op_type == "MaxPool") has_maxpool = true;
        if (node->op_type == "Concat") has_concat = true;
        if (node->op_type == "GlobalAveragePool" || node->op_type == "AveragePool" || node->op_type == "ReduceMean") has_avgpool = true;
        if (node->op_type == "Flatten" || node->op_type == "Reshape" || node->op_type == "Squeeze") has_flatten = true;
    }

    EXPECT_TRUE(has_conv) << "Graph should contain at least one Conv node";
    EXPECT_TRUE(has_relu) << "Graph should contain at least one Relu node";
    EXPECT_TRUE(has_maxpool) << "Graph should contain at least one MaxPool node";
    EXPECT_TRUE(has_concat) << "Graph should contain at least one Concat node (Fire modules)";
    EXPECT_TRUE(has_avgpool) << "Graph should contain an AveragePool node";
    EXPECT_TRUE(has_flatten) << "Graph should contain a Flatten/Reshape node";
}