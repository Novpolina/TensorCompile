#include "graph.hpp"
#include <gtest/gtest.h>

using namespace compiler;

TEST(ComputationGraph, AddNode) {
    ComputationGraph graph;
    auto node = std::make_shared<Node>("conv_layer_1", "Conv");
    
    node->inputs.push_back("input_tensor");
    node->outputs.push_back("conv_output");
    
    graph.addNode(node);

    EXPECT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0]->name, "conv_layer_1");
    EXPECT_EQ(graph.nodes[0]->op_type, "Conv");
    
    EXPECT_EQ(graph.nodes[0]->inputs.size(), 1);
    EXPECT_EQ(graph.nodes[0]->inputs[0], "input_tensor");
}

TEST(ComputationGraph, NodeAttributes) {
    auto node = std::make_shared<Node>("conv_layer", "Conv");
   
    Attribute strides_attr;
    strides_attr.ints = {2, 2};
    node->attributes["strides"] = strides_attr;

    Attribute alpha_attr;
    alpha_attr.f_val = 0.5f;
    node->attributes["alpha"] = alpha_attr;

    EXPECT_TRUE(node->attributes.count("strides") > 0);
    EXPECT_EQ(node->attributes["strides"].ints.size(), 2);
    EXPECT_EQ(node->attributes["strides"].ints[0], 2);
    
    EXPECT_FLOAT_EQ(node->attributes["alpha"].f_val, 0.5f);
}