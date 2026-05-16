#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace compiler {
struct Attribute {
    std::vector<int64_t> ints;
    std::vector<float> floats;
    int64_t i_val = 0;
    float f_val = 0.0f;
    std::string s_val;
};

struct TensorInfo {
    std::vector<int64_t> shape;
};

class Node {
public:
    std::string name;    
    std::string op_type; 

    std::vector<std::string> inputs;
    std::vector<std::string> outputs; 

    std::unordered_map<std::string, Attribute> attributes;

    Node(std::string name, std::string op_type)
        : name(std::move(name)), op_type(std::move(op_type)) {}
};

class ComputationGraph {
public:
    std::vector<std::shared_ptr<Node>> nodes; 
    std::vector<std::string> graph_inputs;    
    std::vector<std::string> graph_outputs;
    
    std::unordered_map<std::string, TensorInfo> tensor_infos;

    void addNode(std::shared_ptr<Node> node) {
        nodes.push_back(std::move(node));
    }

    void dumpGraphViz(const std::string& filepath) const;
};

} // namespace compiler