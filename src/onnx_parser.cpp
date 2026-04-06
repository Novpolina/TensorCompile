#include "onnx_parser.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "onnx.pb.h" 

namespace compiler {

ComputationGraph ONNXParser::parse(const std::string& filepath) {
    ComputationGraph graph;

    std::ifstream input_stream(filepath, std::ios::in | std::ios::binary);
    if (!input_stream) {
        throw std::runtime_error("ONNXParser: Failed to open file " + filepath);
    }

    onnx::ModelProto model;
    if (!model.ParseFromIstream(&input_stream)) {
        throw std::runtime_error("ONNXParser: Failed to parse ONNX protobuf");
    }

    const onnx::GraphProto& onnx_graph = model.graph();

    for (int i = 0; i < onnx_graph.input_size(); ++i) {
        graph.graph_inputs.push_back(onnx_graph.input(i).name());
    }

    for (int i = 0; i < onnx_graph.output_size(); ++i) {
        graph.graph_outputs.push_back(onnx_graph.output(i).name());
    }

    for (int i = 0; i < onnx_graph.node_size(); ++i) {
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);

        auto node = std::make_shared<Node>(onnx_node.name(), onnx_node.op_type());

        for (int j = 0; j < onnx_node.input_size(); ++j) {
            node->inputs.push_back(onnx_node.input(j));
        }

        for (int j = 0; j < onnx_node.output_size(); ++j) {
            node->outputs.push_back(onnx_node.output(j));
        }

        for (int j = 0; j < onnx_node.attribute_size(); ++j) {
            const onnx::AttributeProto& attr = onnx_node.attribute(j);
            Attribute custom_attr;

            switch (attr.type()) {
                case onnx::AttributeProto::INT:
                    custom_attr.i_val = attr.i();
                    break;
                case onnx::AttributeProto::INTS:
                    for (int k = 0; k < attr.ints_size(); ++k) {
                        custom_attr.ints.push_back(attr.ints(k));
                    }
                    break;
                case onnx::AttributeProto::FLOAT:
                    custom_attr.f_val = attr.f();
                    break;
                case onnx::AttributeProto::FLOATS:
                    for (int k = 0; k < attr.floats_size(); ++k) {
                        custom_attr.floats.push_back(attr.floats(k));
                    }
                    break;
                case onnx::AttributeProto::STRING:
                    custom_attr.s_val = attr.s();
                    break;
                default:
                    break;
            }
            node->attributes[attr.name()] = custom_attr;
        }

        graph.addNode(std::move(node));
    }

    return graph;
}

} // namespace compiler