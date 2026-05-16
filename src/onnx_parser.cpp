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

    auto extract_shape = [](const onnx::ValueInfoProto& info) {
        TensorInfo t_info;
        if (info.has_type() && info.type().has_tensor_type() && info.type().tensor_type().has_shape()) {
            const auto& shape = info.type().tensor_type().shape();
            for (int i = 0; i < shape.dim_size(); ++i) {
                t_info.shape.push_back(shape.dim(i).dim_value());
            }
        }
        return t_info;
    };

    for (int i = 0; i < onnx_graph.input_size(); ++i) {
        const auto& input = onnx_graph.input(i);
        graph.graph_inputs.push_back(input.name());
        graph.tensor_infos[input.name()] = extract_shape(input);
    }

    for (int i = 0; i < onnx_graph.output_size(); ++i) {
        const auto& output = onnx_graph.output(i);
        graph.graph_outputs.push_back(output.name());
        graph.tensor_infos[output.name()] = extract_shape(output);
    }

    for (int i = 0; i < onnx_graph.value_info_size(); ++i) {
        const auto& val_info = onnx_graph.value_info(i);
        graph.tensor_infos[val_info.name()] = extract_shape(val_info);
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