#include "graph.hpp"

#include <fstream>
#include <iostream>

namespace compiler {

void ComputationGraph::dumpGraphViz(const std::string& filepath) const {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << " for writing.\n";
        return;
    }

    out << "digraph ComputationGraph {\n";
    out << "  node [shape=box, style=filled, fillcolor=lightblue, fontname=\"Helvetica\"];\n";

    for (const auto& node : nodes) {
        out << "  \"" << node->name << "\" [label=\"" << node->op_type << "\\n(" 
            << node->name << ")\"];\n";
    }

    for (const auto& node : nodes) {
        for (const auto& input_name : node->inputs) {
            out << "  \"" << input_name << "\" -> \"" << node->name << "\";\n";
        }

        for (const auto& output_name : node->outputs) {
            out << "  \"" << node->name << "\" -> \"" << output_name << "\";\n";
            out << "  \"" << output_name << "\" [shape=ellipse];\n"; 
        }
    }

    for (const auto& in : graph_inputs) {
        out << "  \"" << in << "\" [shape=ellipse, fillcolor=lightgreen];\n";
    }
    for (const auto& out_name : graph_outputs) {
        out << "  \"" << out_name << "\" [shape=ellipse, fillcolor=lightcoral];\n";
    }

    out << "}\n";
}

} // namespace compiler