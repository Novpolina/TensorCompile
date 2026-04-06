#pragma once

#include <string>
#include "graph.hpp"

namespace compiler {

class ONNXParser {
public:
    static ComputationGraph parse(const std::string& filepath);
};

} // namespace compiler