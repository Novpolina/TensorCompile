#pragma once

#include "graph.hpp"

namespace mlir {
    class MLIRContext;
    class ModuleOp;
    template <typename OpTy> class OwningOpRef;
}

namespace compiler {

class MLIRCodegen {
public:
    explicit MLIRCodegen(mlir::MLIRContext& context);

    mlir::OwningOpRef<mlir::ModuleOp> generate(const ComputationGraph& graph);

private:
    mlir::MLIRContext& context_;
};

} // namespace compiler