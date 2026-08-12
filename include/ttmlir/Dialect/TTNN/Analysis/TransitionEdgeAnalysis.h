#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_TRANSITIONEDGEANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_TRANSITIONEDGEANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttnn::analysis {

struct TransitionOpInfo {
  llvm::StringRef opName;
  TTNNLayoutAttr inputLayout;
  TTNNLayoutAttr outputLayout;
};

// Input to transition builder
struct GetTransitionOpsInput {
  Operation *producerOp;
  Operation *consumerOp;
  Value producerValue;
  unsigned consumerOperandIndex;

  TTNNLayoutAttr producerOutputLayout;
  TTNNLayoutAttr consumerInputLayout;
};

struct TransitionEdge {
  Edge edge;

  TTNNLayoutAttr producerLayout;
  TTNNLayoutAttr consumerLayout;

  llvm::SmallVector<TransitionOpInfo> transitionOps;
};

class TransitionEdgeAnalysis {
public:
  TransitionEdgeAnalysis(
      const DenseMap<Operation *, OpConfig> &opConfigMap,
      const DenseMap<Operation *, llvm::SmallVector<TTNNLayoutAttr>>
          &inputLayoutsMap,
      const DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &opSchedule)
      : opConfigMap(opConfigMap), inputLayoutsMap(inputLayoutsMap),
        opSchedule(opSchedule) {}

  LogicalResult run();

  const llvm::SmallVector<TransitionEdge> &getTransitionEdges() const {
    return transitionEdges;
  }

private:
  const DenseMap<Operation *, OpConfig> &opConfigMap;
  const DenseMap<Operation *, llvm::SmallVector<TTNNLayoutAttr>>
      &inputLayoutsMap;
  const DenseMap<mlir::func::FuncOp, llvm::SmallVector<Operation *>>
      &opSchedule;

  llvm::SmallVector<TransitionEdge> transitionEdges;

  DenseMap<Value, TTNNLayoutAttr> valueLayoutMap;
  llvm::DenseSet<Edge> emittedEdges; 

  // phases
  void resolveOpLayouts();
  void emitEdges();
  LogicalResult validateTransitions();

  // helpers

  // Return required layout for a given op input (if any)
  std::optional<TTNNLayoutAttr> getRequiredLayout(mlir::Operation *op,
                                                  unsigned operandIndex) const;

  // Collect existing layout transition ops
  llvm::SmallVector<TransitionOpInfo> collectExistingOps(Value v) const;

  // Describe transition sequence (analysis only)
  llvm::SmallVector<TransitionOpInfo>
  getTransitionOps(const GetTransitionOpsInput &input) const;

  // resolve producer op and layout helper
  std::optional<std::pair<Operation *, TTNNLayoutAttr>> resolveProducerAndLayout(Value v);

};

} // namespace mlir::tt::ttnn::analysis

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_TRANSITIONEDGEANALYSIS_H