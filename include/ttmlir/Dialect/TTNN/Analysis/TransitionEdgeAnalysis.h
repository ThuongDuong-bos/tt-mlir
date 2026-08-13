#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_TRANSITIONEDGEANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_TRANSITIONEDGEANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttnn::analysis {

enum class TransitionOpIndex {
  ToLayout,
  Typecast,
  Reshape,
  Permute,
  Pad,
  Unknown,
};

struct TransitionOpInfo {
  llvm::StringRef opName;
  TTNNLayoutAttr inputLayout;
  TTNNLayoutAttr outputLayout;
  TransitionOpIndex opIndex = TransitionOpIndex::Unknown;
};

// Input to transition builder.
struct GetTransitionOpsInput {
  Value consumerOperand;
  TTNNLayoutAttr producerOutputLayout;
  TTNNLayoutAttr consumerInputLayout;
};

struct GetTransitionOpsResult {
  bool isSuccess = false;
  llvm::SmallVector<TransitionOpInfo> transitionOps;
};

// Describes a layout transition for one concrete SSA use. The exact producer
// Value is preserved instead of reconstructing it from a synthetic Edge.
struct TransitionEdge {
  Value producerValue;
  Operation *consumerOp;
  unsigned consumerOperandIndex;

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

  GetTransitionOpsResult
  getTransitionOps(Operation *producerOp, Value consumerOperand,
                  Operation *consumerOp,
                  TTNNLayoutAttr producerOutputLayout,
                  TTNNLayoutAttr consumerInputLayout,
                  uint64_t additionalL1Usage = 0) const;

private:
  const DenseMap<Operation *, OpConfig> &opConfigMap;
  const DenseMap<Operation *, llvm::SmallVector<TTNNLayoutAttr>>
      &inputLayoutsMap;
  const DenseMap<mlir::func::FuncOp, llvm::SmallVector<Operation *>>
      &opSchedule;

  llvm::SmallVector<TransitionEdge> transitionEdges;

  DenseMap<Value, TTNNLayoutAttr> valueLayoutMap;
  llvm::DenseSet<OpOperand *> emittedUses;

  // Phases.
  void resolveOpLayouts();
  void emitEdges();
  LogicalResult validateTransitions();

  // Helpers.

  // Return required layout for a given op input, if any.
  std::optional<TTNNLayoutAttr> getRequiredLayout(mlir::Operation *op,
                                                  unsigned operandIndex) const;

  // Collect existing layout transition ops.
  llvm::SmallVector<TransitionOpInfo> collectExistingOps(Value v) const;

  // Resolve the exact producer Value and the layout currently seen by the
  // consumer.
  std::optional<std::pair<Value, TTNNLayoutAttr>>
  resolveProducerAndLayout(Value value);

};

} // namespace mlir::tt::ttnn::analysis

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_TRANSITIONEDGEANALYSIS_H
