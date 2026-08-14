// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_TRANSITIONEDGEANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_TRANSITIONEDGEANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::tt::ttnn::analysis {

enum class TransitionOpIndex {
  ToLayout,
  Typecast,
  Reshape,
  Permute,
  Pad,
  Unknown,
};

/// Extra info for identifies conversion sequences.
struct TransitionOpInfo {
  llvm::StringRef opName;
  TTNNLayoutAttr inputLayout;
  TTNNLayoutAttr outputLayout;
  TransitionOpIndex opIndex = TransitionOpIndex::Unknown;
};

/// TransitionOpInfo list with a boolean flag.
struct GetTransitionOpsResult {
  bool isSuccess = false;
  llvm::SmallVector<TransitionOpInfo> transitionOps;
};

struct TransitionEdge {
  Value producerValue;
  Operation *consumerOp = nullptr;
  unsigned consumerOperandIndex = 0;
  TTNNLayoutAttr producerLayout;
  TTNNLayoutAttr consumerLayout;

  /// Planned transition ops from producer to consumer.
  llvm::SmallVector<TransitionOpInfo> transitionOps;
};

class TransitionEdgeAnalysis {
public:
  TransitionEdgeAnalysis(
      const llvm::DenseMap<Operation *, OpConfig> &opConfigMap,
      const llvm::DenseMap<Operation *, llvm::SmallVector<TTNNLayoutAttr>>
          &inputLayoutsMap,
      const llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>>
          &opSchedule)
      : opConfigMap(opConfigMap), inputLayoutsMap(inputLayoutsMap),
        opSchedule(opSchedule) {}

  LogicalResult run();

  const llvm::SmallVector<TransitionEdge> &getTransitionEdges() const {
    return transitionEdges;
  }

  /// Get transition ops from producerOp to consumerOp.
  GetTransitionOpsResult getTransitionOps(Operation *producerOp,
                                          Value consumerOperand,
                                          Operation *consumerOp,
                                          TTNNLayoutAttr producerOutputLayout,
                                          TTNNLayoutAttr consumerInputLayout,
                                          uint64_t additionalL1Usage) const;

private:
  /// Assign a concrete TTNNLayoutAttr to every Value in the graph
  /// and store it in valueLayoutMap.
  void resolveOpLayouts();

  /// Identify edges where:
  /// producer layout != expected layout
  void emitEdges();

  /// Ensure that all generated TransitionEdge entries are valid
  /// from an analysis perspective.
  /// Only validate internally consistent, not the execution valid.
  LogicalResult validateTransitions();

  /// Return required layout for a given op input (if any)
  std::optional<TTNNLayoutAttr> getRequiredLayout(Operation *op,
                                                  unsigned operandIndex) const;

  /// Return the producer op and its output layout for a given value.
  std::optional<std::pair<Value, TTNNLayoutAttr>>
  resolveProducerAndLayout(Value value);

  const llvm::DenseMap<Operation *, OpConfig> &opConfigMap;
  const llvm::DenseMap<Operation *, llvm::SmallVector<TTNNLayoutAttr>>
      &inputLayoutsMap;
  const llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>>
      &opSchedule;

  llvm::SmallVector<TransitionEdge> transitionEdges;
  llvm::DenseMap<Value, TTNNLayoutAttr> valueLayoutMap;
  llvm::DenseSet<OpOperand *> emittedUses;
};

} // namespace mlir::tt::ttnn::analysis

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_TRANSITIONEDGEANALYSIS_H
