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

struct TransitionOpInfo {
  llvm::StringRef opName;
  TTNNLayoutAttr inputLayout;
  TTNNLayoutAttr outputLayout;
  TransitionOpIndex opIndex = TransitionOpIndex::Unknown;
};

struct GetTransitionOpsResult {
  bool isSuccess = false;
  llvm::SmallVector<TransitionOpInfo> transitionOps;
};

// Describes a layout transition for one exact SSA use so multi-result producer
// identity is preserved throughout analysis and materialization.
struct TransitionEdge {
  Value producerValue;
  Operation *consumerOp = nullptr;
  unsigned consumerOperandIndex = 0;
  TTNNLayoutAttr producerLayout;
  TTNNLayoutAttr consumerLayout;
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

  // Build and validate the transition sequence for one producer-consumer use.
  GetTransitionOpsResult getTransitionOps(Operation *producerOp,
                                          Value consumerOperand,
                                          Operation *consumerOp,
                                          TTNNLayoutAttr producerOutputLayout,
                                          TTNNLayoutAttr consumerInputLayout,
                                          uint64_t additionalL1Usage) const;

private:
  void resolveOpLayouts();
  void emitEdges();
  LogicalResult validateTransitions();

  std::optional<TTNNLayoutAttr> getRequiredLayout(Operation *op,
                                                  unsigned operandIndex) const;

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
