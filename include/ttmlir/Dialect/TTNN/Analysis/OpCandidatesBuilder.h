// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {

struct OpConfigCandidate {
  llvm::SmallVector<TTNNLayoutAttr> inputLayouts;
  OpConfig opConfig;
};

struct OpCandidateBuilderResult {
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<OpConfigCandidate>>
      candidateMap;
};

class OpCandidatesBuilder {
public:
  OpCandidateBuilderResult buildCandidatesFromTensorLayouts(
      const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
      const llvm::DenseMap<mlir::func::FuncOp,
                           llvm::SmallVector<mlir::Operation *>> &schedule,
      const llvm::DenseMap<mlir::Operation *, OpConfig> &opConfigMap) const;

private:
  llvm::SmallVector<TTNNLayoutAttr> extractTensorLayouts(
      const mlir::RankedTensorType tensorType,
      const TensorTypeLayoutsMap &tensorTypePossibleLayouts) const;

  llvm::SmallVector<TTNNLayoutAttr> extractInputLayouts(
      mlir::Operation *op,
      const TensorTypeLayoutsMap &tensorTypePossibleLayouts) const;
};

} // namespace mlir::tt::ttnn