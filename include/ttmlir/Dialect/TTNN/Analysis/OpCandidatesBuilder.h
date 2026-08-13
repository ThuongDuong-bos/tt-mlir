// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPCANDIDATESBUILDER_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPCANDIDATESBUILDER_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir::tt::ttnn {

// Coarse layout family used during candidate grouping.
enum class LayoutFamily : uint8_t {
  TileHeight = 0,
  TileBlock = 1,
  TileWidth = 2,
  RowMajorHeight = 3,
  RowMajorBlock = 4,
  RowMajorWidth = 5,
};

// Emission/processing groups assigned to candidates.
enum class CandidateGroup : uint8_t {
  DefaultDRAM = 0,
  DefaultL1 = 1,
  TileHeight = 2,
  TileBlock = 3,
  TileWidth = 4,
  RowMajorHeight = 5,
  RowMajorBlock = 6,
  RowMajorWidth = 7,
};

struct OpConfigCandidate {
  llvm::SmallVector<TTNNLayoutAttr> inputLayouts;
  OpConfig opConfig;
  std::optional<size_t> groupIndex;
};

struct OpCandidateBuilderResult {
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<OpConfigCandidate>>
      candidateMap;
};

struct PrunedGraphInfo {
  llvm::DenseMap<mlir::func::FuncOp, llvm::SmallVector<mlir::Operation *>>
      prunedSchedule;

  llvm::DenseMap<mlir::Operation *, size_t> fullOpIndex;
  llvm::DenseMap<mlir::Operation *, size_t> prunedOpIndex;

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      removedOpsAfterPrunedOp;
};

class OpCandidatesBuilder {
public:
  OpCandidatesBuilder() = default;

  void buildFullCandidates(
      const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
      const llvm::DenseMap<mlir::func::FuncOp,
                           llvm::SmallVector<mlir::Operation *>> &schedule,
      const llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>>
          &legalOpConfigs);

  void buildPrunedCandidates(
      const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
      const llvm::DenseMap<mlir::func::FuncOp,
                           llvm::SmallVector<mlir::Operation *>> &schedule,
      const llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>>
          &legalOpConfigs);

  const OpCandidateBuilderResult &getFullCandidates() const {
    return fullCandidates;
  }

  const OpCandidateBuilderResult &getPrunedCandidates() const {
    return prunedCandidates;
  }

  const PrunedGraphInfo &getPrunedGraphInfo() const {
    return prunedGraphInfo;
  }

private:
  PrunedGraphInfo makePrunedSubgraph(
      const llvm::DenseMap<mlir::func::FuncOp,
                           llvm::SmallVector<mlir::Operation *>> &schedule)
      const;

  OpCandidateBuilderResult buildCandidatesFromSchedule(
      const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
      const llvm::DenseMap<mlir::func::FuncOp,
                           llvm::SmallVector<mlir::Operation *>> &schedule,
      const llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>>
          &legalOpConfigs) const;

  OpCandidateBuilderResult fullCandidates;
  OpCandidateBuilderResult prunedCandidates;
  PrunedGraphInfo prunedGraphInfo;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPCANDIDATESBUILDER_H
