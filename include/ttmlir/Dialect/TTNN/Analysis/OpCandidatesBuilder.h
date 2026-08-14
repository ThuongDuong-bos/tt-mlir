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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir::tt::ttnn {

enum class LayoutFamily : uint8_t {
  TileHeight,
  TileBlock,
  TileWidth,
  RowMajorHeight,
  RowMajorBlock,
  RowMajorWidth,
};

enum class CandidateGroup : uint8_t {
  DefaultDRAM,
  DefaultL1,
  TileHeight,
  TileBlock,
  TileWidth,
  RowMajorHeight,
  RowMajorBlock,
  RowMajorWidth,
};

struct OpConfigCandidate {
  llvm::SmallVector<TTNNLayoutAttr> inputLayouts;
  OpConfig opConfig;
  std::optional<std::size_t> groupIndex;
};

struct OpCandidateBuilderResult {
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<OpConfigCandidate>>
      candidateMap;
};

struct PrunedGraphInfo {
  llvm::DenseMap<mlir::func::FuncOp, llvm::SmallVector<mlir::Operation *>>
      prunedSchedule;
  llvm::DenseMap<mlir::Operation *, std::size_t> fullOpIndex;
  llvm::DenseMap<mlir::Operation *, std::size_t> prunedOpIndex;
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

  const PrunedGraphInfo &getPrunedGraphInfo() const { return prunedGraphInfo; }

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
