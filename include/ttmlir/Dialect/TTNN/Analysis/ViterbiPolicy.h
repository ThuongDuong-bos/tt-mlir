// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpCandidatesBuilder.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace mlir::tt::ttnn {

// TODO: CostModel
class CostModel;

// Define lifetime information
struct TensorLifetime {
  size_t startOpIdx; // Index of the operation, combine with schedule to get
                     // actual operation
  size_t endOpIdx;
  size_t span;
};

// Create unordered map of tensor lifetimes for quick lookup during Viterbi cost
// calculation
using TensorLifetimeMap = llvm::DenseMap<mlir::Value, TensorLifetime>;
using TensorLifetimePair = std::pair<mlir::Value, TensorLifetime>;
using SortedLifetimesMap = llvm::SmallVector<TensorLifetimePair>;

using ActivationLifetime = std::pair<mlir::Value, TensorLifetime>;
using ActivationLifetimeList = llvm::SmallVector<ActivationLifetime>;

using OperationSchedule =
    llvm::DenseMap<mlir::func::FuncOp, llvm::SmallVector<mlir::Operation *>>;

// Returned by ViterbiPolicy after execution
struct ViterbiResult {
  llvm::DenseMap<mlir::Operation *, OpConfig> optimalConfigurations;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<TTNNLayoutAttr>>
      inputLayouts;
  double totalCost;
  size_t numL1Configs = 0;   // How many ops got L1 placement
  size_t numDRAMConfigs = 0; // How many ops DRAM spilled
};

class ViterbiPolicy {
public:
  // Constructors
  ViterbiPolicy() = default;

  ViterbiPolicy(const OpCandidateBuilderResult &candidateResult,
                const OperationSchedule &schedule);

  // Main baseline function
  ViterbiResult run();   // Main workflow
  ViterbiResult solve(); // Core Viterbi algorithm

  // Configuratioln and status
  void reset();

private:
  // Main algorithm phases
  void buildViterbiTable();
  void performCostCalculation();
  void performBacktracking();

  // Helpers
  void calculateTensorLifetimes(const OperationSchedule &schedule);
  const TensorLifetimeMap &getTensorLifetimes() const {
    return tensorLifetimes;
  }

  // Checker
  bool isJoinOperation(mlir::Operation *op) const;

  // Debug and logging helpers
  void printOptimalPath() const;
  void printDetailedInputs() const;
  void printScheduleOrder(const OperationSchedule &schedule);
  void printCandidateInfo(const OperationSchedule &schedule);
  void printLivenessInfo(const mlir::LivenessBlockInfo *livenessInfo);
  void printTensorLifetimes(const llvm::SmallVector<mlir::Operation *> &ops,
                            const TensorLifetimeMap &lifetimes) const;

private:
  // Provided data for baseline
  OpCandidateBuilderResult candidateResult;
  OperationSchedule schedule;
  std::shared_ptr<CostModel> costModel;

  // Algorithm state
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<double>>
      viterbiTable; // Store cost for each candidate of each operation
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int>>
      backtrackTable; // Store index of best candidates for backtracking
  llvm::DenseMap<mlir::Operation *, size_t> optimalCandidateIndex;

  // Tensor lifetime map
  TensorLifetimeMap tensorLifetimes;
  SortedLifetimesMap sortLifetimes(const TensorLifetimeMap &lifetimeMap) const;
  ActivationLifetimeList
  getActivationLifetime(mlir::Operation *currentOp) const;
  // Results and status
};

} // namespace mlir::tt::ttnn