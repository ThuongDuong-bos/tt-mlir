// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpCandidatesBuilder.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::tt::ttnn {

class CostModel;

using OperationSchedule =
    llvm::DenseMap<mlir::func::FuncOp, llvm::SmallVector<mlir::Operation *>>;

enum class SolverStatus {
  Success,

  // Input / DP construction state
  MissingCandidateState,
  NoValidGlobalPath,

  // Backtracking state
  NoValidSeed,
  InvalidBacktrackState,
  InvalidCandidateSelection,
  InvalidTransitionState,
  ConflictUnresolved,

  // Final result validation
  PartialAssignment,
};

const char *getSolverStatusString(SolverStatus status);
bool isSolverOk(SolverStatus status);

// Returned by ViterbiPolicy after execution
struct ViterbiResult {
  llvm::DenseMap<mlir::Operation *, OpConfig> optimalConfigurations;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<TTNNLayoutAttr>>
      inputLayouts;
  SolverStatus status = SolverStatus::Success;
  double totalCost = 0.0;
  size_t numL1Configs = 0;   // How many ops got L1 placement
  size_t numDRAMConfigs = 0; // How many ops DRAM spilled
};

struct TensorLifetime {
  size_t startOpIdx = 0;
  size_t endOpIdx = 0;
  size_t span = 0;
};

using TensorLifetimeMap = llvm::DenseMap<mlir::Value, TensorLifetime>;
using TensorLifetimePair = std::pair<mlir::Value, TensorLifetime>;
using SortedLifetimesMap = llvm::SmallVector<TensorLifetimePair>;

struct LiveTensorInfo {
  mlir::Value value;
  TensorLifetime lifetime;
  std::optional<TTNNLayoutAttr> layout;
};

using LiveTensorList = llvm::SmallVector<LiveTensorInfo>;

class ViterbiPolicy {
public:
  // Constructors
  ViterbiPolicy() = default;

  ViterbiPolicy(const OpCandidateBuilderResult &candidateResult,
                const OperationSchedule &schedule,
                PrunedGraphInfo prunedGraphInfo = PrunedGraphInfo());

  // Main baseline function
  ViterbiResult run();   // Main workflow
  ViterbiResult solve(); // Core Viterbi algorithm

  // Configuratioln and status
  void reset();

private:
  struct ParentCandRequest {
    mlir::Operation *currentOp = nullptr;
    size_t selectedCurrentCandIdx = 0;
    size_t expectedParentCandIdx = 0;
  };

  // Main algorithm phases
  SolverStatus buildViterbiTable();
  SolverStatus performCostCalculation();
  SolverStatus performBacktracking();
  
  llvm::SmallVector<std::pair<mlir::Operation *, size_t>>
  getBacktrackingSeeds() const;
  double getBestTransitionCost(
      mlir::Operation *parentOp, const OpConfigCandidate &parentCandidate,
      mlir::Operation *currentOp, const OpConfigCandidate &currentCandidate,
      uint64_t additionalL1Usage) const;
  std::optional<size_t> resolveParentCandConflict(
      mlir::Operation *parentOp,
      llvm::ArrayRef<ParentCandRequest> requests) const;
  SolverStatus validateFinalAssignment() const;
  ViterbiResult
  constructResult(SolverStatus status = SolverStatus::Success) const;

  // Tensor lifetime analysis

  void calculateTensorLifetimes(const OperationSchedule &schedule);
  const TensorLifetimeMap &getTensorLifetimes() const {
    return tensorLifetimes;
  }
  LiveTensorList getAllLiveActivations(mlir::Operation *currentOp) const;
  LiveTensorList getParentActivations(mlir::Operation *currentOp) const;
  llvm::SmallVector<mlir::Operation *>
  getOpParents(mlir::Operation *currentOp) const;
  void storeCandidateOutputSize(mlir::Operation *op, size_t candidateIdx,
                                std::optional<uint64_t> outputSizeBytes);
  LiveTensorList getPassiveActivations(mlir::Operation *currentOp) const;
  llvm::SmallVector<mlir::Operation *>
  getPassiveTensorProducers(const LiveTensorList &passiveActivations) const;
  llvm::SmallVector<mlir::Value>
  getParentOperandsForTransition(mlir::Operation *consumerOp,
                                 mlir::Operation *parentOp) const;
  std::optional<LiveTensorInfo> getActivationInfo(mlir::Operation *currentOp,
                                                  mlir::Value value) const;
  bool isActivationLiveAtOp(mlir::Value value, size_t currentOpIdx,
                            mlir::Operation *currentOp) const;

  // Checker
  bool isJoinOperation(mlir::Operation *op) const;

  // Debug and logging helpers
  void printOptimalPath() const;
  void printDetailedInputs() const;
  void printActivation(mlir::Operation *currentOp, mlir::Value value,
                       LiveTensorList &activations,
                       bool requireRepresentativeMatch = false) const;

private:
  // Provided data for baseline
  OpCandidateBuilderResult candidateResult;
  OperationSchedule schedule;
  std::shared_ptr<CostModel> costModel;

  // Algorithm state
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<double>>
      viterbiTable; // Store cost for each candidate of each operation
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<llvm::SmallVector<int>>>
      backtrackTable; // Store best parent candidate index per
                      // op-candidate-parent
  llvm::DenseMap<mlir::Operation *, size_t> optimalCandidateIndex;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::optional<uint64_t>>>
      candidateOutputSizes; // Store output bytes for each op candidate

  // Tensor lifetime map
  TensorLifetimeMap tensorLifetimes;
  SortedLifetimesMap sortLifetimes(const TensorLifetimeMap &lifetimeMap) const;
  PrunedGraphInfo prunedGraphInfo;
  // Results and status
};

} // namespace mlir::tt::ttnn