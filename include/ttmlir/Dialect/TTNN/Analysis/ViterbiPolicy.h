// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttmlir/Dialect/TTNN/Analysis/OpCandidatesBuilder.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  MissingCandidateState,
  NoValidGlobalPath,
  NoValidSeed,
  InvalidBacktrackState,
  InvalidCandidateSelection,
  InvalidTransitionState,
  ConflictUnresolved,
  PartialAssignment,
};

const char *getSolverStatusString(SolverStatus status);
bool isSolverOk(SolverStatus status);

struct SpillRequest {
  mlir::Value value;
  mlir::Operation *triggerOp = nullptr;
};

// Contains the selected configuration, input layouts, and memory-management
// requests produced by ViterbiPolicy.
struct ViterbiResult {
  llvm::DenseMap<mlir::Operation *, OpConfig> optimalConfigurations;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<TTNNLayoutAttr>>
      inputLayouts;
  llvm::SmallVector<SpillRequest> spillRequests;
  SolverStatus status = SolverStatus::Success;
  double totalCost = 0.0;
  std::size_t numL1Configs = 0;
  std::size_t numDRAMConfigs = 0;
};

struct TensorLifetime {
  std::size_t startOpIdx = 0;
  std::size_t endOpIdx = 0;
  std::size_t span = 0;
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
  ViterbiPolicy() = default;

  ViterbiPolicy(const OpCandidateBuilderResult &candidateResult,
                const OperationSchedule &schedule,
                PrunedGraphInfo prunedGraphInfo = PrunedGraphInfo());

  ViterbiResult run();
  ViterbiResult solve();
  void reset();

private:
  struct ParentCandRequest {
    mlir::Operation *currentOp = nullptr;
    std::size_t selectedCurrentCandIdx = 0;
    std::size_t expectedParentCandIdx = 0;
  };

  SolverStatus buildViterbiTable();
  SolverStatus performCostCalculation();
  SolverStatus performBacktracking();
  SolverStatus validateFinalAssignment() const;

  ViterbiResult
  constructResult(SolverStatus status = SolverStatus::Success) const;

  llvm::SmallVector<std::pair<mlir::Operation *, std::size_t>>
  getBacktrackingSeeds() const;

  double getBestTransitionCost(mlir::Operation *parentOp,
                               const OpConfigCandidate &parentCandidate,
                               mlir::Operation *currentOp,
                               const OpConfigCandidate &currentCandidate,
                               uint64_t additionalL1Usage) const;

  std::optional<std::size_t>
  resolveParentCandConflict(mlir::Operation *parentOp,
                            llvm::ArrayRef<ParentCandRequest> requests) const;

  void calculateTensorLifetimes(const OperationSchedule &schedule);

  const TensorLifetimeMap &getTensorLifetimes() const {
    return tensorLifetimes;
  }

  SortedLifetimesMap sortLifetimes(const TensorLifetimeMap &lifetimeMap) const;

  LiveTensorList getAllLiveActivations(mlir::Operation *currentOp) const;
  LiveTensorList getParentActivations(mlir::Operation *currentOp) const;

  llvm::SmallVector<mlir::Operation *>
  getOpParents(mlir::Operation *currentOp) const;

  void storeCandidateOutputSize(mlir::Operation *op, std::size_t candidateIdx,
                                std::optional<uint64_t> outputSizeBytes);

  LiveTensorList getPassiveActivations(mlir::Operation *currentOp) const;

  void storeCandidateAdditionalL1Usage(mlir::Operation *op,
                                       std::size_t candidateIdx,
                                       uint64_t additionalL1Usage);

  void storeSelectedSpillCount(mlir::Operation *op, std::size_t candidateIdx,
                               std::size_t selectedSpillCount);

  std::optional<uint64_t>
  getCandidateAdditionalL1Usage(mlir::Operation *op,
                                std::size_t candidateIdx) const;

  llvm::SmallVector<mlir::Operation *>
  getPassiveTensorProducers(const LiveTensorList &passiveActivations) const;

  llvm::SmallVector<mlir::Value>
  getParentOperandsForTransition(mlir::Operation *consumerOp,
                                 mlir::Operation *parentOp) const;

  bool isActivationLiveAtOp(mlir::Value value, std::size_t currentOpIdx,
                            mlir::Operation *currentOp) const;

  std::optional<LiveTensorInfo> getActivationInfo(mlir::Operation *currentOp,
                                                  mlir::Value value) const;

  bool isJoinOperation(mlir::Operation *op) const;

  void printOptimalPath() const;
  void printActivation(mlir::Operation *currentOp, mlir::Value value,
                       LiveTensorList &activations,
                       bool requireRepresentativeMatch = false) const;

  OpCandidateBuilderResult candidateResult;
  OperationSchedule schedule;
  std::shared_ptr<CostModel> costModel;

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<double>> viterbiTable;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<llvm::SmallVector<int>>>
      backtrackTable;
  llvm::DenseMap<mlir::Operation *, std::size_t> optimalCandidateIndex;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::optional<uint64_t>>>
      candidateOutputSizes;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::optional<uint64_t>>>
      candidateAdditionalL1Usages;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::size_t>>
      selectedSpillCounts;

  TensorLifetimeMap tensorLifetimes;
  PrunedGraphInfo prunedGraphInfo;
};

} // namespace mlir::tt::ttnn
