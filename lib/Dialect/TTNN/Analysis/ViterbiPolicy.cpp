// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ViterbiPolicy.h"
#include "mlir/IR/Attributes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Analysis/CostModel.h"
#include "ttmlir/Support/Logger.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace {

mlir::Value mapValueFromPrunedGraph(
    mlir::Value value,
    const llvm::DenseMap<mlir::Operation *, size_t> &prunedOpIndex) {
  if (prunedOpIndex.empty()) {
    return value;
  }

  mlir::Value current = value;

  while (mlir::Operation *defOp = current.getDefiningOp()) {
    if (prunedOpIndex.contains(defOp)) {
      break;
    }

    mlir::Value next;
    for (mlir::Value operand : defOp->getOperands()) {
      if (mlir::isa<mlir::TensorType>(operand.getType())) {
        next = operand;
        break;
      }
    }

    if (!next || next == current) {
      break;
    }

    current = next;
  }

  return current;
}

} // namespace

namespace mlir::tt::ttnn {

const double inf = std::numeric_limits<double>::infinity();

const char *getSolverStatusString(SolverStatus status) {
  const char *statusString = "Unknown";

  switch (status) {
  case SolverStatus::Success:
    statusString = "Success";
    break;
  case SolverStatus::MissingCandidateState:
    statusString = "MissingCandidateState";
    break;
  case SolverStatus::NoValidGlobalPath:
    statusString = "NoValidGlobalPath";
    break;
  case SolverStatus::NoValidSeed:
    statusString = "NoValidSeed";
    break;
  case SolverStatus::InvalidBacktrackState:
    statusString = "InvalidBacktrackState";
    break;
  case SolverStatus::InvalidCandidateSelection:
    statusString = "InvalidCandidateSelection";
    break;
  case SolverStatus::InvalidTransitionState:
    statusString = "InvalidTransitionState";
    break;
  case SolverStatus::ConflictUnresolved:
    statusString = "ConflictUnresolved";
    break;
  case SolverStatus::PartialAssignment:
    statusString = "PartialAssignment";
    break;
  }

  return statusString;
}

bool isSolverOk(SolverStatus status) { return status == SolverStatus::Success; }

// Constructor for BOS Optimizer Pass.
// Inputs:
//  - candidateResult: Precomputed op candidates from OpCandidatesBuilder
//  - schedule: Operation schedule (full or pruned)
//  - prunedGraphInfo: Optional pruned graph metadata used for representative
//    value mapping

ViterbiPolicy::ViterbiPolicy(const OpCandidateBuilderResult &candidateResult,
                             const OperationSchedule &schedule,
                             PrunedGraphInfo prunedGraphInfo)
    : candidateResult(candidateResult), schedule(schedule),
      costModel(std::make_shared<CostModel>()),
      prunedGraphInfo(std::move(prunedGraphInfo)) {}

// =================================================================
// MAIN BASELINE FOR VITERBI POLICY
// =================================================================

ViterbiResult ViterbiPolicy::run() {
  // The schedulerResult just have 1 func
  [[maybe_unused]] size_t totalScheduledOps = 0;
  if (!schedule.empty()) {
    totalScheduledOps = schedule.begin()->second.size();
  }
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "Running Viterbi Policy with {} operations", totalScheduledOps);
  calculateTensorLifetimes(schedule);
  [[maybe_unused]] SortedLifetimesMap sorted = sortLifetimes(tensorLifetimes);
  return solve();
}

ViterbiResult ViterbiPolicy::solve() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "Executing Viterbi algorithm\n");

  ViterbiResult result;
  const char *failedStage = nullptr;
  SolverStatus status = buildViterbiTable();

  if (!isSolverOk(status)) {
    failedStage = "buildViterbiTable";
  }

  if (isSolverOk(status)) {
    status = performCostCalculation();
    if (!isSolverOk(status)) {
      failedStage = "performCostCalculation";
    }
  }

  if (isSolverOk(status)) {
    status = performBacktracking();
    if (!isSolverOk(status)) {
      failedStage = "performBacktracking";
    }
  }

  if (isSolverOk(status)) {
    status = validateFinalAssignment();
    if (!isSolverOk(status)) {
      failedStage = "validateFinalAssignment";
    }
  }

  // If any stage of the solver failed, we log the failure and emit a warning on
  // one of the operations
  if (!isSolverOk(status)) {
    for (const auto &[_, operations] : schedule) {
      if (!operations.empty()) {
        operations.front()->emitWarning()
            << "Viterbi: solver stopped after " << failedStage
            << " with status " << getSolverStatusString(status);
        break;
      }
    }

    result.status = status;
  } else {
    // If the solver is successful, we can construct the result with the optimal
    // path and its cost
    result = constructResult(status);

    printOptimalPath();

    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                 "Viterbi solve completed. Status: {0}, Total cost: {1}, L1: "
                 "{2}, DRAM: {3}\n",
                 getSolverStatusString(result.status), result.totalCost,
                 result.numL1Configs, result.numDRAMConfigs);
  }

  return result;
}

SolverStatus ViterbiPolicy::validateFinalAssignment() const {
  SolverStatus status = SolverStatus::Success;
  for (const auto &[_, operations] : schedule) {
    for (size_t opIdx = 0; opIdx < operations.size(); ++opIdx) {
      mlir::Operation *op = operations[opIdx];
      auto candidateIt = candidateResult.candidateMap.find(op);
      if (candidateIt == candidateResult.candidateMap.end() ||
          candidateIt->second.empty()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} has missing candidate "
                     "state",
                     op->getName().getStringRef());
        status = SolverStatus::MissingCandidateState;
        goto returnStatus;
      }

      auto selectedIt = optimalCandidateIndex.find(op);
      if (selectedIt == optimalCandidateIndex.end()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} has no selected "
                     "candidate",
                     op->getName().getStringRef());
        status = SolverStatus::PartialAssignment;
        goto returnStatus;
      }

      const size_t selectedCandidateIdx = selectedIt->second;
      if (selectedCandidateIdx >= candidateIt->second.size()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} selected candidate {} "
                     "is Out-Of-Bounds (candidate count {})",
                     op->getName().getStringRef(), selectedCandidateIdx,
                     candidateIt->second.size());
        status = SolverStatus::InvalidCandidateSelection;
        goto returnStatus;
      }

      auto costIt = viterbiTable.find(op);
      if (costIt == viterbiTable.end() ||
          selectedCandidateIdx >= costIt->second.size()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} selected candidate {} "
                     "has no DP cost entry",
                     op->getName().getStringRef(), selectedCandidateIdx);
        status = SolverStatus::InvalidCandidateSelection;
        goto returnStatus;
      }

      if (!std::isfinite(costIt->second[selectedCandidateIdx])) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} selected candidate {} "
                     "has non-finite DP cost {}",
                     op->getName().getStringRef(), selectedCandidateIdx,
                     costIt->second[selectedCandidateIdx]);
        status = SolverStatus::NoValidGlobalPath;
        goto returnStatus;
      }

      auto backtrackIt = backtrackTable.find(op);
      if (backtrackIt == backtrackTable.end() ||
          selectedCandidateIdx >= backtrackIt->second.size()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} selected candidate {} "
                     "has invalid backtrack state",
                     op->getName().getStringRef(), selectedCandidateIdx);
        status = SolverStatus::InvalidBacktrackState;
        goto returnStatus;
      }

      const llvm::SmallVector<mlir::Operation *> parents = getOpParents(op);
      const llvm::SmallVector<int> &bestPrevIdxByParent =
          backtrackIt->second[selectedCandidateIdx];
      if (bestPrevIdxByParent.size() < parents.size()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} selected candidate {} "
                     "has {} backtrack parents, expected {}",
                     op->getName().getStringRef(), selectedCandidateIdx,
                     bestPrevIdxByParent.size(), parents.size());
        status = SolverStatus::InvalidBacktrackState;
        goto returnStatus;
      }

      for (size_t parentIdx = 0; parentIdx < parents.size(); ++parentIdx) {
        const int expectedParentCandIdx = bestPrevIdxByParent[parentIdx];
        if (expectedParentCandIdx < 0) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "ValidateFinalAssignment: op {} candidate {} "
                       "parentIdx {} has invalid parent candidate {}",
                       op->getName().getStringRef(), selectedCandidateIdx,
                       parentIdx, expectedParentCandIdx);
          status = SolverStatus::InvalidBacktrackState;
          goto returnStatus;
        }

        mlir::Operation *parentOp = parents[parentIdx];
        auto parentSelectedIt = optimalCandidateIndex.find(parentOp);
        if (parentSelectedIt == optimalCandidateIndex.end()) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "ValidateFinalAssignment: parent op {} requested by "
                       "op {} candidate {} is unassigned",
                       parentOp->getName().getStringRef(),
                       op->getName().getStringRef(), selectedCandidateIdx);
          status = SolverStatus::PartialAssignment;
          goto returnStatus;
        }

        if (parentSelectedIt->second !=
            static_cast<size_t>(expectedParentCandIdx)) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "ValidateFinalAssignment: parent op {} selected "
                       "candidate {} does not match op {} candidate {} "
                       "expected parent candidate {}",
                       parentOp->getName().getStringRef(),
                       parentSelectedIt->second, op->getName().getStringRef(),
                       selectedCandidateIdx, expectedParentCandIdx);
          status = SolverStatus::InvalidTransitionState;
          goto returnStatus;
        }
      }
    }
  }

returnStatus:
  return status;
}

ViterbiResult ViterbiPolicy::constructResult(SolverStatus status) const {
  ViterbiResult result;
  result.status = status;
  result.totalCost = 0.0;
  result.numL1Configs = 0;
  result.numDRAMConfigs = 0;

  for (const auto &[op, bestIdx] : optimalCandidateIndex) {
    auto candidateIt = candidateResult.candidateMap.find(op);
    bool hasCandidate = candidateIt != candidateResult.candidateMap.end();
    if (hasCandidate) {
      hasCandidate = bestIdx < candidateIt->second.size();
    }
    if (!hasCandidate) {
      continue;
    }

    const auto &bestCandidate = candidateIt->second[bestIdx];
    result.optimalConfigurations[op] = bestCandidate.opConfig;
    result.inputLayouts[op] = bestCandidate.inputLayouts;

    const auto bufferType = bestCandidate.opConfig.outputLayout.getBufferType();
    if (bufferType == BufferType::L1) {
      ++result.numL1Configs;
    } else if (bufferType == BufferType::DRAM) {
      ++result.numDRAMConfigs;
    }
  }

  // Use the backtracking seeds to calculate total cost.
  // TODO: Handle multi-output case for sink ops.
  // Only considered sink ops
  for (const auto &[sinkOp, seedIdx] : getBacktrackingSeeds()) {
    auto selectedIt = optimalCandidateIndex.find(sinkOp);
    const size_t selectedIdx = selectedIt != optimalCandidateIndex.end()
                                   ? selectedIt->second
                                   : seedIdx;
    const auto &sinkCosts = viterbiTable.lookup(sinkOp);
    if (selectedIdx >= sinkCosts.size()) {
      continue;
    }

    const double sinkCost = sinkCosts[selectedIdx];
    if (std::isfinite(sinkCost)) {
      result.totalCost += sinkCost;
    }
  }

  return result;
}

// =================================================================
// CORE ALGORITHM FUNCTIONS
// =================================================================

// Create Viterbi DP table (N x T) based on candidate configurations and costs
SolverStatus ViterbiPolicy::buildViterbiTable() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer, "Building Viterbi tables");
  SolverStatus status = SolverStatus::Success;

  for (const auto &[func, operations] : schedule) {
    for (auto *op : operations) {
      // Find candidates for this operation
      auto it = candidateResult.candidateMap.find(op);
      if (it == candidateResult.candidateMap.end() || it->second.empty()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "BuildViterbiTable: op {} has missing candidate state",
                     op->getName().getStringRef());
        status = SolverStatus::MissingCandidateState;
        continue;
      }

      const auto &candidates = it->second;

      size_t numCandidates = candidates.size();
      viterbiTable[op].resize(numCandidates);
      backtrackTable[op].resize(numCandidates);

      for (size_t i = 0; i < numCandidates; ++i) {
        viterbiTable[op][i] = inf;
        backtrackTable[op][i].clear();
      }
    }
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "Finished building Viterbi tables");
  return status;
}

SolverStatus ViterbiPolicy::performCostCalculation() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "Performing cost calculation for Viterbi DP table");
  SolverStatus status = SolverStatus::Success;
  bool stopCostCalculation = false;

  // Loop through each operation in scheduled topological order
  for (const auto &entry : schedule) {
    if (stopCostCalculation) {
      break;
    }

    const auto &ops = entry.second;
    // Loop through scheduled operations until we encounter a failure that
    // requires us to stop the cost calculation
    for (size_t opIdx = 0; opIdx < ops.size() && !stopCostCalculation;
         ++opIdx) {
      mlir::Operation *op = ops[opIdx];

      auto it = candidateResult.candidateMap.find(op);
      if (it == candidateResult.candidateMap.end() || it->second.empty()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "CostCalculation: op {} has missing candidate state",
                     op->getName().getStringRef());
        status = SolverStatus::MissingCandidateState;
        continue;
      }

      auto &candidates = it->second;

      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                   "Calculating costs for op {} with {} candidates",
                   op->getName().getStringRef(), candidates.size());

      // Prepare parent/passive producer context for cost evaluation.
      [[maybe_unused]] bool isJoin = isJoinOperation(op);
      llvm::SmallVector<mlir::Operation *> opParents;
      llvm::SmallVector<mlir::Operation *> passiveTensorProducers;

      // Get current op parents (used for transition cost).
      opParents = getOpParents(op);

      // Pre-retrieve passive tensor producers for current op to avoid redundant
      // retrieval during cost model query.
      LiveTensorList passiveActivations = getPassiveActivations(op);
      passiveTensorProducers = getPassiveTensorProducers(passiveActivations);

      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                   "Op {} passive producers=[{}]", op->getName().getStringRef(),
                   optimizer_utils::parentOpsToString(passiveTensorProducers));
      // Debug join condition
      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer, "Op {} isJoin={}",
                   op->getName().getStringRef(), isJoin);

      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                   "Calculating {} costs for op {} with {} candidates",
                   isJoin ? "join" : "linear", op->getName().getStringRef(),
                   candidates.size());

      // Loop through each candidate for current op
      // dp[u][i] = local[u][i] + sum_p min_j(dp[p][j] + t(p_j -> u_i)).
      for (size_t i = 0; i < candidates.size();) {
        const OpConfigCandidate &candidate = candidates[i];

        // Retrieve local cost and output size for current candidate.
        const CostModel::LocalCostResult localResult = costModel->getLocalCost(
            op, candidate, passiveTensorProducers,
            /*shouldComputeOutputSize=*/true, candidateOutputSizes);

        // Store candidate output size for later use
        storeCandidateOutputSize(op, i, localResult.outputSizeBytes);

        // Initialize total cost infinity
        // The total cost will be updated only when we have valid local cost and
        // valid paths from parents
        double totalCost = inf;
        llvm::SmallVector<int> bestPrevIdxByParent(opParents.size(), -1);

        if (std::isfinite(localResult.cost)) {
          double parentCostSum = 0.0;
          bool parentPathValid = true;

          // Loop through each parent of current op
          // For linear op, only 1 parent. For join op, multiple parents.
          for (size_t parentIdx = 0; parentIdx < opParents.size();
               ++parentIdx) {
            // Pick equivalent parent op in the pruned graph based on consumer
            // operand's index
            mlir::Operation *const parentOp = opParents[parentIdx];
            // Find the corresponding parent candidate costs from DP table
            auto parentCandidatesIt =
                candidateResult.candidateMap.find(parentOp);
            auto parentDpIt = viterbiTable.find(parentOp);
            // Guard check against missing parent candidate
            if (parentCandidatesIt == candidateResult.candidateMap.end() ||
                parentDpIt == viterbiTable.end()) {
              TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                           "DP op={} candidate={} -> inf: parent {} missing "
                           "{}{}",
                           op->getName().getStringRef(), i,
                           parentOp->getName().getStringRef(),
                           parentCandidatesIt ==
                                   candidateResult.candidateMap.end()
                               ? "candidates"
                               : "",
                           parentDpIt == viterbiTable.end() ? " dpTable" : "");
              parentPathValid = false;
              break;
            }

            const auto &parentCandidates = parentCandidatesIt->second;
            const auto &parentDp = parentDpIt->second;

            double bestParentCost = inf;
            int bestParentIdx = -1;

            const size_t parentCandidateCount =
                std::min(parentCandidates.size(), parentDp.size());

            // Loop through each parent's candidates to find the best transition
            // cost to current candidate
            for (size_t j = 0; j < parentCandidateCount; ++j) {
              // Skip parent's invalid candidates
              if (!std::isfinite(parentDp[j])) {
                continue;
              }

              const double bestEdgeCost = getBestTransitionCost(
                  parentOp, parentCandidates[j], op, candidate,
                  localResult.additionalL1Usage);

              // Only assign best parent cost
              const double parentCandidateCost = parentDp[j] + bestEdgeCost;
              if (parentCandidateCost < bestParentCost) {
                bestParentCost = parentCandidateCost;
                bestParentIdx = static_cast<int>(j);
                TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                             "DP best edge op={} c={} parent={} pc={} "
                             "parentDp={} edge={} combined={}",
                             op->getName().getStringRef(), i,
                             parentOp->getName().getStringRef(), j, parentDp[j],
                             bestEdgeCost, parentCandidateCost);
              }
            }

            // Guard against no valid path from parent candidate to current
            // candidate
            if (!std::isfinite(bestParentCost)) {
              TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                           "DP op={} candidate={} -> inf: no finite path from "
                           "parent {}",
                           op->getName().getStringRef(), i,
                           parentOp->getName().getStringRef());
              parentPathValid = false;
              break;
            }

            // Accumulate parent cost sum and record best parent candidate index
            // for backtracking
            parentCostSum += bestParentCost;
            bestPrevIdxByParent[parentIdx] = bestParentIdx;
          }

          // If we have valid paths from all parents, then we can update the
          // total cost for current candidate
          if (parentPathValid) {
            totalCost = localResult.cost + parentCostSum;
          } else {
            std::fill(bestPrevIdxByParent.begin(), bestPrevIdxByParent.end(),
                      -1);
          }
        } else {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "DP op={} candidate={} -> inf: localCost=inf",
                       op->getName().getStringRef(), i);
        }

        // dp[u][i] = local[u][i] + sum_p min_j(dp[p][j] + t(p_j -> u_i))
        // backtrackTable[u][i][p] = argmin_j(dp[p][j] + t(p_j -> u_i))
        viterbiTable[op][i] = totalCost;
        backtrackTable[op][i] = bestPrevIdxByParent;
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "DP op={} c={} local={} parentSum={} total={} btSize={}",
                     op->getName().getStringRef(), i, localResult.cost,
                     std::isfinite(totalCost) ? totalCost - localResult.cost
                                              : inf,
                     totalCost, bestPrevIdxByParent.size());
        for (size_t parentIdx = 0; parentIdx < bestPrevIdxByParent.size();
             ++parentIdx) {
          TTMLIR_DEBUG(
              ttmlir::LogComponent::ViterbiOptimizer,
              "DP backtrack op={} c={} parentIdx={} parentCandidate={}",
              op->getName().getStringRef(), i, parentIdx,
              bestPrevIdxByParent[parentIdx]);
        }

        // Keep layout-group skip behavior for grouped candidates.
        // TODO: check the outer prunning case, especially the grid mismatch
        // case Ex: Maybe height shard 20x1 invalid (non-OOM case) but height
        // shard 14x1 valid.
        // If current candidate belong to default group, we don't skip any
        // candidate in here
        const bool isDefaultGroup =
            optimizer_utils::getLayout(candidate.opConfig.outputLayout) ==
            TensorMemoryLayout::Interleaved;
        const bool shouldSkipGroup = localResult.skipGroup &&
                                     candidate.groupIndex.has_value() &&
                                     !isDefaultGroup;
        if (!shouldSkipGroup) {
          ++i;
          continue;
        }

        const size_t groupIndex = *candidate.groupIndex;
        ++i;
        while (i < candidates.size() && candidates[i].groupIndex &&
               *candidates[i].groupIndex == groupIndex) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "DP op={} candidate={} -> inf: skipped by invalid "
                       "layout group {}",
                       op->getName().getStringRef(), i, groupIndex);
          viterbiTable[op][i] = std::numeric_limits<double>::infinity();
          backtrackTable[op][i].assign(opParents.size(), -1);
          ++i;
        }
      }

      // Only used for debugging purpose
      // The sort below only for logging, not affecting/overlapping to the main
      // flow
      auto storeIt = candidateOutputSizes.find(op);
      if (storeIt == candidateOutputSizes.end()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "computeOutputSize store check: op={} no stored entry",
                     op->getName().getStringRef());
      } else {
        // Define a struct to hold candidate index and its corresponding
        // output size for sorting
        const auto &storedSizes = storeIt->second;
        struct RankedCacheEntry {
          size_t candidateIdx;
          uint64_t bytes;
        };
        llvm::SmallVector<RankedCacheEntry> rankedEntries;
        rankedEntries.reserve(storedSizes.size());
        for (size_t storeIdx = 0; storeIdx < storedSizes.size(); ++storeIdx) {
          // Only consider candidates that have stored output size (valid one)
          if (!storedSizes[storeIdx]) {
            continue;
          }
          rankedEntries.push_back({storeIdx, *storedSizes[storeIdx]});
        }

        // Larger output size is put first
        // Insertion sort since the candidate count is usually small
        for (size_t rankedIdx = 1; rankedIdx < rankedEntries.size();
             ++rankedIdx) {
          RankedCacheEntry currentEntry = rankedEntries[rankedIdx];
          size_t insertIdx = rankedIdx;
          while (insertIdx > 0) {
            const RankedCacheEntry &previousEntry =
                rankedEntries[insertIdx - 1];
            const bool shouldMovePrevious =
                previousEntry.bytes < currentEntry.bytes;
            if (!shouldMovePrevious) {
              break;
            }
            rankedEntries[insertIdx] = previousEntry;
            --insertIdx;
          }
          rankedEntries[insertIdx] = currentEntry;
        }

        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "computeOutputSize store check: op={} storedCount={}",
                     op->getName().getStringRef(), rankedEntries.size());
        for ([[maybe_unused]] const RankedCacheEntry &entry : rankedEntries) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer, "  c[{}] = {}",
                       entry.candidateIdx, entry.bytes);
        }
      }

      // After cost calculation, check if we have at least one finite candidate
      // for this op for early stopping
      bool hasFiniteCandidate = false;
      for (double cost : viterbiTable[op]) {
        if (std::isfinite(cost)) {
          hasFiniteCandidate = true;
          break;
        }
      }
      if (!hasFiniteCandidate) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Viterbi: op has no finite candidate after cost "
                     "calculation");
        status = SolverStatus::NoValidGlobalPath;
        stopCostCalculation = true;
      } else {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Finished cost calculation for op {} \n",
                     op->getName().getStringRef());
      }
    }
  }

  if (status != SolverStatus::NoValidGlobalPath) {
    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                 "Finished cost calculation");
  }
  return status;
}

// Backtracking to find optimal layout for each operation
SolverStatus ViterbiPolicy::performBacktracking() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "Performing backtracking to find optimal layout configurations");

  optimalCandidateIndex.clear();
  SolverStatus status = SolverStatus::Success;

  // currentOp = op being traversed right now
  // parentOp = ops that feed their output tensor to currentOp in forward
  // direction

  llvm::SmallVector<mlir::Operation *> worklist;
  llvm::DenseMap<mlir::Operation *, bool> traversedOps;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<ParentCandRequest>>
      expectedCandsByParent;

  // Remove requests issued by currentOp so that outdated parent candidate
  // requests are not kept when currentOp is traversed again with a new selected
  // candidate.
  auto removeRequestFromCurrentOp = [&](mlir::Operation *currentOp) {
    llvm::SmallVector<mlir::Operation *> emptyParentOps;
    // Remove requests issued by currentOp from expectedCandsByParent and erase
    // parents with no remaining requests
    for (auto &entry : expectedCandsByParent) {
      llvm::SmallVector<ParentCandRequest> &requests = entry.second;
      for (auto requestIt = requests.begin(); requestIt != requests.end();) {
        if (requestIt->currentOp == currentOp) {
          requestIt = requests.erase(requestIt);
        } else {
          ++requestIt;
        }
      }

      // If after removal, there is no more request for this parent op, we can
      // remove this parent op from expectedCandsByParent
      if (requests.empty()) {
        emptyParentOps.push_back(entry.first);
      }
    }

    // Remove parent ops with no more requests from expectedCandsByParent
    for (mlir::Operation *emptyParentOp : emptyParentOps) {
      expectedCandsByParent.erase(emptyParentOp);
    }
  };

  // Record those information for later traversing and resolving conflict:
  // parentOp, currentOp, candidate idx requested by currentOp and expected
  // parent candidate idx by currentOp
  auto recordExpectedParentCand = [&](mlir::Operation *parentOp,
                                      const ParentCandRequest &request) {
    expectedCandsByParent[parentOp].push_back(request);
  };

  // Initialize worklist with sink ops and their best candidate index as the
  // backtracking seed
  llvm::SmallVector<std::pair<mlir::Operation *, size_t>> seeds =
      getBacktrackingSeeds();
  if (seeds.empty()) {
    bool hasScheduledCandidateOp = false;
    for (const auto &[_, operations] : schedule) {
      for (mlir::Operation *scheduledOp : operations) {
        auto candidateIt = candidateResult.candidateMap.find(scheduledOp);
        if (candidateIt != candidateResult.candidateMap.end() &&
            !candidateIt->second.empty()) {
          hasScheduledCandidateOp = true;
          break;
        }
      }
      if (hasScheduledCandidateOp) {
        break;
      }
    }

    if (hasScheduledCandidateOp) {
      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                   "Backtracking: no valid finite sink seed; stopping");
      status = SolverStatus::NoValidSeed;
    }
  }

  for (const auto &[sinkOp, bestIdx] : seeds) {
    optimalCandidateIndex[sinkOp] = bestIdx;
    worklist.push_back(sinkOp);
  }

  while (isSolverOk(status)) {
    // Phase 1: Traverse selected current ops and collect expected parent
    // candidate requests. Late conflicts are resolved here if a parent was
    // already selected before this request is discovered.
    while (!worklist.empty() && isSolverOk(status)) {
      mlir::Operation *const op = worklist.pop_back_val();
      // Skip if we already traversed this op before
      if (traversedOps.find(op) != traversedOps.end()) {
        continue;
      }
      // Mark this op as traversed
      traversedOps[op] = true;

      // Find the selected candidate index for this op in the optimal path
      const size_t selectedCandidateIdx = optimalCandidateIndex.lookup(op);

      // Find the backtracking info for this op
      auto backtrackIt = backtrackTable.find(op);
      // Guard check against missing backtracking entry for this op
      if (backtrackIt == backtrackTable.end()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Backtracking: op {} candidate {} has missing "
                     "backtrack entry; stopping",
                     op->getName().getStringRef(), selectedCandidateIdx);
        status = SolverStatus::InvalidBacktrackState;
        break;
      }

      if (selectedCandidateIdx >= backtrackIt->second.size()) {
        // Selected candidate index is out-of-bounds
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Backtracking: op {} selected candidate index {} is "
                     "Out-Of-Bounds (backtrack table size {}); stopping",
                     op->getName().getStringRef(), selectedCandidateIdx,
                     backtrackIt->second.size());
        status = SolverStatus::InvalidBacktrackState;
        break;
      }

      // For each parent of current op, record the candidate index requested
      // by current op.
      // Linear op -> 1 parent. Join op -> multiple parents
      const llvm::SmallVector<int> &bestPrevIdxByParent =
          backtrackIt->second[selectedCandidateIdx];
      const llvm::SmallVector<mlir::Operation *> parents = getOpParents(op);

      // Retrieve parent's op info from backtrack and DP table
      if (bestPrevIdxByParent.size() < parents.size()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Backtracking: op {} candidate {} has {} backtrack "
                     "parents, expected {}; stopping",
                     op->getName().getStringRef(), selectedCandidateIdx,
                     bestPrevIdxByParent.size(), parents.size());
        status = SolverStatus::InvalidBacktrackState;
        break;
      }

      const size_t parentCount =
          std::min(parents.size(), bestPrevIdxByParent.size());
      for (size_t parentIdx = 0; parentIdx < parentCount; ++parentIdx) {
        const int parentCandidateIdx = bestPrevIdxByParent[parentIdx];
        // Guard check against invalid parent candidate index.
        if (parentCandidateIdx < 0) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "Backtracking: op {} candidate {} parentIdx {} has "
                       "invalid parent candidate {}; stopping",
                       op->getName().getStringRef(), selectedCandidateIdx,
                       parentIdx, parentCandidateIdx);
          status = SolverStatus::InvalidBacktrackState;
          break;
        }

        mlir::Operation *const parentOp = parents[parentIdx];
        // Find the corresponding parent candidate costs from DP table for this
        // parent op.
        auto parentCostIt = viterbiTable.find(parentOp);
        // Guard check against missing parent DP entry.
        if (parentCostIt == viterbiTable.end()) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "Backtracking: parent op {} requested by op {} "
                       "candidate {} has no DP entry; stopping",
                       parentOp->getName().getStringRef(),
                       op->getName().getStringRef(), selectedCandidateIdx);
          status = SolverStatus::InvalidBacktrackState;
          break;
        }

        const size_t parentCandIdxSize =
            static_cast<size_t>(parentCandidateIdx);
        if (parentCandIdxSize >= parentCostIt->second.size()) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "Backtracking: parent op {} candidate {} requested by "
                       "op {} candidate {} is Out-Of-Bounds (DP table size "
                       "{}); stopping",
                       parentOp->getName().getStringRef(), parentCandIdxSize,
                       op->getName().getStringRef(), selectedCandidateIdx,
                       parentCostIt->second.size());
          status = SolverStatus::InvalidCandidateSelection;
          break;
        }

        // Skip invalid parent candidates
        if (!std::isfinite(parentCostIt->second[parentCandIdxSize])) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "Backtracking: parent op {} candidate {} requested by "
                       "op {} candidate {} has non-finite DP cost {}; "
                       "stopping",
                       parentOp->getName().getStringRef(), parentCandIdxSize,
                       op->getName().getStringRef(), selectedCandidateIdx,
                       parentCostIt->second[parentCandIdxSize]);
          status = SolverStatus::InvalidCandidateSelection;
          break;
        }

        // Record that current op selects `selectedCandidateIdx` and expects
        // this parent op to use `parentCandIdxSize`. If the parent is shared,
        // multiple current ops may request different candidates.
        ParentCandRequest parentRequest{op, selectedCandidateIdx,
                                        parentCandIdxSize};

        // Find the selected candidate index for this parent op
        if (auto assignedIt = optimalCandidateIndex.find(parentOp);
            assignedIt != optimalCandidateIndex.end()) {
          // Record parent's information
          recordExpectedParentCand(parentOp, parentRequest);
          llvm::SmallVector<ParentCandRequest> &requests =
              expectedCandsByParent[parentOp];
          if (assignedIt->second != parentCandIdxSize) {
            // If x != y, resolve conflict
            std::optional<size_t> resolvedParentCandIdx =
                resolveParentCandConflict(parentOp, requests);
            // If resolved candidate is found, assign that candidate to this
            // parent op
            if (resolvedParentCandIdx) {
              const size_t previousParentCandIdx = assignedIt->second;
              assignedIt->second = *resolvedParentCandIdx;
              // Clean up the worklist and traversedOps
              if (*resolvedParentCandIdx != previousParentCandIdx) {
                removeRequestFromCurrentOp(parentOp);
                traversedOps.erase(parentOp);
                worklist.push_back(parentOp);
              }
            } else {
              parentOp->emitWarning()
                  << "Backtracking: parent op conflict with assigned "
                  << "candidate " << assignedIt->second
                  << ", requested candidate " << parentCandidateIdx
                  << ", no resolved candidate";
              status = SolverStatus::ConflictUnresolved;
              break;
            }
          }
          continue;
        }

        // Record this current op request for later selection.
        recordExpectedParentCand(parentOp, parentRequest);
      }
    }

    if (!isSolverOk(status)) {
      continue;
    }

    // Phase 2: Select the next unresolved parent from collected requests.
    // Prefer the latest scheduled parent op so later current ops are
    // traversed before earlier forked parents.
    mlir::Operation *nextParent = nullptr;
    std::optional<size_t> nextParentScheduleIdx;
    // Loop through all unresolved parent ops
    for (const auto &entry : expectedCandsByParent) {
      mlir::Operation *const parentOp = entry.first;
      // Skip already assigned parents.
      if (optimalCandidateIndex.find(parentOp) != optimalCandidateIndex.end()) {
        continue;
      }

      // Find the scheduled index of this parent op in the schedule.
      std::optional<size_t> parentScheduleIdx =
          optimizer_utils::getScheduledOpIndex(schedule, parentOp);

      // Guard check against missing schedule entry for this parent op
      if (!parentScheduleIdx) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Backtracking: unresolved parent op {} has no schedule "
                     "entry; stopping",
                     parentOp->getName().getStringRef());
        status = SolverStatus::InvalidBacktrackState;
        break;
      }

      // Assign the first encountered parent as the next parent to traverse if
      // nextParent is not assigned yet
      if (!nextParent) {
        nextParent = parentOp;
        nextParentScheduleIdx = parentScheduleIdx;
        continue;
      }

      // Select the requested parent op with the latest schedule index.
      if (*parentScheduleIdx <= *nextParentScheduleIdx) {
        continue;
      }

      nextParent = parentOp;
      nextParentScheduleIdx = parentScheduleIdx;
    }

    if (!isSolverOk(status)) {
      continue;
    }

    // Break the loop if no parent selected
    if (!nextParent) {
      break;
    }

    // Retrieve all requests for this selected parent
    const auto &requests = expectedCandsByParent[nextParent];

    size_t selectedParentCandidateIdx = requests.front().expectedParentCandIdx;
    bool hasConflict = false;

    // Check whether current op requests expect the same parent candidate.
    // If x == y, assign that parent candidate without conflict.
    for (const ParentCandRequest &request : requests) {
      if (request.expectedParentCandIdx != selectedParentCandidateIdx) {
        hasConflict = true;
        break;
      }
    }

    // If x != y, current op expect different candidates from this parent.
    if (hasConflict) {
      std::optional<size_t> resolvedParentCandIdx =
          resolveParentCandConflict(nextParent, requests);
      if (resolvedParentCandIdx) {
        selectedParentCandidateIdx = *resolvedParentCandIdx;
      } else {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Backtracking: parent op conflict with {} current op "
                     "requests has no resolved candidate; stopping instead "
                     "of using first requested candidate {}",
                     requests.size(), selectedParentCandidateIdx);
        status = SolverStatus::ConflictUnresolved;
      }
    }

    if (!isSolverOk(status)) {
      continue;
    }

    // Assign the parent candidate and traverse this parent in the next round.
    optimalCandidateIndex[nextParent] = selectedParentCandidateIdx;
    worklist.push_back(nextParent);
  }

  return status;
}

void ViterbiPolicy::reset() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "Resetting ViterbiPolicy state\n");
  viterbiTable.clear();
  backtrackTable.clear();
  optimalCandidateIndex.clear();
  candidateOutputSizes.clear();
}

// =================================================================
// HELPER FUNCTIONS
// =================================================================

void ViterbiPolicy::calculateTensorLifetimes(
    const OperationSchedule &schedule) {
  // Clear the map before calculating lifetimes
  tensorLifetimes.clear();

  auto updateLifetimeAt = [&](mlir::Value value, size_t opIdx) {
    auto [it, inserted] =
        tensorLifetimes.try_emplace(value, TensorLifetime{opIdx, opIdx, 1});
    if (!inserted) {
      it->second.startOpIdx = std::min(it->second.startOpIdx, opIdx);
      it->second.endOpIdx = std::max(it->second.endOpIdx, opIdx);
    }
  };

  for (const auto &entry : schedule) {
    // schedule is llvm::DenseMap<mlir::func::FuncOp,
    // llvm::SmallVector<mlir::Operation *>>
    const auto &ops = entry.second;

    for (size_t i = 0; i < ops.size(); ++i) {
      // Current scheduled operation at index i
      mlir::Operation *op = ops[i];

      for (mlir::Value result : op->getResults()) {
        // Result becomes live at defining op index in the pruned schedule.
        updateLifetimeAt(mapValueFromPrunedGraph(
                             result, prunedGraphInfo.prunedOpIndex),
                         i);
      }

      for (mlir::Value operand : op->getOperands()) {
        // Operand is live at this pruned op index. Values coming from removed
        // ops are folded back to their nearest kept source value.
        updateLifetimeAt(mapValueFromPrunedGraph(
                             operand, prunedGraphInfo.prunedOpIndex),
                         i);
      }
    }
  }

  for (auto &kv : tensorLifetimes) {
    auto &lt = kv.second;
    // Calculate span as endOpIdx - startOpIdx + 1
    lt.span = lt.endOpIdx - lt.startOpIdx + 1;
  }
}

SortedLifetimesMap
ViterbiPolicy::sortLifetimes(const TensorLifetimeMap &lifetimeMap) const {
  SortedLifetimesMap sorted;
  sorted.reserve(lifetimeMap.size());

  // Copy all (Value, TensorLifetime) pairs from the map into this vector
  for (const auto &kv : lifetimeMap) {
    sorted.push_back({kv.first, kv.second});
  }

  // Sort by spill priority:
  // 1. Older value first  -> smaller startOpIdx first
  // 2. If same startOpIdx, prefer shorter span -> smaller span first
  llvm::sort(sorted,
             [](const TensorLifetimePair &a, const TensorLifetimePair &b) {
               const TensorLifetime &lhs = a.second;
               const TensorLifetime &rhs = b.second;

               // Prefer older values first.
               if (lhs.startOpIdx != rhs.startOpIdx) {
                 return lhs.startOpIdx < rhs.startOpIdx;
               }

               // If they start at the same op, prefer the shorter live range
               if (lhs.span != rhs.span) {
                 return lhs.span < rhs.span;
               }

               // Tie-breaker: earlier end first (not gonna happen often)
               return lhs.endOpIdx < rhs.endOpIdx;
             });

  // sorted = [(value0, {startOpIdx, endOpIdx, span}),
  //           (value1, {startOpIdx, endOpIdx, span}),
  //           (value2, {startOpIdx, endOpIdx, span})]

  return sorted;
}

// =================================================================
// LIVE TENSOR HELPERS
// =================================================================

// Get all live activations at the currentOp execution point
LiveTensorList
ViterbiPolicy::getAllLiveActivations(mlir::Operation *currentOp) const {
  LiveTensorList liveActivations;
  if (!currentOp) {
    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                 "getAllLiveActivations: currentOp is null");
  } else {
    auto currentOpIdx = optimizer_utils::getScheduledOpIndex(schedule, currentOp);
    if (!currentOpIdx) {
      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                   "getAllLiveActivations: op {} is not in the schedule",
                   currentOp->getName().getStringRef());
    } else {
      TTMLIR_DEBUG(
          ttmlir::LogComponent::ViterbiOptimizer,
          "getAllLiveActivations: scanning live activations for op[{0}] {1}",
          *currentOpIdx, currentOp->getName().getStringRef());

      // Check if each value in tensorLifetimes is live at currentOpIdx
      for (const auto &entry : tensorLifetimes) {
        mlir::Value value = entry.first;
        if (!isActivationLiveAtOp(value, *currentOpIdx, currentOp)) {
          continue;
        }

        printActivation(currentOp, value, liveActivations,
                        /*requireRepresentativeMatch=*/true);
      }

      // Rule: older values first, if tie, shorter span first
      llvm::sort(liveActivations, [](const LiveTensorInfo &lhs,
                                     const LiveTensorInfo &rhs) {
        if (lhs.lifetime.startOpIdx != rhs.lifetime.startOpIdx) {
          return lhs.lifetime.startOpIdx < rhs.lifetime.startOpIdx;
        }
        if (lhs.lifetime.endOpIdx != rhs.lifetime.endOpIdx) {
          return lhs.lifetime.endOpIdx < rhs.lifetime.endOpIdx;
        }
        return lhs.value.getAsOpaquePointer() < rhs.value.getAsOpaquePointer();
      });

      TTMLIR_DEBUG(
          ttmlir::LogComponent::ViterbiOptimizer,
          "getAllLiveActivations: final ordered live activation count = {0}",
          liveActivations.size());
    }
  }

  return liveActivations;
}

// Get parent activations at the currentOp execution point
LiveTensorList
ViterbiPolicy::getParentActivations(mlir::Operation *currentOp) const {
  LiveTensorList activations;
  if (currentOp) {
    auto currentOpIdx = optimizer_utils::getScheduledOpIndex(schedule, currentOp);
    if (currentOpIdx) {
      // For first operation, treat operand[0] as the activation
      if (*currentOpIdx == 0) {
        if (currentOp->getNumOperands() != 0) {
          printActivation(currentOp, currentOp->getOperand(0), activations);
        }
      } else {
        // For later ops, collect only intermediate tensor operands when:
        // - must be a tensor
        // - must not be a block argument (function input)
        // - must be produced by some defining op
        // - must not be produced by ttcore.load_cached
        // - must have a recorded lifetime
        for (mlir::Value operand : currentOp->getOperands()) {
          printActivation(currentOp, operand, activations);
        }
      }
    }
  }

  return activations;
}

// Get parent operations that produce activations for the current join operation
llvm::SmallVector<mlir::Operation *>
ViterbiPolicy::getOpParents(mlir::Operation *currentOp) const {
  llvm::SmallVector<mlir::Operation *> parentOps;
  if (currentOp) {
    LiveTensorList parentActivations = getParentActivations(currentOp);
    parentOps.reserve(parentActivations.size());
    for (const LiveTensorInfo &activation : parentActivations) {
      mlir::Operation *parentOp = activation.value.getDefiningOp();
      if (!parentOp) {
        continue;
      }
      if (!llvm::is_contained(parentOps, parentOp)) {
        parentOps.push_back(parentOp);
      }
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                 "getOpParents: op {} parentOps=[{}]",
                 currentOp->getName().getStringRef(),
                 optimizer_utils::parentOpsToString(parentOps));
  }

  return parentOps;
}

// Store the output size of a candidate configuration for an operation
// This is used for future cost calculation of join operations that this op
// contributes to.
void ViterbiPolicy::storeCandidateOutputSize(
    mlir::Operation *op, size_t candidateIdx,
    std::optional<uint64_t> outputSizeBytes) {
  if (!op || !outputSizeBytes) {
    return;
  }

  // Only store numeric output size (valid candidates), otherwise leave it as
  // nullopt
  auto &storedSizes = candidateOutputSizes[op];
  if (storedSizes.size() <= candidateIdx) {
    storedSizes.resize(candidateIdx + 1, std::nullopt);
  }
  storedSizes[candidateIdx] = outputSizeBytes;
}

// Get passive activations that are live at the current op.
// Passive = All live - Parents
LiveTensorList
ViterbiPolicy::getPassiveActivations(mlir::Operation *currentOp) const {
  LiveTensorList passiveActivations;
  const LiveTensorList liveActivations = getAllLiveActivations(currentOp);

  if (!liveActivations.empty()) {
    // Non-empty live activations imply currentOp was present in the schedule;
    // getAllLiveActivations returns empty for null or unscheduled ops.
    const size_t currentOpIdx =
        *optimizer_utils::getScheduledOpIndex(schedule, currentOp);
    const LiveTensorList parentActivations = getParentActivations(currentOp);
    const bool includeRetainedParents = parentActivations.size() < 2;

    auto isParentActivation = [&](mlir::Value value) {
      return llvm::any_of(parentActivations,
                          [&](const LiveTensorInfo &parentActivation) {
                            return parentActivation.value == value;
                          });
    };

    for (const auto &activation : liveActivations) {
      const bool isParent = isParentActivation(activation.value);
      const bool retainedLinearParent =
          includeRetainedParents && isParent &&
          activation.lifetime.endOpIdx > currentOpIdx;

      // Baseline passive = all live minus parents.
      // Linear-only exception: include parents that remain live after current
      // op.
      if (!isParent || retainedLinearParent) {
        passiveActivations.push_back(activation);
      }
    }
  }

  return passiveActivations;
}

// Find producer operations for the passive activations. These producer ops are
// used as context for cost model to determine the cost of keeping passive
// tensors live through the current
llvm::SmallVector<mlir::Operation *> ViterbiPolicy::getPassiveTensorProducers(
    const LiveTensorList &passiveActivations) const {
  llvm::SmallVector<mlir::Operation *> passiveTensorProducers;
  passiveTensorProducers.reserve(passiveActivations.size());
  for (const LiveTensorInfo &passiveActivation : passiveActivations) {
    mlir::Operation *producerOp = passiveActivation.value.getDefiningOp();
    if (!producerOp) {
      continue;
    }
    // Avoid duplicates in passive tensor producers.
    if (!llvm::is_contained(passiveTensorProducers, producerOp)) {
      passiveTensorProducers.push_back(producerOp);
    }
  }
  return passiveTensorProducers;
}

// Return parent operands of the current consumer op that are produced by the
// given parent op
llvm::SmallVector<mlir::Value>
ViterbiPolicy::getParentOperandsForTransition(mlir::Operation *consumerOp,
                                              mlir::Operation *parentOp) const {
  llvm::SmallVector<mlir::Value> parentOperands;
  if (consumerOp && parentOp) {
    for (mlir::Value operand : consumerOp->getOperands()) {
      if (!mlir::isa<mlir::RankedTensorType>(operand.getType())) {
        continue;
      }

      mlir::Value representativeValue = mapValueFromPrunedGraph(
          operand, prunedGraphInfo.prunedOpIndex);
      if (representativeValue.getDefiningOp() == parentOp) {
        // Keep the original consumer operand: CostModel uses it to recover the
        // consumer input-layout index.
        parentOperands.push_back(operand);
      }
    }
  }

  return parentOperands;
}

// Check if a given activation value is live at the current operation execution
// point
bool ViterbiPolicy::isActivationLiveAtOp(mlir::Value value, size_t currentOpIdx,
                                         mlir::Operation *currentOp) const {
  bool isLive = false;
  auto it = tensorLifetimes.find(value);
  if (it != tensorLifetimes.end()) {
    const TensorLifetime &lifetime = it->second;
    isLive = lifetime.startOpIdx <= currentOpIdx &&
             lifetime.endOpIdx >= currentOpIdx;

    // Query the live set immediately before the current operation executes.
    // That excludes values first produced by the current operation itself.
    if (isLive) {
      if (mlir::Operation *definingOp = value.getDefiningOp()) {
        if (definingOp == currentOp) {
          isLive = false;
        }
      }
    }
  }

  return isLive;
}

// Get detailed information of activation tensors at current op
// Currently serving for debugging log only
std::optional<LiveTensorInfo>
ViterbiPolicy::getActivationInfo(mlir::Operation *currentOp,
                                 mlir::Value value) const {
  std::optional<LiveTensorInfo> activationInfo;

  // Find the representative value in the pruned graph
  auto buildLiveTensorInfo =
      [&](mlir::Value representativeValue) -> std::optional<LiveTensorInfo> {
    std::optional<LiveTensorInfo> info;
    auto it = tensorLifetimes.find(representativeValue);
    if (it != tensorLifetimes.end()) {
      info = LiveTensorInfo{
          representativeValue, it->second,
          optimizer_utils::extractLayoutFromValue(representativeValue)};
    }

    return info;
  };

  if (currentOp) {
    // Find equivalent op index in the pruned schedule
    auto currentOpIdx = optimizer_utils::getScheduledOpIndex(schedule, currentOp);
    // If current op is in the schedule
    if (currentOpIdx) {
      // Special handling for the first op: only consider its operand[0] as the
      // activation
      if (*currentOpIdx == 0) {
        if (currentOp->getNumOperands() != 0) {
          mlir::Value firstOperand = currentOp->getOperand(0);
          mlir::Value representativeValue =
              mapValueFromPrunedGraph(
                  firstOperand, prunedGraphInfo.prunedOpIndex);

          bool isTensorBlockArgumentActivation =
              llvm::isa<mlir::BlockArgument>(representativeValue) &&
              mlir::isa<mlir::TensorType>(representativeValue.getType());
          bool isStandardActivation =
              optimizer_utils::isTensorActivationOperand(representativeValue);
          bool isActivation =
              isStandardActivation | isTensorBlockArgumentActivation;

          if (value == representativeValue && isActivation) {
            activationInfo = buildLiveTensorInfo(representativeValue);
          }
        }
      } else if (optimizer_utils::isTensorActivationOperand(value)) {
        mlir::Value representativeValue =
            mapValueFromPrunedGraph(
                value, prunedGraphInfo.prunedOpIndex);

        // For non-first ops, keep activation semantics after representative
        // mapping. This filters out operands that map back to block arguments
        // (e.g. weights / constants routed through removed ops), which would
        // otherwise be counted as live/parent activations and distort passive
        // detection for linear ops.
        if (optimizer_utils::isTensorActivationOperand(representativeValue)) {
          activationInfo = buildLiveTensorInfo(representativeValue);
        }
      }
    }
  }

  return activationInfo;
}

// Check if the current operation is a join point (i.e., has multiple
// activations)
bool ViterbiPolicy::isJoinOperation(mlir::Operation *op) const {
  bool isJoin = false;
  if (op) {
    auto activations = getParentActivations(op);
    isJoin = activations.size() >= 2;
  }

  return isJoin;
}

// Find the best transition cost between a parent candidate and a current
// candidate by checking all pairs that connect parentOp to currentOp.
double ViterbiPolicy::getBestTransitionCost(
    mlir::Operation *parentOp, const OpConfigCandidate &parentCandidate,
    mlir::Operation *currentOp, const OpConfigCandidate &currentCandidate,
    uint64_t additionalL1Usage) const {
  double transitionCost = inf;
  const llvm::SmallVector<mlir::Value> parentOperands =
      getParentOperandsForTransition(currentOp, parentOp);

  // Feed required information to the cost model
  for (mlir::Value parentOperand : parentOperands) {
    CostModel::TransitionEdgeCostInput transitionInput;
    transitionInput.producerOp = parentOp;
    transitionInput.consumerOp = currentOp;
    transitionInput.consumerOperand = parentOperand;
    transitionInput.producerCandidate = &parentCandidate;
    transitionInput.consumerCandidate = &currentCandidate;
    transitionInput.opConfigMap = nullptr;
    transitionInput.additionalL1Usage = additionalL1Usage;

    const double edgeCost = costModel->getTransitionCost(transitionInput);
    if (edgeCost < transitionCost) {
      transitionCost = edgeCost;
    }
  }

  return transitionCost;
}

// Find scheduled sink ops and initialize the backtracking seeds.
llvm::SmallVector<std::pair<mlir::Operation *, size_t>>
ViterbiPolicy::getBacktrackingSeeds() const {
  llvm::SmallVector<std::pair<mlir::Operation *, size_t>> seeds;

  for (const auto &[_, operations] : schedule) {
    llvm::SmallPtrSet<mlir::Operation *, 16> scheduledOps(operations.begin(),
                                                          operations.end());

    for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
      mlir::Operation *const sinkOp = *it;

      // Check if this sinkOp has any user that is also scheduled
      // If none of its users are scheduled, this sinkOp can be a backtracking
      // seed
      bool isScheduledSink = true;
      for (mlir::OpResult result : sinkOp->getResults()) {
        for (mlir::Operation *user : result.getUsers()) {
          if (scheduledOps.contains(user)) {
            isScheduledSink = false;
            TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                         "Skipping sink op {} as it has scheduled user {}",
                         sinkOp->getName().getStringRef(),
                         user->getName().getStringRef());
            break;
          }
        }
        if (!isScheduledSink) {
          break;
        }
      }
      if (!isScheduledSink) {
        continue;
      }

      auto costIt = viterbiTable.find(sinkOp);
      // Guard check against missing DP entry or empty candidate costs.
      if (costIt == viterbiTable.end() || costIt->second.empty()) {
        continue;
      }

      const auto &costs = costIt->second;
      size_t bestIdx = 0;
      double bestCost = inf;
      for (size_t candIdx = 0; candIdx < costs.size(); ++candIdx) {
        // Find the best finite candidate with minimum cost for this sink op.
        if (!std::isfinite(costs[candIdx])) {
          continue;
        }
        if (costs[candIdx] < bestCost) {
          bestCost = costs[candIdx];
          bestIdx = candIdx;
        }
      }

      // Skip if no valid candidate exists for this sink op.
      if (std::isfinite(bestCost)) {
        seeds.push_back({sinkOp, bestIdx});
      }
    }
  }

  return seeds;
}

// Resolve x != y case. Example: op4[b] expects op2[x], op3[a] expects op2[y]
// Try every valid op2 candidate and pick the one with the lowest
// accumulated parent DP cost plus transition costs to all selected
// current ops.
std::optional<size_t> ViterbiPolicy::resolveParentCandConflict(
    mlir::Operation *parentOp,
    llvm::ArrayRef<ParentCandRequest> requests) const {
  std::optional<size_t> resolvedCandIdx = std::nullopt;
  double bestCost = inf;

  // Get dp[parentOp][j]. The DP table stores one cost per op candidate
  auto parentDpIt = viterbiTable.find(parentOp);
  auto parentCandidatesIt = candidateResult.candidateMap.find(parentOp);

  // Guard check against missing parent DP entry, missing parent candidates,
  // or no current op requests.
  const bool hasParentState =
      parentDpIt != viterbiTable.end() &&
      parentCandidatesIt != candidateResult.candidateMap.end() &&
      !requests.empty();

  if (hasParentState) {
    const auto &parentDp = parentDpIt->second;
    const auto &parentCandidates = parentCandidatesIt->second;
    // Check how many parent that need to do the sum calculation
    const size_t parentCandCount =
        std::min(parentDp.size(), parentCandidates.size());

    // score[j] = dp[parentOp][j]
    //          + sum transition(parentOp[j]
    //                           -> currentOp[selectedCurrentCand])
    for (size_t parentCandIdx = 0; parentCandIdx < parentCandCount;
         ++parentCandIdx) {
      // Skip invalid parent candidates
      if (!std::isfinite(parentDp[parentCandIdx])) {
        continue;
      }

      // Start with dp[parentOp][parentCandIdx].
      double candidateCost = parentDp[parentCandIdx];
      bool candidateValid = true;
      for (const ParentCandRequest &request : requests) {
        // Each request contributes one transition from this parent candidate
        // to the selected candidate of a current op.
        auto currentCandidatesIt =
            candidateResult.candidateMap.find(request.currentOp);
        const bool hasCurrentCandidate =
            currentCandidatesIt != candidateResult.candidateMap.end() &&
            request.selectedCurrentCandIdx < currentCandidatesIt->second.size();
        if (!hasCurrentCandidate) {
          candidateValid = false;
          break;
        }

        // Same logic with forward DP
        const double transitionCost = getBestTransitionCost(
            parentOp, parentCandidates[parentCandIdx], request.currentOp,
            currentCandidatesIt->second[request.selectedCurrentCandIdx],
            /*additionalL1Usage=*/0);

        // Guard check against invalid transition cost to avoid selecting
        // invalid parent candidates
        if (!std::isfinite(transitionCost)) {
          candidateValid = false;
          break;
        }
        candidateCost += transitionCost;
      }

      // Keep the best finite shared parent candidate.
      if (candidateValid && candidateCost < bestCost) {
        bestCost = candidateCost;
        resolvedCandIdx = parentCandIdx;
      }
    }
  }

  return resolvedCandIdx;
}

// =================================================================
// LOGGING AND DEBUGGING
// =================================================================

void ViterbiPolicy::printOptimalPath() const {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "=== Optimal Path Summary ===\n");

  [[maybe_unused]] size_t totalOps = 0;
  [[maybe_unused]] double totalCost = 0.0;

  for (const auto &[func, operations] : schedule) {
    for ([[maybe_unused]] size_t opIdx = 0; opIdx < operations.size();
         ++opIdx) {
      mlir::Operation *op = operations[opIdx];
      auto costIt = viterbiTable.find(op);
      if (costIt == viterbiTable.end()) {
        continue;
      }

      const auto &costs = costIt->second;
      if (!costs.empty()) {
        size_t bestIdx = 0;
        double bestCost = 0.0;
        bool foundBest = false;

        if (auto bestIt = optimalCandidateIndex.find(op);
            bestIt != optimalCandidateIndex.end() &&
            bestIt->second < costs.size()) {
          bestIdx = bestIt->second;
          bestCost = costs[bestIdx];
          foundBest = true;
        }

        if (!foundBest) {
          auto minIt = costs.begin();
          for (auto it = costs.begin() + 1; it != costs.end(); ++it) {
            if (*it < *minIt) {
              minIt = it;
            }
          }
          bestIdx = static_cast<size_t>(minIt - costs.begin());
          bestCost = *minIt;
        }

        totalCost += bestCost;
        totalOps++;

        auto candIt = candidateResult.candidateMap.find(op);
        if (candIt == candidateResult.candidateMap.end() ||
            bestIdx >= candIt->second.size()) {
          continue;
        }

        const auto &bestCandidate = candIt->second[bestIdx];
        std::vector<TTNNLayoutAttr> inputLayouts(
            bestCandidate.inputLayouts.begin(),
            bestCandidate.inputLayouts.end());
        std::optional<TTNNLayoutAttr> outputLayout =
            bestCandidate.opConfig.outputLayout;

        [[maybe_unused]] std::string inputLayoutsStr =
            optimizer_utils::layoutsToString(inputLayouts);
        [[maybe_unused]] std::string outputLayoutStr =
            optimizer_utils::layoutToString(outputLayout);

        // Count valid candidates (finite cost entries) for this operation
        [[maybe_unused]] size_t numValidCandidates = static_cast<size_t>(
            std::count_if(costs.begin(), costs.end(),
                          [](double c) { return std::isfinite(c); }));

        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Op[{0}]: {1} -> Candidate {2} (Cost: {3})\n"
                     "Valid candidates: {4}\n"
                     "Input layouts: {5}\n"
                     "Output layout: {6}\n",
                     opIdx, op->getName().getStringRef(), bestIdx, bestCost,
                     numValidCandidates, inputLayoutsStr, outputLayoutStr);
      } else {
        continue;
      }
    }
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "Total operations: {0}, Total cost: {1}\n", totalOps, totalCost);
}

void ViterbiPolicy::printActivation(mlir::Operation *currentOp,
                                    mlir::Value value,
                                    LiveTensorList &activations,
                                    bool requireRepresentativeMatch) const {
  auto activationInfo = getActivationInfo(currentOp, value);
  const bool shouldPrint = activationInfo && (!requireRepresentativeMatch ||
                                              activationInfo->value == value);

  if (shouldPrint) {
    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                 "  activation {0}: start={1} end={2} span={3} layout={4}",
                 optimizer_utils::formatValueShort(activationInfo->value),
                 activationInfo->lifetime.startOpIdx,
                 activationInfo->lifetime.endOpIdx,
                 activationInfo->lifetime.span,
                 optimizer_utils::layoutToString(activationInfo->layout));

    activations.push_back(*activationInfo);
  }
}

} // namespace mlir::tt::ttnn
