// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ViterbiPolicy.h"

#include "ttmlir/Dialect/TTNN/Analysis/CostModel.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {

mlir::Value mapValueFromPrunedGraph(
    mlir::Value value,
    const llvm::DenseMap<mlir::Operation *, std::size_t> &prunedOpIndex) {
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

// Construct the policy from precomputed candidates and the selected schedule.
// Pruned graph metadata preserves representative SSA values across conversions.
ViterbiPolicy::ViterbiPolicy(const OpCandidateBuilderResult &candidateResult,
                             const OperationSchedule &schedule,
                             PrunedGraphInfo prunedGraphInfo)
    : candidateResult(candidateResult), schedule(schedule),
      costModel(std::make_shared<CostModel>()),
      prunedGraphInfo(std::move(prunedGraphInfo)) {}

void ViterbiPolicy::reset() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "Resetting ViterbiPolicy state\n");

  viterbiTable.clear();
  backtrackTable.clear();
  optimalCandidateIndex.clear();
  candidateOutputSizes.clear();
  candidateAdditionalL1Usages.clear();
  selectedSpillCounts.clear();
}

void ViterbiPolicy::calculateTensorLifetimes(
    const OperationSchedule &schedule) {
  // Clear stale lifetime state before recalculating it.
  tensorLifetimes.clear();

  auto updateLifetimeAt = [&](mlir::Value value, std::size_t operationIndex) {
    auto [lifetimeIt, inserted] = tensorLifetimes.try_emplace(
        value, TensorLifetime{operationIndex, operationIndex, 1});
    if (!inserted) {
      lifetimeIt->second.startOpIdx =
          std::min(lifetimeIt->second.startOpIdx, operationIndex);
      lifetimeIt->second.endOpIdx =
          std::max(lifetimeIt->second.endOpIdx, operationIndex);
    }
  };

  for (const auto &entry : schedule) {
    const auto &ops = entry.second;

    for (std::size_t operationIndex = 0; operationIndex < ops.size();
         ++operationIndex) {
      mlir::Operation *op = ops[operationIndex];

      for (mlir::Value result : op->getResults()) {
        // Result becomes live at defining op index in the pruned schedule.
        updateLifetimeAt(
            mapValueFromPrunedGraph(result, prunedGraphInfo.prunedOpIndex),
            operationIndex);
      }

      for (mlir::Value operand : op->getOperands()) {
        // Operand is live at this pruned op index. Values coming from removed
        // ops are folded back to their nearest kept source value.
        updateLifetimeAt(
            mapValueFromPrunedGraph(operand, prunedGraphInfo.prunedOpIndex),
            operationIndex);
      }
    }
  }

  for (auto &lifetimeEntry : tensorLifetimes) {
    TensorLifetime &lifetime = lifetimeEntry.second;
    // Include both defining and last-use operations in the live-range span.
    lifetime.span = lifetime.endOpIdx - lifetime.startOpIdx + 1;
  }
}

SortedLifetimesMap
ViterbiPolicy::sortLifetimes(const TensorLifetimeMap &lifetimeMap) const {
  SortedLifetimesMap sorted;
  sorted.reserve(lifetimeMap.size());

  // Copy lifetime records so they can be sorted without mutating the map.
  for (const auto &lifetimeEntry : lifetimeMap) {
    sorted.push_back({lifetimeEntry.first, lifetimeEntry.second});
  }

  // Sort older values first, then prefer the shorter live range when two
  // values become live at the same operation.
  llvm::sort(sorted, [](const TensorLifetimePair &lhsEntry,
                        const TensorLifetimePair &rhsEntry) {
    const TensorLifetime &lhs = lhsEntry.second;
    const TensorLifetime &rhs = rhsEntry.second;

    // Prefer older values first.
    if (lhs.startOpIdx != rhs.startOpIdx) {
      return lhs.startOpIdx < rhs.startOpIdx;
    }

    // If both values start together, prefer the shorter live
    // range.
    if (lhs.span != rhs.span) {
      return lhs.span < rhs.span;
    }

    // Use the earlier end index as the final deterministic
    // tie-breaker.
    return lhs.endOpIdx < rhs.endOpIdx;
  });

  return sorted;
}

void ViterbiPolicy::storeCandidateOutputSize(
    mlir::Operation *op, std::size_t candidateIdx,
    std::optional<uint64_t> outputSizeBytes) {
  if (!op || !outputSizeBytes) {
    return;
  }

  // Keep invalid candidates unset so later passive-memory queries can skip
  // them.
  auto &storedSizes = candidateOutputSizes[op];
  if (storedSizes.size() <= candidateIdx) {
    storedSizes.resize(candidateIdx + 1, std::nullopt);
  }
  storedSizes[candidateIdx] = outputSizeBytes;
}

void ViterbiPolicy::storeCandidateAdditionalL1Usage(
    mlir::Operation *op, std::size_t candidateIdx, uint64_t additionalL1Usage) {
  if (!op) {
    return;
  }

  auto &storedUsages = candidateAdditionalL1Usages[op];
  if (storedUsages.size() <= candidateIdx) {
    storedUsages.resize(candidateIdx + 1, std::nullopt);
  }

  storedUsages[candidateIdx] = additionalL1Usage;
}

void ViterbiPolicy::storeSelectedSpillCount(mlir::Operation *op,
                                            std::size_t candidateIdx,
                                            std::size_t selectedSpillCount) {
  if (!op) {
    return;
  }

  auto &spillCounts = selectedSpillCounts[op];
  if (spillCounts.size() <= candidateIdx) {
    spillCounts.resize(candidateIdx + 1, 0);
  }

  spillCounts[candidateIdx] = selectedSpillCount;
}

std::optional<uint64_t>
ViterbiPolicy::getCandidateAdditionalL1Usage(mlir::Operation *op,
                                             std::size_t candidateIdx) const {
  auto usageIt = candidateAdditionalL1Usages.find(op);
  if (usageIt == candidateAdditionalL1Usages.end() ||
      candidateIdx >= usageIt->second.size()) {
    return std::nullopt;
  }

  return usageIt->second[candidateIdx];
}

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

llvm::SmallVector<mlir::Value>
ViterbiPolicy::getParentOperandsForTransition(mlir::Operation *consumerOp,
                                              mlir::Operation *parentOp) const {
  llvm::SmallVector<mlir::Value> parentOperands;
  if (consumerOp && parentOp) {
    for (mlir::Value operand : consumerOp->getOperands()) {
      if (!mlir::isa<mlir::RankedTensorType>(operand.getType())) {
        continue;
      }

      mlir::Value representativeValue =
          mapValueFromPrunedGraph(operand, prunedGraphInfo.prunedOpIndex);
      if (representativeValue.getDefiningOp() == parentOp) {
        // Keep the original consumer operand: CostModel uses it to recover the
        // consumer input-layout index.
        parentOperands.push_back(operand);
      }
    }
  }

  return parentOperands;
}

bool ViterbiPolicy::isActivationLiveAtOp(mlir::Value value,
                                         std::size_t currentOpIdx,
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

std::optional<LiveTensorInfo>
ViterbiPolicy::getActivationInfo(mlir::Operation *currentOp,
                                 mlir::Value value) const {
  std::optional<LiveTensorInfo> activationInfo;

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
    auto currentOpIdx =
        optimizer_utils::getScheduledOpIndex(schedule, currentOp);
    if (currentOpIdx) {
      // The first scheduled operation can consume a block argument, so its
      // first tensor operand is treated as the activation.
      if (*currentOpIdx == 0) {
        if (currentOp->getNumOperands() != 0) {
          mlir::Value firstOperand = currentOp->getOperand(0);
          mlir::Value representativeValue = mapValueFromPrunedGraph(
              firstOperand, prunedGraphInfo.prunedOpIndex);

          bool isTensorBlockArgumentActivation =
              llvm::isa<mlir::BlockArgument>(representativeValue) &&
              mlir::isa<mlir::TensorType>(representativeValue.getType());
          bool isStandardActivation =
              optimizer_utils::isTensorActivationOperand(representativeValue);
          bool isActivation =
              isStandardActivation || isTensorBlockArgumentActivation;

          if (value == representativeValue && isActivation) {
            activationInfo = buildLiveTensorInfo(representativeValue);
          }
        }
      } else if (optimizer_utils::isTensorActivationOperand(value)) {
        mlir::Value representativeValue =
            mapValueFromPrunedGraph(value, prunedGraphInfo.prunedOpIndex);

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

LiveTensorList
ViterbiPolicy::getAllLiveActivations(mlir::Operation *currentOp) const {
  LiveTensorList liveActivations;
  if (!currentOp) {
    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                 "getAllLiveActivations: currentOp is null");
  } else {
    auto currentOpIdx =
        optimizer_utils::getScheduledOpIndex(schedule, currentOp);
    if (!currentOpIdx) {
      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                   "getAllLiveActivations: op {} is not in the schedule",
                   currentOp->getName().getStringRef());
    } else {
      TTMLIR_DEBUG(
          ttmlir::LogComponent::ViterbiOptimizer,
          "getAllLiveActivations: scanning live activations for op[{0}] {1}",
          *currentOpIdx, currentOp->getName().getStringRef());

      for (const auto &entry : tensorLifetimes) {
        mlir::Value value = entry.first;
        if (!isActivationLiveAtOp(value, *currentOpIdx, currentOp)) {
          continue;
        }

        printActivation(currentOp, value, liveActivations,
                        /*requireRepresentativeMatch=*/true);
      }

      // Keep a deterministic spill order with older activations first.
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

LiveTensorList
ViterbiPolicy::getParentActivations(mlir::Operation *currentOp) const {
  LiveTensorList activations;
  if (currentOp) {
    auto currentOpIdx =
        optimizer_utils::getScheduledOpIndex(schedule, currentOp);
    if (currentOpIdx) {
      // The first operation may consume a block-argument activation.
      if (*currentOpIdx == 0) {
        if (currentOp->getNumOperands() != 0) {
          printActivation(currentOp, currentOp->getOperand(0), activations);
        }
      } else {
        // Later operations only consider intermediate tensor values with a
        // defining operation and recorded lifetime. Block arguments and values
        // produced by ttcore.load_cached are filtered by printActivation().
        for (mlir::Value operand : currentOp->getOperands()) {
          printActivation(currentOp, operand, activations);
        }
      }
    }
  }

  return activations;
}

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

LiveTensorList
ViterbiPolicy::getPassiveActivations(mlir::Operation *currentOp) const {
  LiveTensorList passiveActivations;
  const LiveTensorList liveActivations = getAllLiveActivations(currentOp);

  if (!liveActivations.empty()) {
    // Non-empty live activations imply currentOp was present in the schedule;
    // getAllLiveActivations returns empty for null or unscheduled ops.
    const std::size_t currentOpIdx =
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

bool ViterbiPolicy::isJoinOperation(mlir::Operation *op) const {
  bool isJoin = false;
  if (op) {
    auto activations = getParentActivations(op);
    isJoin = activations.size() >= 2;
  }

  return isJoin;
}

double ViterbiPolicy::getBestTransitionCost(
    mlir::Operation *parentOp, const OpConfigCandidate &parentCandidate,
    mlir::Operation *currentOp, const OpConfigCandidate &currentCandidate,
    uint64_t additionalL1Usage) const {
  double transitionCost = inf;

  const llvm::SmallVector<mlir::Value> parentOperands =
      getParentOperandsForTransition(currentOp, parentOp);

  auto getEdgeCost = [&](mlir::Value parentOperand) {
    CostModel::TransitionEdgeCostInput transitionInput;
    transitionInput.producerOp = parentOp;
    transitionInput.consumerOp = currentOp;
    transitionInput.consumerOperand = parentOperand;
    transitionInput.producerCandidate = &parentCandidate;
    transitionInput.consumerCandidate = &currentCandidate;
    transitionInput.opConfigMap = nullptr;
    transitionInput.additionalL1Usage = additionalL1Usage;

    return costModel->getTransitionCost(transitionInput);
  };

  if (parentOperands.size() == 1) {
    return getEdgeCost(parentOperands.front());
  }

  if (parentOperands.size() > 1) {
    transitionCost = 0.0;

    for (mlir::Value parentOperand : parentOperands) {
      const double edgeCost = getEdgeCost(parentOperand);

      if (!std::isfinite(edgeCost)) {
        return inf;
      }

      transitionCost += edgeCost;
    }
  }

  return transitionCost;
}

llvm::SmallVector<std::pair<mlir::Operation *, std::size_t>>
ViterbiPolicy::getBacktrackingSeeds() const {
  llvm::SmallVector<std::pair<mlir::Operation *, std::size_t>> seeds;

  for (const auto &[_, operations] : schedule) {
    llvm::SmallPtrSet<mlir::Operation *, 16> scheduledOps(operations.begin(),
                                                          operations.end());

    for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
      mlir::Operation *const sinkOp = *it;

      // A scheduled operation is a seed only when none of its result users
      // are also scheduled.
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
      std::size_t bestIdx = 0;
      double bestCost = inf;
      for (std::size_t candIdx = 0; candIdx < costs.size(); ++candIdx) {
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

std::optional<std::size_t> ViterbiPolicy::resolveParentCandConflict(
    mlir::Operation *parentOp,
    llvm::ArrayRef<ParentCandRequest> requests) const {
  std::optional<std::size_t> resolvedCandIdx = std::nullopt;
  double bestCost = inf;

  // Read one DP cost for each candidate of the shared parent.
  auto parentDpIt = viterbiTable.find(parentOp);
  auto parentCandidatesIt = candidateResult.candidateMap.find(parentOp);

  // Conflict resolution requires parent DP state, candidates, and requests.
  const bool hasParentState =
      parentDpIt != viterbiTable.end() &&
      parentCandidatesIt != candidateResult.candidateMap.end() &&
      !requests.empty();

  if (hasParentState) {
    const auto &parentDp = parentDpIt->second;
    const auto &parentCandidates = parentCandidatesIt->second;
    const std::size_t parentCandCount =
        std::min(parentDp.size(), parentCandidates.size());

    // Score each parent candidate using its DP cost plus all required
    // transitions to the selected consumer candidates.
    for (std::size_t parentCandIdx = 0; parentCandIdx < parentCandCount;
         ++parentCandIdx) {
      // Invalid parent candidates cannot resolve a shared-parent conflict.
      if (!std::isfinite(parentDp[parentCandIdx])) {
        continue;
      }

      // Start with dp[parentOp][parentCandIdx].
      double candidateCost = parentDp[parentCandIdx];
      bool candidateValid = true;
      for (const ParentCandRequest &request : requests) {
        // Each request contributes the transition cost to one selected
        // consumer.
        auto currentCandidatesIt =
            candidateResult.candidateMap.find(request.currentOp);
        const bool hasCurrentCandidate =
            currentCandidatesIt != candidateResult.candidateMap.end() &&
            request.selectedCurrentCandIdx < currentCandidatesIt->second.size();
        if (!hasCurrentCandidate) {
          candidateValid = false;
          break;
        }

        const double transitionCost = getBestTransitionCost(
            parentOp, parentCandidates[parentCandIdx], request.currentOp,
            currentCandidatesIt->second[request.selectedCurrentCandIdx],
            getCandidateAdditionalL1Usage(request.currentOp,
                                          request.selectedCurrentCandIdx)
                .value_or(0));

        // Reject a parent candidate when any requested transition is invalid.
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

SolverStatus ViterbiPolicy::buildViterbiTable() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "Building Viterbi tables");
  SolverStatus status = SolverStatus::Success;

  for (const auto &[func, operations] : schedule) {
    for (auto *op : operations) {
      auto it = candidateResult.candidateMap.find(op);
      if (it == candidateResult.candidateMap.end() || it->second.empty()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "BuildViterbiTable: op {} has missing candidate state",
                     op->getName().getStringRef());
        status = SolverStatus::MissingCandidateState;
        continue;
      }

      const auto &candidates = it->second;

      std::size_t numCandidates = candidates.size();
      viterbiTable[op].resize(numCandidates);
      backtrackTable[op].resize(numCandidates);

      for (std::size_t candidateIndex = 0; candidateIndex < numCandidates;
           ++candidateIndex) {
        viterbiTable[op][candidateIndex] = inf;
        backtrackTable[op][candidateIndex].clear();
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

  // Evaluate operations in scheduled topological order.
  for (const auto &entry : schedule) {
    if (stopCostCalculation) {
      break;
    }

    const auto &ops = entry.second;
    for (std::size_t opIdx = 0; opIdx < ops.size() && !stopCostCalculation;
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
      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer, "Op {} isJoin={}",
                   op->getName().getStringRef(), isJoin);

      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                   "Calculating {} costs for op {} with {} candidates",
                   isJoin ? "join" : "linear", op->getName().getStringRef(),
                   candidates.size());

      // dp[u][i] = local[u][i] + sum_p min_j(dp[p][j] + t(p_j -> u_i)).
      for (std::size_t candidateIndex = 0;
           candidateIndex < candidates.size();) {
        const OpConfigCandidate &candidate = candidates[candidateIndex];

        // Retrieve local cost and output size for current candidate.
        const CostModel::LocalCostResult localResult = costModel->getLocalCost(
            op, candidate, passiveTensorProducers,
            /*shouldComputeOutputSize=*/true, candidateOutputSizes);

        storeCandidateOutputSize(op, candidateIndex,
                                 localResult.outputSizeBytes);
        storeCandidateAdditionalL1Usage(op, candidateIndex,
                                        localResult.additionalL1Usage);
        storeSelectedSpillCount(op, candidateIndex,
                                localResult.selectedSpillCount);

        // A candidate remains unreachable until both its local cost and
        // parent paths are valid.
        double totalCost = inf;
        llvm::SmallVector<int> bestPrevIdxByParent(opParents.size(), -1);

        if (std::isfinite(localResult.cost)) {
          double parentCostSum = 0.0;
          bool parentPathValid = true;

          for (std::size_t parentIdx = 0; parentIdx < opParents.size();
               ++parentIdx) {
            mlir::Operation *const parentOp = opParents[parentIdx];
            auto parentCandidatesIt =
                candidateResult.candidateMap.find(parentOp);
            auto parentDpIt = viterbiTable.find(parentOp);
            if (parentCandidatesIt == candidateResult.candidateMap.end() ||
                parentDpIt == viterbiTable.end()) {
              TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                           "DP op={} candidate={} -> inf: parent {} missing "
                           "{}{}",
                           op->getName().getStringRef(), candidateIndex,
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

            const std::size_t parentCandidateCount =
                std::min(parentCandidates.size(), parentDp.size());

            for (std::size_t parentCandidateIndex = 0;
                 parentCandidateIndex < parentCandidateCount;
                 ++parentCandidateIndex) {
              if (!std::isfinite(parentDp[parentCandidateIndex])) {
                continue;
              }

              const double bestEdgeCost = getBestTransitionCost(
                  parentOp, parentCandidates[parentCandidateIndex], op,
                  candidate, localResult.additionalL1Usage);

              const double parentCandidateCost =
                  parentDp[parentCandidateIndex] + bestEdgeCost;
              if (parentCandidateCost < bestParentCost) {
                bestParentCost = parentCandidateCost;
                bestParentIdx = static_cast<int>(parentCandidateIndex);
                TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                             "DP best edge op={} c={} parent={} pc={} "
                             "parentDp={} edge={} combined={}",
                             op->getName().getStringRef(), candidateIndex,
                             parentOp->getName().getStringRef(),
                             parentCandidateIndex,
                             parentDp[parentCandidateIndex], bestEdgeCost,
                             parentCandidateCost);
              }
            }

            if (!std::isfinite(bestParentCost)) {
              TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                           "DP op={} candidate={} -> inf: no finite path from "
                           "parent {}",
                           op->getName().getStringRef(), candidateIndex,
                           parentOp->getName().getStringRef());
              parentPathValid = false;
              break;
            }

            parentCostSum += bestParentCost;
            bestPrevIdxByParent[parentIdx] = bestParentIdx;
          }

          if (parentPathValid) {
            totalCost = localResult.cost + parentCostSum;
          } else {
            std::fill(bestPrevIdxByParent.begin(), bestPrevIdxByParent.end(),
                      -1);
          }
        } else {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "DP op={} candidate={} -> inf: localCost=inf",
                       op->getName().getStringRef(), candidateIndex);
        }

        // Store the Viterbi recurrence and its minimizing parent candidate
        // for every parent edge.
        viterbiTable[op][candidateIndex] = totalCost;
        backtrackTable[op][candidateIndex] = bestPrevIdxByParent;
        TTMLIR_DEBUG(
            ttmlir::LogComponent::ViterbiOptimizer,
            "DP op={} c={} local={} parentSum={} total={} btSize={}",
            op->getName().getStringRef(), candidateIndex, localResult.cost,
            std::isfinite(totalCost) ? totalCost - localResult.cost : inf,
            totalCost, bestPrevIdxByParent.size());
        for (std::size_t parentIdx = 0; parentIdx < bestPrevIdxByParent.size();
             ++parentIdx) {
          TTMLIR_DEBUG(
              ttmlir::LogComponent::ViterbiOptimizer,
              "DP backtrack op={} c={} parentIdx={} parentCandidate={}",
              op->getName().getStringRef(), candidateIndex, parentIdx,
              bestPrevIdxByParent[parentIdx]);
        }

        // Preserve the existing layout-group skip heuristic. A non-OOM
        // failure for one sharded grid can still hide a later valid grid in
        // the same group, while default-layout groups must never be skipped.
        const bool hasCandidateOutputLayout =
            static_cast<bool>(candidate.opConfig.outputLayout);
        const bool isDefaultGroup =
            optimizer_utils::getLayout(candidate.opConfig.outputLayout) ==
            TensorMemoryLayout::Interleaved;
        const bool shouldSkipGroup =
            localResult.skipGroup && candidate.groupIndex.has_value() &&
            (!hasCandidateOutputLayout || !isDefaultGroup);
        if (!shouldSkipGroup) {
          ++candidateIndex;
          continue;
        }

        const std::size_t groupIndex = *candidate.groupIndex;
        ++candidateIndex;
        while (candidateIndex < candidates.size() &&
               candidates[candidateIndex].groupIndex &&
               *candidates[candidateIndex].groupIndex == groupIndex) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "DP op={} candidate={} -> inf: skipped by invalid "
                       "layout group {}",
                       op->getName().getStringRef(), candidateIndex,
                       groupIndex);
          viterbiTable[op][candidateIndex] =
              std::numeric_limits<double>::infinity();
          backtrackTable[op][candidateIndex].assign(opParents.size(), -1);
          ++candidateIndex;
        }
      }

      // Sort cached output sizes only for deterministic debug logging.
      auto storeIt = candidateOutputSizes.find(op);
      if (storeIt == candidateOutputSizes.end()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "computeOutputSize store check: op={} no stored entry",
                     op->getName().getStringRef());
      } else {
        const auto &storedSizes = storeIt->second;
        struct RankedCacheEntry {
          std::size_t candidateIdx;
          uint64_t bytes;
        };
        llvm::SmallVector<RankedCacheEntry> rankedEntries;
        rankedEntries.reserve(storedSizes.size());
        for (std::size_t storeIdx = 0; storeIdx < storedSizes.size();
             ++storeIdx) {
          // Only rank candidates that have a stored output size.
          if (!storedSizes[storeIdx]) {
            continue;
          }
          rankedEntries.push_back({storeIdx, *storedSizes[storeIdx]});
        }

        // Candidate counts are small, so insertion sort keeps this debug path
        // simple.
        for (std::size_t rankedIdx = 1; rankedIdx < rankedEntries.size();
             ++rankedIdx) {
          RankedCacheEntry currentEntry = rankedEntries[rankedIdx];
          std::size_t insertIdx = rankedIdx;
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

      double opMinCost = inf;

      for (double cost : viterbiTable[op]) {
        if (std::isfinite(cost)) {
          opMinCost = std::min(opMinCost, cost);
        }
      }

      if (std::isfinite(opMinCost)) {
        for (double &cost : viterbiTable[op]) {
          if (std::isfinite(cost)) {
            cost -= opMinCost;
          }
        }

        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "DP normalized op={} opMinCost={}",
                     op->getName().getStringRef(), opMinCost);
      }

      // Stop early when the current operation has no reachable candidate.
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

SolverStatus ViterbiPolicy::performBacktracking() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "Performing backtracking to find optimal layout configurations");

  optimalCandidateIndex.clear();
  SolverStatus status = SolverStatus::Success;

  llvm::SmallVector<mlir::Operation *> worklist;
  llvm::DenseMap<mlir::Operation *, bool> traversedOps;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<ParentCandRequest>>
      expectedCandsByParent;

  // Remove requests issued by currentOp so that outdated parent candidate
  // requests are not kept when currentOp is traversed again with a new selected
  // candidate.
  auto removeRequestFromCurrentOp = [&](mlir::Operation *currentOp) {
    llvm::SmallVector<mlir::Operation *> emptyParentOps;
    // Remove stale requests and erase parents with no remaining requests.
    for (auto &entry : expectedCandsByParent) {
      llvm::SmallVector<ParentCandRequest> &requests = entry.second;
      for (auto requestIt = requests.begin(); requestIt != requests.end();) {
        if (requestIt->currentOp == currentOp) {
          requestIt = requests.erase(requestIt);
        } else {
          ++requestIt;
        }
      }

      // Remove the parent entry when no consumer requests remain.
      if (requests.empty()) {
        emptyParentOps.push_back(entry.first);
      }
    }

    // Erase parent entries whose request lists are empty.
    for (mlir::Operation *emptyParentOp : emptyParentOps) {
      expectedCandsByParent.erase(emptyParentOp);
    }
  };

  // Record each consumer request so shared-parent conflicts can be resolved.
  auto recordExpectedParentCand = [&](mlir::Operation *parentOp,
                                      const ParentCandRequest &request) {
    expectedCandsByParent[parentOp].push_back(request);
  };

  // Initialize backtracking from finite sink candidates.
  llvm::SmallVector<std::pair<mlir::Operation *, std::size_t>> seeds =
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
      // Skip operations already traversed for the current selection.
      if (traversedOps.find(op) != traversedOps.end()) {
        continue;
      }
      // Mark the selected operation as traversed.
      traversedOps[op] = true;

      // Read the selected candidate for this operation.
      const std::size_t selectedCandidateIdx = optimalCandidateIndex.lookup(op);

      // Read the backtracking state for this operation.
      auto backtrackIt = backtrackTable.find(op);
      // Missing backtracking state makes the selected path invalid.
      if (backtrackIt == backtrackTable.end()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Backtracking: op {} candidate {} has missing "
                     "backtrack entry; stopping",
                     op->getName().getStringRef(), selectedCandidateIdx);
        status = SolverStatus::InvalidBacktrackState;
        break;
      }

      if (selectedCandidateIdx >= backtrackIt->second.size()) {
        // Reject an out-of-bounds selected candidate.
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
      // Linear operations have one parent, while joins can have several.
      const llvm::SmallVector<int> &bestPrevIdxByParent =
          backtrackIt->second[selectedCandidateIdx];
      const llvm::SmallVector<mlir::Operation *> parents = getOpParents(op);

      if (bestPrevIdxByParent.size() < parents.size()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Backtracking: op {} candidate {} has {} backtrack "
                     "parents, expected {}; stopping",
                     op->getName().getStringRef(), selectedCandidateIdx,
                     bestPrevIdxByParent.size(), parents.size());
        status = SolverStatus::InvalidBacktrackState;
        break;
      }

      const std::size_t parentCount =
          std::min(parents.size(), bestPrevIdxByParent.size());
      for (std::size_t parentIdx = 0; parentIdx < parentCount; ++parentIdx) {
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

        const std::size_t parentCandIdxSize =
            static_cast<std::size_t>(parentCandidateIdx);
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

        // Invalid parent candidates cannot resolve a shared-parent conflict.
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

        // Check whether this parent already has a selected candidate.
        if (auto assignedIt = optimalCandidateIndex.find(parentOp);
            assignedIt != optimalCandidateIndex.end()) {
          // Record the new consumer request for this parent.
          recordExpectedParentCand(parentOp, parentRequest);
          llvm::SmallVector<ParentCandRequest> &requests =
              expectedCandsByParent[parentOp];
          if (assignedIt->second != parentCandIdxSize) {
            // Resolve the conflict when the assigned and requested candidates
            // differ.
            std::optional<std::size_t> resolvedParentCandIdx =
                resolveParentCandConflict(parentOp, requests);
            // Assign the resolved candidate to the parent when one exists.
            if (resolvedParentCandIdx) {
              const std::size_t previousParentCandIdx = assignedIt->second;
              assignedIt->second = *resolvedParentCandIdx;
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
    std::optional<std::size_t> nextParentScheduleIdx;
    // Search all unresolved parent operations.
    for (const auto &entry : expectedCandsByParent) {
      mlir::Operation *const parentOp = entry.first;
      // Skip already assigned parents.
      if (optimalCandidateIndex.find(parentOp) != optimalCandidateIndex.end()) {
        continue;
      }

      // Find the scheduled index of this parent op in the schedule.
      std::optional<std::size_t> parentScheduleIdx =
          optimizer_utils::getScheduledOpIndex(schedule, parentOp);

      // Every unresolved parent must have a schedule position.
      if (!parentScheduleIdx) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Backtracking: unresolved parent op {} has no schedule "
                     "entry; stopping",
                     parentOp->getName().getStringRef());
        status = SolverStatus::InvalidBacktrackState;
        break;
      }

      // Seed the traversal choice with the first unresolved parent.
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

    // Backtracking is complete when no unresolved parent remains.
    if (!nextParent) {
      break;
    }

    // Resolve all consumer requests for the selected parent together.
    const auto &requests = expectedCandsByParent[nextParent];

    std::size_t selectedParentCandidateIdx =
        requests.front().expectedParentCandIdx;
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
      std::optional<std::size_t> resolvedParentCandIdx =
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

SolverStatus ViterbiPolicy::validateFinalAssignment() const {
  for (const auto &[_, operations] : schedule) {
    for (std::size_t opIdx = 0; opIdx < operations.size(); ++opIdx) {
      mlir::Operation *op = operations[opIdx];
      auto candidateIt = candidateResult.candidateMap.find(op);
      if (candidateIt == candidateResult.candidateMap.end() ||
          candidateIt->second.empty()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} has missing candidate "
                     "state",
                     op->getName().getStringRef());
        return SolverStatus::MissingCandidateState;
      }

      auto selectedIt = optimalCandidateIndex.find(op);
      if (selectedIt == optimalCandidateIndex.end()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} has no selected "
                     "candidate",
                     op->getName().getStringRef());
        return SolverStatus::PartialAssignment;
      }

      const std::size_t selectedCandidateIdx = selectedIt->second;
      if (selectedCandidateIdx >= candidateIt->second.size()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} selected candidate {} "
                     "is Out-Of-Bounds (candidate count {})",
                     op->getName().getStringRef(), selectedCandidateIdx,
                     candidateIt->second.size());
        return SolverStatus::InvalidCandidateSelection;
      }

      auto costIt = viterbiTable.find(op);
      if (costIt == viterbiTable.end() ||
          selectedCandidateIdx >= costIt->second.size()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} selected candidate {} "
                     "has no DP cost entry",
                     op->getName().getStringRef(), selectedCandidateIdx);
        return SolverStatus::InvalidCandidateSelection;
      }

      if (!std::isfinite(costIt->second[selectedCandidateIdx])) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} selected candidate {} "
                     "has non-finite DP cost {}",
                     op->getName().getStringRef(), selectedCandidateIdx,
                     costIt->second[selectedCandidateIdx]);
        return SolverStatus::NoValidGlobalPath;
      }

      auto backtrackIt = backtrackTable.find(op);
      if (backtrackIt == backtrackTable.end() ||
          selectedCandidateIdx >= backtrackIt->second.size()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ValidateFinalAssignment: op {} selected candidate {} "
                     "has invalid backtrack state",
                     op->getName().getStringRef(), selectedCandidateIdx);
        return SolverStatus::InvalidBacktrackState;
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
        return SolverStatus::InvalidBacktrackState;
      }

      for (std::size_t parentIdx = 0; parentIdx < parents.size(); ++parentIdx) {
        const int expectedParentCandIdx = bestPrevIdxByParent[parentIdx];
        if (expectedParentCandIdx < 0) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "ValidateFinalAssignment: op {} candidate {} "
                       "parentIdx {} has invalid parent candidate {}",
                       op->getName().getStringRef(), selectedCandidateIdx,
                       parentIdx, expectedParentCandIdx);
          return SolverStatus::InvalidBacktrackState;
        }

        mlir::Operation *parentOp = parents[parentIdx];
        auto parentSelectedIt = optimalCandidateIndex.find(parentOp);
        if (parentSelectedIt == optimalCandidateIndex.end()) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "ValidateFinalAssignment: parent op {} requested by "
                       "op {} candidate {} is unassigned",
                       parentOp->getName().getStringRef(),
                       op->getName().getStringRef(), selectedCandidateIdx);
          return SolverStatus::PartialAssignment;
        }

        const std::size_t selectedParentCandIdx = parentSelectedIt->second;

        if (selectedParentCandIdx !=
            static_cast<std::size_t>(expectedParentCandIdx)) {
          auto parentCandidateIt = candidateResult.candidateMap.find(parentOp);

          const bool hasParentCandidate =
              parentCandidateIt != candidateResult.candidateMap.end() &&
              selectedParentCandIdx < parentCandidateIt->second.size();

          if (!hasParentCandidate) {
            TTMLIR_DEBUG(
                ttmlir::LogComponent::ViterbiOptimizer,
                "ValidateFinalAssignment: parent op {} selected candidate {} "
                "is "
                "invalid while validating op {} candidate {} expected parent "
                "candidate {}",
                parentOp->getName().getStringRef(), selectedParentCandIdx,
                op->getName().getStringRef(), selectedCandidateIdx,
                expectedParentCandIdx);
            return SolverStatus::InvalidCandidateSelection;
          }

          const double resolvedTransitionCost = getBestTransitionCost(
              parentOp, parentCandidateIt->second[selectedParentCandIdx], op,
              candidateIt->second[selectedCandidateIdx],
              getCandidateAdditionalL1Usage(op, selectedCandidateIdx)
                  .value_or(0));

          if (!std::isfinite(resolvedTransitionCost)) {
            TTMLIR_DEBUG(
                ttmlir::LogComponent::ViterbiOptimizer,
                "ValidateFinalAssignment: parent op {} selected candidate {} "
                "does not match op {} candidate {} expected parent candidate "
                "{}, "
                "and resolved transition is invalid",
                parentOp->getName().getStringRef(), selectedParentCandIdx,
                op->getName().getStringRef(), selectedCandidateIdx,
                expectedParentCandIdx);
            return SolverStatus::InvalidTransitionState;
          }

          TTMLIR_DEBUG(
              ttmlir::LogComponent::ViterbiOptimizer,
              "ValidateFinalAssignment: accepted conflict-resolved parent op "
              "{} "
              "candidate {} for op {} candidate {} originally expected parent "
              "candidate {}",
              parentOp->getName().getStringRef(), selectedParentCandIdx,
              op->getName().getStringRef(), selectedCandidateIdx,
              expectedParentCandIdx);
        }
      }
    }
  }

  return SolverStatus::Success;
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

    auto spillCountsIt = selectedSpillCounts.find(op);
    if (spillCountsIt != selectedSpillCounts.end() &&
        bestIdx < spillCountsIt->second.size()) {
      const std::size_t selectedSpillCount = spillCountsIt->second[bestIdx];

      if (selectedSpillCount > 0) {
        const LiveTensorList passiveActivations = getPassiveActivations(op);

        llvm::SmallVector<mlir::Value> spillCandidates;

        for (const LiveTensorInfo &activation : passiveActivations) {
          mlir::Operation *producerOp = activation.value.getDefiningOp();
          if (!producerOp) {
            continue;
          }

          auto cachedIt = candidateOutputSizes.find(producerOp);
          if (cachedIt == candidateOutputSizes.end()) {
            continue;
          }

          const bool hasNonZeroL1Option = llvm::any_of(
              cachedIt->second, [](const std::optional<uint64_t> &cachedBytes) {
                return cachedBytes && *cachedBytes != 0;
              });

          if (!hasNonZeroL1Option) {
            continue;
          }

          if (!llvm::is_contained(spillCandidates, activation.value)) {
            spillCandidates.push_back(activation.value);
          }
        }

        const std::size_t spillLimit =
            std::min(selectedSpillCount, spillCandidates.size());

        for (std::size_t spillIdx = 0; spillIdx < spillLimit; ++spillIdx) {
          result.spillRequests.push_back(
              SpillRequest{spillCandidates[spillIdx], op});
        }
      }
    }

    const auto bufferType = bestCandidate.opConfig.outputLayout.getBufferType();
    if (bufferType == BufferType::L1) {
      ++result.numL1Configs;
    } else if (bufferType == BufferType::DRAM) {
      ++result.numDRAMConfigs;
    }
  }

  // The DP table is row-normalized, so sink cost alone no longer
  // represents the selected path. Sum selected normalized costs for solver
  // reporting, while preserving the seed fallback for tests or
  // partial states where no selected assignment exists yet.
  bool hasSelectedAssignment = false;
  bool hasFiniteSelectedCost = false;
  for (const auto &[_, operations] : schedule) {
    for (mlir::Operation *op : operations) {
      auto selectedIt = optimalCandidateIndex.find(op);
      if (selectedIt == optimalCandidateIndex.end()) {
        continue;
      }
      hasSelectedAssignment = true;

      auto costIt = viterbiTable.find(op);
      if (costIt == viterbiTable.end()) {
        continue;
      }

      const std::size_t selectedIdx = selectedIt->second;
      const auto &costs = costIt->second;
      if (selectedIdx >= costs.size()) {
        continue;
      }

      const double selectedCost = costs[selectedIdx];
      if (std::isfinite(selectedCost)) {
        result.totalCost += selectedCost;
        hasFiniteSelectedCost = true;
      }
    }
  }

  if (!hasFiniteSelectedCost && !hasSelectedAssignment) {
    for (const auto &[sinkOp, seedIdx] : getBacktrackingSeeds()) {
      const auto &sinkCosts = viterbiTable.lookup(sinkOp);
      if (seedIdx < sinkCosts.size() && std::isfinite(sinkCosts[seedIdx])) {
        result.totalCost += sinkCosts[seedIdx];
      }
    }
  }

  return result;
}

void ViterbiPolicy::printOptimalPath() const {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "=== Optimal Path Summary ===\n");

  [[maybe_unused]] std::size_t totalOps = 0;
  [[maybe_unused]] double totalCost = 0.0;

  for (const auto &[func, operations] : schedule) {
    for ([[maybe_unused]] std::size_t opIdx = 0; opIdx < operations.size();
         ++opIdx) {
      mlir::Operation *op = operations[opIdx];
      auto costIt = viterbiTable.find(op);
      if (costIt == viterbiTable.end()) {
        continue;
      }

      const auto &costs = costIt->second;
      if (!costs.empty()) {
        std::size_t bestIdx = 0;
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
          bestIdx = static_cast<std::size_t>(minIt - costs.begin());
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

        // Count finite candidates for debug reporting.
        [[maybe_unused]] std::size_t numValidCandidates =
            static_cast<std::size_t>(
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

  // Report the first failed solver stage on one scheduled operation.
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
    // Construct and report the selected path after all solver stages succeed.
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

ViterbiResult ViterbiPolicy::run() {
  // The current scheduler result is expected to contain one function.
  [[maybe_unused]] std::size_t totalScheduledOps = 0;
  if (!schedule.empty()) {
    totalScheduledOps = schedule.begin()->second.size();
  }
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "Running Viterbi Policy with {} operations", totalScheduledOps);
  calculateTensorLifetimes(schedule);
  [[maybe_unused]] SortedLifetimesMap sorted = sortLifetimes(tensorLifetimes);
  return solve();
}

} // namespace mlir::tt::ttnn
