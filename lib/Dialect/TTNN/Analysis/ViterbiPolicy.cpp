// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ViterbiPolicy.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Attributes.h"
#include "ttmlir/Dialect/TTNN/Analysis/CostModel.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <limits>

namespace mlir::tt::ttnn {

// Constructor for BOS Optimizer Pass
// Inputs:
//  - candidateResult: Precomputed op candidates from OpCandidatesBuilder
//  - schedule: Operation schedule (order of operations) to consider for
//  optimization: DenseMap<FuncOp, SmallVector<Operation*>>

ViterbiPolicy::ViterbiPolicy(const OpCandidateBuilderResult &candidateResult,
                             const OperationSchedule &schedule)
    : candidateResult(candidateResult), schedule(schedule),
      costModel(std::make_shared<CostModel>()) {}

// =================================================================
// MAIN BASELINE FOR VITERBI POLICY
// =================================================================

ViterbiResult ViterbiPolicy::run() {
  // The schedulerResult just have 1 func
  [[maybe_unused]] size_t totalScheduledOps = 0;
  totalScheduledOps = schedule.begin()->second.size();
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "Running Viterbi Policy with {} operations", totalScheduledOps);

  printScheduleOrder(schedule);
  // printCandidateInfo(schedule);

  calculateTensorLifetimes(schedule);
  [[maybe_unused]] SortedLifetimesMap sorted = sortLifetimes(tensorLifetimes);

  // // Get whole tensor lifetime map
  // const TensorLifetimeMap &lifetimes = getTensorLifetimes();

  // for (const auto &[value, lifetime] : lifetimes) {
  //   TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
  //                "Current Value: {} start={} end={} span={}",
  //                analysis::bos::formatValueShort(value),
  //                lifetime.startOpIdx, lifetime.endOpIdx, lifetime.span);
  // }

  // // Print detailed liveness information
  // for (auto &[func, ops] : schedule) {
  //     mlir::Liveness liveness(func);
  //     mlir::Block *block = &func.getBody().front();
  //     const mlir::LivenessBlockInfo *livenessInfo =
  //     liveness.getLiveness(block); printLivenessInfo(livenessInfo);
  // }

  // for (const auto &[value, lt] : sorted) {
  //   TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
  //               "value={0} start={1} end={2} span={3}",
  //               analysis::bos::formatValueShort(value),
  //               lt.startOpIdx, lt.endOpIdx, lt.span);
  // }

  // for (const auto &[func, ops] : schedule) {
  //   for (size_t opIdx = 0; opIdx < ops.size(); ++opIdx) {
  //     mlir::Operation *op = ops[opIdx];

  //     auto activations = getActivationLifetime(op);

  //     TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
  //                 "=== ActivationLifetime: Op[{0}] {1} ===",
  //                 opIdx, op->getName().getStringRef());

  //     if (activations.empty()) {
  //       TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
  //                   "  no activation found");
  //       continue;
  //     }

  //     for (const auto &[value, lt] : activations) {
  //       TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
  //                   "  activation {0} : type={1} start={2} end={3} span={4}",
  //                   analysis::bos::formatValueShort(value), value.getType(),
  //                   lt.startOpIdx, lt.endOpIdx, lt.span);

  //     }
  //   }
  // }

  // Debug print lifetimes for testing purpose
  [[maybe_unused]] auto schedIt = schedule.begin();
  if (schedIt != schedule.end()) {
    printTensorLifetimes(schedIt->second, tensorLifetimes);
  }

  return solve();
}

ViterbiResult ViterbiPolicy::solve() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "Executing Viterbi algorithm\n");

  // Initialize result
  ViterbiResult result;
  result.totalCost = 0.0;
  result.numL1Configs = 0;
  result.numDRAMConfigs = 0;

  // Initialize DP and backtracking tables
  buildViterbiTable();

  // Calculate cost and update Viterbi table
  performCostCalculation();

  // Backtracking to find optimal path
  performBacktracking();

  // Construct result based on optimal candidate indices
  for (const auto &[op, bestIdx] : optimalCandidateIndex) {
    auto candidateIt = candidateResult.candidateMap.find(op);
    if (candidateIt == candidateResult.candidateMap.end() ||
        bestIdx >= candidateIt->second.size()) {
      continue;
    }

    const auto &bestCandidate = candidateIt->second[bestIdx];
    result.optimalConfigurations[op] = bestCandidate.opConfig;
    result.inputLayouts[op] = bestCandidate.inputLayouts;

    // Accumulate total cost from Viterbi table
    auto costIt = viterbiTable.find(op);
    if (costIt != viterbiTable.end() && bestIdx < costIt->second.size()) {
      result.totalCost += costIt->second[bestIdx];
    }

    // Count how many ops got L1 vs DRAM placement based on best candidate's
    // output layout
    const auto bufferType = bestCandidate.opConfig.outputLayout.getBufferType();
    if (bufferType == BufferType::L1) {
      ++result.numL1Configs;
    } else if (bufferType == BufferType::DRAM) {
      ++result.numDRAMConfigs;
    }
  }

  printOptimalPath();

  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "Viterbi solve completed. Total cost: {0}, L1: {1}, DRAM: {2}\n",
               result.totalCost, result.numL1Configs, result.numDRAMConfigs);

  return result;
}

// =================================================================
// CORE ALGORITHM FUNCTIONS
// =================================================================

// Create Viterbi DP table (N x T) based on candidate configurations and costs
void ViterbiPolicy::buildViterbiTable() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy, "Building Viterbi tables");

  double inf = std::numeric_limits<double>::infinity();
  for (const auto &[func, operations] : schedule) {
    for (auto *op : operations) {
      // Find candidates for this operation
      auto it = candidateResult.candidateMap.find(op);
      if (it != candidateResult.candidateMap.end()) {
        const auto &candidates = it->second;

        size_t numCandidates = candidates.size();
        viterbiTable[op].resize(numCandidates);
        backtrackTable[op].resize(numCandidates);

        for (size_t i = 0; i < numCandidates; ++i) {
          viterbiTable[op][i] = inf;
          backtrackTable[op][i] = static_cast<int>(-1);
        }
      }
    }
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "Finished building Viterbi tables");
}

void ViterbiPolicy::performCostCalculation() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "Performing cost calculation for Viterbi DP table");

  // Loop through each operation in scheduled topological order
  for (const auto &[func, ops] : schedule) {
    for (size_t opIdx = 0; opIdx < ops.size(); ++opIdx) {
      mlir::Operation *op = ops[opIdx];

      auto it = candidateResult.candidateMap.find(op);
      if (it == candidateResult.candidateMap.end()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
                     "Op {} has no candidates, skip",
                     op->getName().getStringRef());
        continue;
      }

      const auto &candidates = it->second;
      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
                   "Op {} has {} candidates", op->getName().getStringRef(),
                   candidates.size());

      bool isJoin = isJoinOperation(op);
      // Debug join condition
      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy, "Op {} isJoin={}",
                   op->getName().getStringRef(), isJoin);

      // Handle linear & join separately
      // Since those opeartion are scheduled, each operation has at least one
      // activation Therefore, we don't need to check empty activation case
      if (!isJoin) { // If linear case
        // auto activations = getActivationLifetime(op);
        for (size_t i = 0; i < candidates.size(); ++i) {
          const auto &candidate = candidates[i];
          double cost = costModel->getLinearCost(op, candidate);
          viterbiTable[op][i] = cost;
        }

      } else { // If join case
        auto activations = getActivationLifetime(op);

        llvm::SmallVector<mlir::Operation *> parentOps;
        parentOps.reserve(activations.size());

        for (const auto &[value, lt] : activations) {
          if (mlir::Operation *parentOp = value.getDefiningOp()) {
            parentOps.push_back(parentOp);
          }
        }

        for (size_t i = 0; i < candidates.size(); ++i) {
          const auto &candidate = candidates[i];

          double cost = costModel->getJoinCost(parentOps, op, candidate);
          viterbiTable[op][i] = cost;
        }
      }

      // total = prev_score + t + e (transition + emission)
      // if total < best_score:
      //     best_score = total
      //     best_prev = prev_cand
    }
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "Finished cost calculation");
}

// Backtracking to find optimal layout for each operation
void ViterbiPolicy::performBacktracking() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "Performing backtracking to find optimal layout configurations");

  optimalCandidateIndex.clear();

  // For baseline: select minimum cost candidate for each operation
  for (const auto &[op, costs] : viterbiTable) {
    if (!costs.empty()) {
      // Find minimum cost candidate
      auto minIt = costs.begin();
      for (auto it = costs.begin() + 1; it != costs.end(); ++it) {
        if (*it < *minIt) {
          minIt = it;
        }
      }

      // Assign best candidate index for this operation
      size_t bestIdx = minIt - costs.begin();
      optimalCandidateIndex[op] = bestIdx;
    }
  }
}

void ViterbiPolicy::reset() {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "Resetting ViterbiPolicy state\n");
  viterbiTable.clear();
  backtrackTable.clear();
  optimalCandidateIndex.clear();
}

// =================================================================
// HELPER FUNCTIONS
// =================================================================

void ViterbiPolicy::calculateTensorLifetimes(
    const OperationSchedule &schedule) {
  // Clear the map before calculating lifetimes
  tensorLifetimes.clear();

  for (const auto &entry : schedule) {
    // schedule is llvm::DenseMap<mlir::func::FuncOp,
    // llvm::SmallVector<mlir::Operation *>>
    mlir::func::FuncOp func = entry.first;
    const auto &ops = entry.second;

    for (mlir::BlockArgument arg : func.getArguments()) {
      // Initialize lifetime for function arguments
      tensorLifetimes.try_emplace(
          arg, TensorLifetime{/*startOpIdx=*/0, /*endOpIdx=*/0, /*span=*/1});
    }

    for (size_t i = 0; i < ops.size(); ++i) {
      // Current scheduled operation at index i
      mlir::Operation *op = ops[i];

      for (mlir::Value result : op->getResults()) {
        // Initialize lifetime for each result value
        auto [it, inserted] =
            tensorLifetimes.try_emplace(result, TensorLifetime{i, i, 1});
        // If value already exists, update the endOpIdx to current index i
        if (!inserted) {
          it->second.startOpIdx = std::min(it->second.startOpIdx, i);
          it->second.endOpIdx = std::max(it->second.endOpIdx, i);
        }
      }

      for (mlir::Value operand : op->getOperands()) {
        // Check whether this operand already has a recorded lifetime
        auto it = tensorLifetimes.find(operand);
        // If this is the first time we see the operand, initialize its lifetime
        // - Block arguments are considered live from the start of the schedule
        // - Other values is set to i
        if (it == tensorLifetimes.end()) {
          size_t start = llvm::isa<mlir::BlockArgument>(operand) ? 0 : i;
          it = tensorLifetimes.try_emplace(operand, TensorLifetime{start, i, 1})
                   .first;
        } else {
          // Operand is used by the current op, so extend its live range
          // to at least the current op index.
          it->second.endOpIdx = std::max(it->second.endOpIdx, i);
        }
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
    sorted.push_back(kv);
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

ActivationLifetimeList
ViterbiPolicy::getActivationLifetime(mlir::Operation *currentOp) const {
  ActivationLifetimeList activations;
  if (!currentOp) {
    return activations;
  }

  size_t currentOpIdx = 0;
  bool found = false;
  // Find the index of the current operation in the schedule
  for (const auto &[func, ops] : schedule) {
    for (size_t i = 0; i < ops.size(); ++i) {
      if (ops[i] == currentOp) {
        currentOpIdx = i;
        found = true;
        break;
      }
    }
    if (found) {
      break;
    }
  }

  // If the operation is not in the schedule, there is no lifetime info to use
  if (!found) {
    return activations;
  }

  // For first operation, treat operand[0] as the activation
  if (currentOpIdx == 0) {
    if (currentOp->getNumOperands() > 0) {
      mlir::Value value = currentOp->getOperand(0);

      if (mlir::isa<mlir::TensorType>(value.getType())) {
        auto it = tensorLifetimes.find(value);
        if (it != tensorLifetimes.end()) {
          activations.push_back({value, it->second});
        }
      }
    }
    return activations;
  }

  // For later ops, collect only intermediate tensor operands when:
  // - must be a tensor
  // - must not be a block argument (function input)
  // - must be produced by some defining op
  // - must have a recorded lifetime
  for (mlir::Value operand : currentOp->getOperands()) {
    if (!mlir::isa<mlir::TensorType>(operand.getType())) {
      continue;
    }

    if (llvm::isa<mlir::BlockArgument>(operand)) {
      continue;
    }

    if (!operand.getDefiningOp()) {
      continue;
    }

    auto it = tensorLifetimes.find(operand);
    if (it == tensorLifetimes.end()) {
      continue;
    }

    activations.push_back({operand, it->second});
  }

  return activations;
}

// Check if the current operation is a join point (i.e., has multiple
// activations)
bool ViterbiPolicy::isJoinOperation(mlir::Operation *op) const {
  if (!op) {
    return false;
  }

  auto activations = getActivationLifetime(op);
  return activations.size() >= 2;
}

// =================================================================
// LOGGING AND DEBUGGING
// =================================================================

void ViterbiPolicy::printScheduleOrder(const OperationSchedule &schedule) {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "=== Operation Schedule Order ===\n");

  for (const auto &[func, operations] : schedule) {
    for (size_t idx = 0; idx < operations.size(); ++idx) {
      [[maybe_unused]] mlir::Operation *op = operations[idx];
      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy, "Op[{0}]: {1}", idx,
                   op->getName().getStringRef());
    }
  }
}

void ViterbiPolicy::printCandidateInfo(const OperationSchedule &schedule) {
  // This check details the candidates for each operation before running the
  // Viterbi algorithm
  for (const auto &[func, operations] : schedule) {
    for (mlir::Operation *op : operations) {
      auto candIt = candidateResult.candidateMap.find(op);
      if (candIt == candidateResult.candidateMap.end()) {
        continue;
      }

      const auto &candidates = candIt->second;
      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
                   "Layout candidates for op {0}: {1}",
                   op->getName().getStringRef(), candidates.size());

      for (size_t i = 0; i < candidates.size(); ++i) {
        std::string layoutStr;
        llvm::raw_string_ostream os(layoutStr);
        mlir::Attribute(candidates[i].opConfig.outputLayout).print(os);
        os.flush();

        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
                     "  op={0} cand={1} output_layout={2}",
                     op->getName().getStringRef(), i, layoutStr);
      }
    }
  }
}

void ViterbiPolicy::printOptimalPath() const {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "=== Optimal Path Summary ===\n");

  [[maybe_unused]] size_t totalOps = 0;
  [[maybe_unused]] double totalCost = 0.0;

  for (const auto &[func, operations] : schedule) {
    for (mlir::Operation *op : operations) {
      auto costIt = viterbiTable.find(op);
      if (costIt == viterbiTable.end()) {
        continue;
      }

      const auto &costs = costIt->second;
      if (costs.empty()) {
        continue;
      }

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

      auto inputLayouts = utils::extractInputLayouts(op);
      auto outputLayout = analysis::bos::extractOutputLayoutFromIR(op);

      std::string inputLayoutsStr =
          analysis::bos::layoutsToString(inputLayouts);
      std::string outputLayoutStr = analysis::bos::layoutToString(outputLayout);

      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
                   "Operation: {0} -> Candidate {1} (Cost: {2})\n"
                   "Input layouts: {3}\n"
                   "Output layout: {4}\n",
                   op->getName().getStringRef(), bestIdx, bestCost,
                   inputLayoutsStr, outputLayoutStr);
    }
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "Total operations: {0}, Total cost: {1}\n", totalOps, totalCost);
}

void ViterbiPolicy::printLivenessInfo(
    const mlir::LivenessBlockInfo *livenessInfo) {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
               "=== Detailed Liveness Analysis ===");

  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy, "Live-in values: {}",
               livenessInfo->in().size());
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy, "Live-out values: {}",
               livenessInfo->out().size());

  for (auto &[func, operations] : schedule) {
    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy, "--- Function: {} ---",
                 func.getSymName());

    for (size_t opIdx = 0; opIdx < operations.size(); ++opIdx) {
      mlir::Operation *op = operations[opIdx];
      auto currentlyLive = livenessInfo->currentlyLiveValues(op);

      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
                   "Op[{}] '{}': {} currently live values", opIdx,
                   op->getName().getStringRef(), currentlyLive.size());

      for (mlir::Value liveValue : currentlyLive) {
        std::string valueStr = analysis::bos::formatValueShort(liveValue);

        std::string typeStr;
        llvm::raw_string_ostream typeOS(typeStr);
        liveValue.getType().print(typeOS);

        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy, "  - {} : {}",
                     valueStr, typeStr); // operation name : tensor value
      }
    }
  }
}

void ViterbiPolicy::printTensorLifetimes(
    const llvm::SmallVector<mlir::Operation *> &ops,
    const TensorLifetimeMap &lifetimes) const {

  for (size_t opIdx = 0; opIdx < ops.size(); ++opIdx) {
    mlir::Operation *op = ops[opIdx];

    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
                 "=== TensorLifetime sanity: Op[{0}] === {1}", opIdx,
                 op->getName().getStringRef());

    for (unsigned idx = 0; idx < op->getNumOperands(); ++idx) {
      mlir::Value v = op->getOperand(idx);
      auto it = lifetimes.find(v);

      if (it == lifetimes.end()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
                     "  operand[{0}] {1} : MISSING", idx,
                     analysis::bos::formatValueShort(v));
      } else {
        [[maybe_unused]] const auto &lt = it->second;
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
                     "  operand[{0}] {1} : start={2} end={3} span={4}", idx,
                     analysis::bos::formatValueShort(v), lt.startOpIdx,
                     lt.endOpIdx, lt.span);
      }
    }

    for (unsigned idx = 0; idx < op->getNumResults(); ++idx) {
      mlir::Value v = op->getResult(idx);
      auto it = lifetimes.find(v);

      if (it == lifetimes.end()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
                     "  result[{0}] {1} : MISSING", idx,
                     analysis::bos::formatValueShort(v));
      } else {
        [[maybe_unused]] const auto &lt = it->second;
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiPolicy,
                     "  result[{0}] {1} : start={2} end={3} span={4}", idx,
                     analysis::bos::formatValueShort(v), lt.startOpIdx,
                     lt.endOpIdx, lt.span);
      }
    }
  }
}

} // namespace mlir::tt::ttnn