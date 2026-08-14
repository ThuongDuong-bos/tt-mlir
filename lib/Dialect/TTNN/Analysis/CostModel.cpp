// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/CostModel.h"

#include "ttmlir/Dialect/TTNN/Analysis/TransitionEdgeAnalysis.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlir::tt::ttnn {

namespace {

constexpr double inf = std::numeric_limits<double>::infinity();

// Get the effective L1 capacity in bytes for the given op.
double getEffectiveL1CapacityBytes(mlir::Operation *op) {
  if (!op) {
    return 0.0;
  }

  return static_cast<double>(utils::getUsableL1PerCore(op));
}

double computeL1Utility(
    mlir::Operation *op,
    const op_constraint_validation::ValidationResult &validationResult,
    uint64_t additionalL1Usage) {
  const double capacity = getEffectiveL1CapacityBytes(op);

  if (capacity <= 0.0) {
    return 0.0;
  }

  return static_cast<double>(validationResult.overallPeakL1Usage +
                             additionalL1Usage) /
         capacity;
}

// Declare max heap entry to process combinations from largest totalBytes to
// smallest.
struct HeapEntry {
  uint64_t totalBytes = 0;
  llvm::SmallVector<std::size_t> indices;
};

// Build a unique ID of combination
// E.g. indices = [0, 2, 1] -> key = "0#2#1#"
std::string buildCombinationKey(llvm::ArrayRef<std::size_t> indices) {
  std::string key;
  key.reserve(indices.size() * 4);

  for (std::size_t index : indices) {
    key.append(std::to_string(index));
    key.push_back('#');
  }

  return key;
}

// Compute the total L1 bytes freed by spilling selected passive producers.
// passiveTensorList contains only non-zero L1 size options; DRAM cached sizes
// are filtered out before this point.
uint64_t
computeSelectedSpillBytes(const CostModel::PassiveTensorList &passiveTensorList,
                          std::size_t selectedSpillCount) {
  uint64_t selectedSpillBytes = 0;

  for (std::size_t spillIdx = 0;
       spillIdx < selectedSpillCount && spillIdx < passiveTensorList.size();
       ++spillIdx) {
    if (passiveTensorList[spillIdx].empty()) {
      continue;
    }

    selectedSpillBytes += passiveTensorList[spillIdx][0];
  }

  return selectedSpillBytes;
}

// Update validation result on failure and return whether the failure is.
bool updateValidationFailure(
    const op_constraint_validation::ValidationResult &validationResult,
    op_constraint_validation::ValidationResult &result,
    op_constraint_validation::ValidationResult &lastOOMResult) {
  const bool isOOM =
      validationResult.status ==
      op_constraint_validation::ValidationStatus::OutOfMemoryError;

  if (!isOOM) {
    result = validationResult;
    return true;
  }

  lastOOMResult = validationResult;
  return false;
}

//===----------------------------------------------------------------------===//
// Transition cost helpers
//===----------------------------------------------------------------------===//

// Construct and build the transition cost map.
struct TransitionCostMap {
  double tMem = 0.0;
  double tShape = 0.0;
};

TransitionCostMap
buildTransitionCostMap(llvm::ArrayRef<analysis::TransitionOpInfo> transitionOps,
                       const TransitionCostParams &params) {
  TransitionCostMap costMap;

  for (const analysis::TransitionOpInfo &op : transitionOps) {
    switch (op.opIndex) {
    case analysis::TransitionOpIndex::ToLayout:
      costMap.tMem += params.wToLayout;
      break;

    case analysis::TransitionOpIndex::Typecast:
      costMap.tMem += params.wTypecast;
      break;

    case analysis::TransitionOpIndex::Reshape:
      costMap.tShape += params.wReshape;
      break;

    case analysis::TransitionOpIndex::Permute:
      costMap.tShape += params.wPermute;
      break;

    case analysis::TransitionOpIndex::Pad:
      costMap.tShape += params.wPad;
      break;

    case analysis::TransitionOpIndex::Unknown:
      break;
    }
  }

  return costMap;
}

bool isShardedLayout(TTNNLayoutAttr layout) {
  return layout && layout.hasShardedTensorMemoryLayout();
}

bool isDRAMLayout(TTNNLayoutAttr layout) {
  return layout && layout.getBufferType() == BufferType::DRAM;
}

bool isL1Layout(TTNNLayoutAttr layout) {
  return layout && layout.getBufferType() == BufferType::L1;
}

TensorMemoryLayout getMemoryLayout(TTNNLayoutAttr layout) {
  if (!layout || !layout.getMemLayout()) {
    return TensorMemoryLayout::Interleaved;
  }

  return layout.getMemLayout().getValue();
}

// Calculate the TSwitch cost for the layout transition from fromLayout to
// toLayout.
double computeSwitchCost(TTNNLayoutAttr fromLayout, TTNNLayoutAttr toLayout,
                         const TransitionCostParams &params) {
  if (!fromLayout || !toLayout || fromLayout == toLayout) {
    return 0.0;
  }

  if (isDRAMLayout(fromLayout) && isL1Layout(toLayout)) {
    return params.wSwitch * params.wSwitchDramToL1;
  }

  if (isL1Layout(fromLayout) && isDRAMLayout(toLayout)) {
    return params.wSwitch * params.wSwitchL1ToDram;
  }

  if (isL1Layout(fromLayout) && isL1Layout(toLayout)) {
    const bool fromSharded = isShardedLayout(fromLayout);
    const bool toSharded = isShardedLayout(toLayout);

    if (fromSharded != toSharded) {
      return params.wSwitch * params.wSwitchL1InterleavedSharded;
    }

    if (fromSharded && toSharded &&
        getMemoryLayout(fromLayout) != getMemoryLayout(toLayout)) {
      return params.wSwitch * params.wSwitchL1ShardedKind;
    }
  }

  return 0.0;
}

} // namespace

CostModel::CostModel(EmissionCostParams emissionParams,
                     TransitionCostParams transitionParams)
    : emissionCostParams(std::move(emissionParams)),
      transitionCostParams(std::move(transitionParams)) {}

// Size on DRAM equal size on L1 * number of cores.
uint64_t CostModel::getTensorTotalSize(uint64_t outputTensorUsagePerCore,
                                       TTNNLayoutAttr outputLayout) const {
  const uint64_t numCores =
      static_cast<uint64_t>(ttmlir::utils::volume(outputLayout.getGridShape()));

  return numCores * outputTensorUsagePerCore;
}

// Get core usage of a tensor layout.
double CostModel::getLayoutCores(TTNNLayoutAttr layout) const {
  if (!layout) {
    return 1.0;
  }

  const uint64_t coreUsage =
      static_cast<uint64_t>(ttmlir::utils::volume(layout.getGridShape()));

  return static_cast<double>(std::max<uint64_t>(uint64_t{1}, coreUsage));
}

CostModel::PassiveTensorList CostModel::collectPassiveTensor(
    llvm::ArrayRef<mlir::Operation *> passiveTensorProducers,
    const CandidateOutputSizesMap &storedCandidateOutputSizes) const {
  // Build one size-option list per passive producer from cached candidate
  // output bytes. Invalid/unavailable cache entries are ignored.
  PassiveTensorList passiveTensorList;
  passiveTensorList.reserve(passiveTensorProducers.size());

  for (mlir::Operation *passiveProducerOp : passiveTensorProducers) {
    // Find the cached output size list for this passive producer op.
    const auto cacheIt = storedCandidateOutputSizes.find(passiveProducerOp);

    // If not found, skip this producer.
    if (cacheIt == storedCandidateOutputSizes.end()) {
      continue;
    }

    CandidateSizeList candidateSizes;

    candidateSizes.reserve(cacheIt->second.size());

    for (const std::optional<uint64_t> &cachedBytes : cacheIt->second) {
      if (!cachedBytes || *cachedBytes == 0) {
        continue;
      }
      // All retained sizes are L1 usage bytes.
      candidateSizes.push_back(*cachedBytes);
    }

    if (candidateSizes.empty()) {
      continue;
    }

    // Sort candidate sizes in descending order
    // This ensures that we explore passive tensor options with larger L1 usage
    // first which can lead to earlier pruning in validation and thus faster
    // search.
    llvm::sort(candidateSizes,
               [](uint64_t lhs, uint64_t rhs) { return lhs > rhs; });

    passiveTensorList.push_back(std::move(candidateSizes));
  }

  // Return a list of size options for each passive tensor producer.
  return passiveTensorList;
}

// Build combinations between current op and passive tensors
// To validate if the op is actually valid with additional L1 usage.
op_constraint_validation::ValidationResult
CostModel::validateOpConfig(mlir::Operation *op,
                            const OpConfigCandidate &candidate,
                            const PassiveTensorList &passiveTensorList,
                            uint64_t &selectedAdditionalL1Usage,
                            std::size_t &selectedSpillCount) const {
  // Define a lambda to validate the candidate with given additional L1 usage
  // from passive tensors.
  const auto validateForUsage = [&](uint64_t additionalL1Usage) {
    return op_constraint_validation::validateOperation(
        op, candidate.inputLayouts, candidate.opConfig, additionalL1Usage);
  };

  op_constraint_validation::ValidationResult lastOOMResult =
      op_constraint_validation::ValidationResult::outOfMemoryError(
          "Passive combinations exhausted");

  op_constraint_validation::ValidationResult result = lastOOMResult;

  // Normal validation.
  if (passiveTensorList.empty()) {
    selectedAdditionalL1Usage = 0;
    selectedSpillCount = 0;

    return validateForUsage(
        /*additionalL1Usage=*/0);
  }

  bool foundResult = false;

  // Validation with passive tensor combinations
  // Define threshold for additional L1 usage to avoid validating combinations
  // that are very likely to be invalid and thus reduce search time.
  const uint64_t additionalUsageThreshold =
      static_cast<uint64_t>(getEffectiveL1CapacityBytes(op));

  // Get the per-core additional L1 usage for the candidate combination.
  const double candidateCores = getLayoutCores(candidate.opConfig.outputLayout);

  const auto toPerCoreAdditionalUsage =
      [candidateCores](uint64_t totalBytes) -> uint64_t {
    const double perCore = static_cast<double>(totalBytes) / candidateCores;

    return static_cast<uint64_t>(std::ceil(perCore));
  };

  const std::size_t passiveCount = passiveTensorList.size();

  const auto getOptionCount = [&](std::size_t dim,
                                  std::size_t spillCount) -> std::size_t {
    return dim < spillCount ? std::size_t{1} : passiveTensorList[dim].size();
  };

  const auto computeTotalBytes = [&](llvm::ArrayRef<std::size_t> indices,
                                     std::size_t spillCount) -> uint64_t {
    uint64_t sum = 0;

    for (std::size_t dim = 0; dim < indices.size(); ++dim) {
      if (dim < spillCount) {
        // spilled to DRAM => size 0
        continue;
      }

      sum += passiveTensorList[dim][indices[dim]];
    }

    return sum;
  };

  for (std::size_t spillCount = 0; spillCount <= passiveCount && !foundResult;
       ++spillCount) {
    auto lessByTotalBytes = [](const HeapEntry &lhs, const HeapEntry &rhs) {
      return lhs.totalBytes < rhs.totalBytes;
    };

    // Define max heap structure.
    const llvm::SmallVector<std::size_t> startIndices(passiveCount, 0);

    std::priority_queue<HeapEntry, std::vector<HeapEntry>,
                        decltype(lessByTotalBytes)>
        maxHeap(lessByTotalBytes);

    // Initialize the max heap with the starting combination (all indices at
    // 0).
    maxHeap.push(
        HeapEntry{computeTotalBytes(startIndices, spillCount), startIndices});

    // Define a visited set and start with the initial combination.
    std::unordered_set<std::string> visited;
    visited.insert(buildCombinationKey(startIndices));

    // Continue until all combinations for this spill count are explored.
    while (!maxHeap.empty() && !foundResult) {
      // Get the largest combination from the max heap.
      const HeapEntry current = maxHeap.top();
      maxHeap.pop();
      selectedAdditionalL1Usage = toPerCoreAdditionalUsage(current.totalBytes);
      selectedSpillCount = spillCount;

      // If the additional L1 usage of this combination already exceeds the
      // threshold Skip validating this combination and all the smaller
      // combinations that follow Directly move to the next spill count.
      if (selectedAdditionalL1Usage > additionalUsageThreshold) {
        // Additional passive usage already exceeds threshold, skip
        // validation.
        lastOOMResult =
            op_constraint_validation::ValidationResult::outOfMemoryError(
                "Passive additionalL1Usage exceeds threshold");
      } else {
        const op_constraint_validation::ValidationResult validationResult =
            validateForUsage(selectedAdditionalL1Usage);
        // If validation is successful, return the result.
        if (validationResult.isSuccess()) {
          result = validationResult;
          foundResult = true;

          if (selectedSpillCount > 0) {
            TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                         "Passive spill required: op={} spillCount={} "
                         "additionalL1PerCore={} outputLayout={}",
                         op->getName().getStringRef(), selectedSpillCount,
                         selectedAdditionalL1Usage,
                         optimizer_utils::layoutToString(
                             candidate.opConfig.outputLayout));
          }
        } else {
          // If validation fails, check if it's due to OOM. Non-OOM
          // failures can skip the rest of the combinations in this
          // spill count.
          foundResult =
              updateValidationFailure(validationResult, result, lastOOMResult);
        }
      }

      if (foundResult) {
        break;
      }

      for (std::size_t dim = 0; dim < current.indices.size(); ++dim) {
        llvm::SmallVector<std::size_t> nextIndices = current.indices;

        ++nextIndices[dim];

        if (nextIndices[dim] >= getOptionCount(dim, spillCount)) {
          continue;
        }

        // Check if this combination has been visited before to avoid
        // duplicates in the max heap.
        const std::string key = buildCombinationKey(nextIndices);

        if (!visited.insert(key).second) {
          continue;
        }
        // If not visited, compute the total additional L1 usage for this
        // new combination and add to the max heap.
        maxHeap.push(
            HeapEntry{computeTotalBytes(nextIndices, spillCount), nextIndices});
      }
    }
  }

  if (!foundResult) {
    result = lastOOMResult;
  }

  return result;
}

// Compute output tensor size for later passive/spill accounting.
std::optional<uint64_t> CostModel::computeOutputSize(
    std::optional<TTNNLayoutAttr> outputLayout,
    const op_constraint_validation::ValidationResult &validationResult) const {
  // If the candidate is valid and has a valid output layout, compute the output
  // tensor size.
  if (!validationResult.isSuccess() || !outputLayout || !*outputLayout) {
    return std::nullopt;
  }

  if ((*outputLayout).getBufferType() == BufferType::DRAM) {
    return uint64_t{0};
  }

  return getTensorTotalSize(validationResult.outputL1Usage, *outputLayout);
}

double CostModel::calculateEmission(
    mlir::Operation *op, const OpConfigCandidate &candidate,
    const op_constraint_validation::ValidationResult &validationResult,
    uint64_t additionalL1Usage) const {
  if (!op || !candidate.opConfig.outputLayout) {
    return inf;
  }

  const TTNNLayoutAttr outputLayout = candidate.opConfig.outputLayout;

  // Get core usage of current candidate's output layout.
  const double cores = getLayoutCores(outputLayout);

  // Normalize all usage metrics by L1 capacity to get relative usage ratios.
  const double capacity = getEffectiveL1CapacityBytes(op);

  if (capacity <= 0.0) {
    return inf;
  }

  const auto normalize = [capacity](uint64_t bytes) {
    return static_cast<double>(bytes) / capacity;
  };

  const double uTotal =
      normalize(validationResult.overallPeakL1Usage + additionalL1Usage);

  if (uTotal > emissionCostParams.tauHard) {
    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                 "Emission rejected: op={} uTotal={} > tauHard={}",
                 op->getName().getStringRef(), uTotal,
                 emissionCostParams.tauHard);

    return inf;
  }

  const double risk = std::max(0.0, uTotal - emissionCostParams.tauSoft);

  // Prefer L1 sharded, then L1 interleaved, then DRAM.
  double layoutTier = 1.0;

  if (outputLayout.hasShardedTensorMemoryLayout()) {
    layoutTier = 0.0;
  } else if (outputLayout.hasInterleavedDRAMTensorMemoryLayout()) {
    layoutTier = 2.0;
  }

  double score = emissionCostParams.wCore * (-cores);
  score += emissionCostParams.wRisk * risk * risk;
  score += emissionCostParams.wCb * normalize(validationResult.cbPeakUsage);
  score +=
      emissionCostParams.wBuf * normalize(validationResult.l1BuffersPeakUsage);
  score += emissionCostParams.wOut * normalize(validationResult.outputL1Usage);
  score += emissionCostParams.wShard * layoutTier;

  return score;
}

double CostModel::getEmissionCost(
    mlir::Operation *op, const OpConfigCandidate &candidate,
    const op_constraint_validation::ValidationResult &validationResult,
    uint64_t additionalL1Usage) const {
  return calculateEmission(op, candidate, validationResult, additionalL1Usage);
}

double
CostModel::calculateTransition(const TransitionEdgeCostInput &input) const {
  if (!input.producerOp || !input.consumerOp || !input.producerCandidate ||
      !input.consumerCandidate || !input.consumerOperand) {
    return inf;
  }

  // Unfold the input struct.
  Operation *producerOp = input.producerOp;
  Operation *consumerOp = input.consumerOp;
  const TTNNLayoutAttr producerOutputLayout =
      input.producerCandidate->opConfig.outputLayout;

  if (!producerOutputLayout) {
    return inf;
  }

  const std::optional<unsigned> consumerInputIndex =
      optimizer_utils::findTensorInputLayoutIndex(consumerOp,
                                                  input.consumerOperand);

  if (!consumerInputIndex ||
      *consumerInputIndex >= input.consumerCandidate->inputLayouts.size()) {
    return inf;
  }

  const TTNNLayoutAttr consumerInputLayout =
      input.consumerCandidate->inputLayouts[*consumerInputIndex];

  if (!consumerInputLayout) {
    return inf;
  }

  llvm::DenseMap<Operation *, OpConfig> localOpConfigMap;

  if (input.opConfigMap) {
    localOpConfigMap = *input.opConfigMap;
  }

  localOpConfigMap[producerOp] = input.producerCandidate->opConfig;

  localOpConfigMap[consumerOp] = input.consumerCandidate->opConfig;

  llvm::DenseMap<Operation *, llvm::SmallVector<TTNNLayoutAttr>>
      localInputLayoutsMap;

  localInputLayoutsMap[consumerOp] = llvm::SmallVector<TTNNLayoutAttr>(
      input.consumerCandidate->inputLayouts.begin(),
      input.consumerCandidate->inputLayouts.end());

  const llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>>
      emptySchedule;

  analysis::TransitionEdgeAnalysis transitionEdgeAnalysis(
      localOpConfigMap, localInputLayoutsMap, emptySchedule);

  const analysis::GetTransitionOpsResult transitionResult =
      transitionEdgeAnalysis.getTransitionOps(
          producerOp, input.consumerOperand, consumerOp, producerOutputLayout,
          consumerInputLayout, input.additionalL1Usage);

  if (!transitionResult.isSuccess) {
    return inf;
  }

  const TransitionCostMap costMap = buildTransitionCostMap(
      transitionResult.transitionOps, transitionCostParams);

  const double switchCost = computeSwitchCost(
      producerOutputLayout, consumerInputLayout, transitionCostParams);

  return costMap.tMem + costMap.tShape + switchCost;
}

// Getter for transition cost.
double
CostModel::getTransitionCost(const TransitionEdgeCostInput &input) const {
  return calculateTransition(input);
}

double CostModel::computeSpillCost(uint64_t selectedSpillBytes,
                                   std::optional<TTNNLayoutAttr> outputLayout,
                                   mlir::Operation *op) const {
  if (selectedSpillBytes == 0) {
    return 0.0;
  }

  const double perCoreCapacity = std::max(1.0, getEffectiveL1CapacityBytes(op));

  const double cores =
      outputLayout && *outputLayout ? getLayoutCores(*outputLayout) : 1.0;

  const double totalCapacity = std::max(1.0, perCoreCapacity * cores);

  return emissionCostParams.wSpill *
         (static_cast<double>(selectedSpillBytes) / totalCapacity);
}

// Local cost calculation for current candidate.
CostModel::LocalCostResult CostModel::getLocalCost(
    mlir::Operation *op, const OpConfigCandidate &candidate,
    llvm::ArrayRef<mlir::Operation *> passiveTensorProducers,
    bool shouldComputeOutputSize,
    const CandidateOutputSizesMap &storedCandidateOutputSizes) const {
  LocalCostResult result;
  result.cost = inf;

  if (!op || !candidate.opConfig.outputLayout) {
    return result;
  }

  uint64_t selectedAdditionalL1Usage = 0;
  std::size_t selectedSpillCount = 0;

  // Collect passive tensor at different layouts.
  const PassiveTensorList passiveTensorList =
      collectPassiveTensor(passiveTensorProducers, storedCandidateOutputSizes);

  // Validate the candidate op configuration.
  const op_constraint_validation::ValidationResult validationResult =
      validateOpConfig(op, candidate, passiveTensorList,
                       selectedAdditionalL1Usage, selectedSpillCount);

  // Compute selected spill bytes based on the L1 captured from passive tensors.
  const uint64_t selectedSpillBytes =
      computeSelectedSpillBytes(passiveTensorList, selectedSpillCount);

  // Store output tensor bytes for potential future passive/spill accounting.
  std::optional<uint64_t> outputSize;

  if (shouldComputeOutputSize) {
    outputSize =
        computeOutputSize(candidate.opConfig.outputLayout, validationResult);
  }

  if (!validationResult.isSuccess()) {
    // If the candidate is invalid
    // Non-OOM errors can skip the remaining candidates in this layout group.
    const bool isOOM =
        validationResult.status ==
        op_constraint_validation::ValidationStatus::OutOfMemoryError;

    // If candidate belong to specific candidate group (groupIndex has value),
    // skip the rest of the group.
    result.skipGroup = !isOOM && candidate.groupIndex.has_value();

    result.outputSizeBytes = outputSize;
    result.additionalL1Usage = selectedAdditionalL1Usage;
    result.selectedSpillCount = selectedSpillCount;

    TTMLIR_DEBUG(
        ttmlir::LogComponent::ViterbiOptimizer,
        "Candidate invalid: op={} outputLayout={} status={} error={} "
        "additionalL1Usage={} selectedSpillCount={} skipGroup={}",
        op->getName().getStringRef(),
        optimizer_utils::layoutToString(candidate.opConfig.outputLayout),
        op_constraint_validation::validationStatusToString(
            validationResult.status),
        validationResult.errorMessage.empty() ? std::string("<none>")
                                              : validationResult.errorMessage,
        selectedAdditionalL1Usage, selectedSpillCount, result.skipGroup);

    return result;
  }

  // Retrieve emission cost and spill cost for local cost calculation.
  const double emission = getEmissionCost(op, candidate, validationResult,
                                          selectedAdditionalL1Usage);

  const double spillCost =
      computeSpillCost(selectedSpillBytes, candidate.opConfig.outputLayout, op);

  // Assign cost and other related info to the result struct.
  result.cost = emission + spillCost;
  result.skipGroup = false;
  result.outputSizeBytes = outputSize;
  result.additionalL1Usage = selectedAdditionalL1Usage;
  result.selectedSpillCount = selectedSpillCount;

  TTMLIR_DEBUG(
      ttmlir::LogComponent::ViterbiOptimizer,
      "Candidate cost: op={} inputLayouts={} outputLayout={} "
      "cost={} emission={} spill={} "
      "cbPeak={} l1BuffersPeak={} outputL1PerCore={} "
      "overallPeak={} additionalL1={} spillCount={} utility={}",
      op->getName().getStringRef(),
      optimizer_utils::layoutsToString(std::vector<TTNNLayoutAttr>(
          candidate.inputLayouts.begin(), candidate.inputLayouts.end())),
      optimizer_utils::layoutToString(candidate.opConfig.outputLayout),
      result.cost, emission, spillCost, validationResult.cbPeakUsage,
      validationResult.l1BuffersPeakUsage, validationResult.outputL1Usage,
      validationResult.overallPeakL1Usage, selectedAdditionalL1Usage,
      selectedSpillCount,
      computeL1Utility(op, validationResult, selectedAdditionalL1Usage));

  return result;
}

} // namespace mlir::tt::ttnn
