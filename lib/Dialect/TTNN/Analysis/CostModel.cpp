// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/CostModel.h"

#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

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

struct HeapEntry {
  uint64_t totalBytes = 0;
  llvm::SmallVector<size_t> indices;
};

std::string buildCombinationKey(llvm::ArrayRef<size_t> indices) {
  std::string key;
  key.reserve(indices.size() * 4);

  for (size_t index : indices) {
    key.append(std::to_string(index));
    key.push_back('#');
  }

  return key;
}

uint64_t
computeSelectedSpillBytes(const CostModel::PassiveTensorList &passiveTensorList,
                          size_t selectedSpillCount) {
  uint64_t selectedSpillBytes = 0;

  for (size_t spillIdx = 0;
       spillIdx < selectedSpillCount &&
       spillIdx < passiveTensorList.size();
       ++spillIdx) {
    if (passiveTensorList[spillIdx].empty()) {
      continue;
    }

    selectedSpillBytes += passiveTensorList[spillIdx][0];
  }

  return selectedSpillBytes;
}

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
  size_t selectedSpillCount = 0;

  const PassiveTensorList passiveTensorList =
      collectPassiveTensor(passiveTensorProducers,
                           storedCandidateOutputSizes);

  const op_constraint_validation::ValidationResult validationResult =
      validateOpConfig(op, candidate, passiveTensorList,
                       selectedAdditionalL1Usage, selectedSpillCount);

  const uint64_t selectedSpillBytes =
      computeSelectedSpillBytes(passiveTensorList, selectedSpillCount);

  std::optional<uint64_t> outputSize;

  if (shouldComputeOutputSize) {
    outputSize =
        computeOutputSize(candidate.opConfig.outputLayout, validationResult);
  }

  if (!validationResult.isSuccess()) {
    const bool isOOM =
        validationResult.status ==
        op_constraint_validation::ValidationStatus::OutOfMemoryError;

    result.skipGroup = !isOOM && candidate.groupIndex.has_value();
    result.outputSizeBytes = outputSize;

    TTMLIR_DEBUG(
        ttmlir::LogComponent::ViterbiOptimizer,
        "Candidate invalid: op={} outputLayout={} status={} error={} "
        "additionalL1Usage={} skipGroup={}",
        op->getName().getStringRef(),
        optimizer_utils::layoutToString(candidate.opConfig.outputLayout),
        op_constraint_validation::validationStatusToString(
            validationResult.status),
        validationResult.errorMessage.empty()
            ? std::string("<none>")
            : validationResult.errorMessage,
        selectedAdditionalL1Usage, result.skipGroup);

    return result;
  }

  const double emission =
      getEmissionCost(op, candidate, validationResult,
                      selectedAdditionalL1Usage);

  const double spillCost =
      computeSpillCost(selectedSpillBytes,
                       candidate.opConfig.outputLayout, op);

  result.cost = emission + spillCost;
  result.skipGroup = false;
  result.outputSizeBytes = outputSize;
  result.additionalL1Usage = selectedAdditionalL1Usage;

  TTMLIR_DEBUG(
      ttmlir::LogComponent::ViterbiOptimizer,
      "Candidate cost: op={} inputLayouts={} outputLayout={} "
      "cost={} emission={} spill={} "
      "cbPeak={} l1BuffersPeak={} outputL1PerCore={} "
      "overallPeak={} additionalL1={} utility={}",
      op->getName().getStringRef(),
      optimizer_utils::layoutsToString(std::vector<TTNNLayoutAttr>(
          candidate.inputLayouts.begin(), candidate.inputLayouts.end())),
      optimizer_utils::layoutToString(candidate.opConfig.outputLayout),
      result.cost, emission, spillCost, validationResult.cbPeakUsage,
      validationResult.l1BuffersPeakUsage, validationResult.outputL1Usage,
      validationResult.overallPeakL1Usage, selectedAdditionalL1Usage,
      computeL1Utility(op, validationResult, selectedAdditionalL1Usage));

  return result;
}

double
CostModel::getTransitionCost(const TransitionEdgeCostInput &input) const {
  return calculateTransition(input);
}

std::optional<uint64_t> CostModel::computeOutputSize(
    std::optional<TTNNLayoutAttr> outputLayout,
    const op_constraint_validation::ValidationResult &validationResult) const {
  if (!validationResult.isSuccess() || !outputLayout || !*outputLayout) {
    return std::nullopt;
  }

  if ((*outputLayout).getBufferType() == BufferType::DRAM) {
    return uint64_t{0};
  }

  return getTensorTotalSize(validationResult.outputL1Usage, *outputLayout);
}

double CostModel::getEmissionCost(
    mlir::Operation *op, const OpConfigCandidate &candidate,
    const op_constraint_validation::ValidationResult &validationResult,
    uint64_t additionalL1Usage) const {
  return calculateEmission(op, candidate, validationResult,
                           additionalL1Usage);
}

double CostModel::calculateEmission(
    mlir::Operation *op, const OpConfigCandidate &candidate,
    const op_constraint_validation::ValidationResult &validationResult,
    uint64_t additionalL1Usage) const {
  if (!op || !candidate.opConfig.outputLayout) {
    return inf;
  }

  const TTNNLayoutAttr outputLayout = candidate.opConfig.outputLayout;

  const double cores = getLayoutCores(outputLayout);
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
    TTMLIR_DEBUG(
        ttmlir::LogComponent::ViterbiOptimizer,
        "Emission rejected: op={} uTotal={} > tauHard={}",
        op->getName().getStringRef(), uTotal, emissionCostParams.tauHard);

    return inf;
  }

  const double risk =
      std::max(0.0, uTotal - emissionCostParams.tauSoft);

  // Prefer L1 sharded, then L1 interleaved, then DRAM.
  double layoutTier = 1.0;

  if (outputLayout.hasShardedTensorMemoryLayout()) {
    layoutTier = 0.0;
  } else if (outputLayout.hasInterleavedDRAMTensorMemoryLayout()) {
    layoutTier = 2.0;
  }

  double score = emissionCostParams.wCore * (-cores);

  score += emissionCostParams.wRisk * risk * risk;

  score += emissionCostParams.wCb *
           normalize(validationResult.cbPeakUsage);

  score += emissionCostParams.wBuf *
           normalize(validationResult.l1BuffersPeakUsage);

  score += emissionCostParams.wOut *
           normalize(validationResult.outputL1Usage);

  score += emissionCostParams.wShard * layoutTier;

  return score;
}

double
CostModel::calculateTransition(const TransitionEdgeCostInput &input) const {
  if (!input.producerOp || !input.consumerOp || !input.producerCandidate ||
      !input.consumerCandidate || !input.consumerOperand) {
    return inf;
  }

  const TTNNLayoutAttr producerLayout =
      input.producerCandidate->opConfig.outputLayout;

  if (!producerLayout) {
    return inf;
  }

  const std::optional<unsigned> consumerInputLayoutIndex =
      optimizer_utils::findTensorInputLayoutIndex(input.consumerOp,
                                                  input.consumerOperand);

  if (!consumerInputLayoutIndex ||
      *consumerInputLayoutIndex >= input.consumerCandidate->inputLayouts.size()) {
    return inf;
  }

  const TTNNLayoutAttr consumerLayout =
      input.consumerCandidate->inputLayouts[*consumerInputLayoutIndex];

  if (!consumerLayout) {
    return inf;
  }

  if (producerLayout == consumerLayout) {
    return 0.0;
  }

  // TransitionEdgeAnalysis later creates the exact SSA transition chain.
  // CostModel only estimates the relative layout-transition penalty here.
  double transitionCost = transitionCostParams.wToLayout;

  if (producerLayout.getDataType() != consumerLayout.getDataType()) {
    transitionCost += transitionCostParams.wTypecast;
  }

  transitionCost +=
      computeSwitchCost(producerLayout, consumerLayout, transitionCostParams);

  return transitionCost;
}

double CostModel::computeSpillCost(
    uint64_t selectedSpillBytes,
    std::optional<TTNNLayoutAttr> outputLayout, mlir::Operation *op) const {
  if (selectedSpillBytes == 0) {
    return 0.0;
  }

  const double perCoreCapacity =
      std::max(1.0, getEffectiveL1CapacityBytes(op));

  const double cores =
      outputLayout && *outputLayout ? getLayoutCores(*outputLayout) : 1.0;

  const double totalCapacity =
      std::max(1.0, perCoreCapacity * cores);

  return emissionCostParams.wSpill *
         (static_cast<double>(selectedSpillBytes) / totalCapacity);
}

uint64_t
CostModel::getTensorTotalSize(uint64_t outputTensorUsagePerCore,
                              TTNNLayoutAttr outputLayout) const {
  const uint64_t numCores =
      static_cast<uint64_t>(ttmlir::utils::volume(outputLayout.getGridShape()));

  return numCores * outputTensorUsagePerCore;
}

double CostModel::getLayoutCores(TTNNLayoutAttr layout) const {
  if (!layout) {
    return 1.0;
  }

  const uint64_t coreUsage =
      static_cast<uint64_t>(ttmlir::utils::volume(layout.getGridShape()));

  return static_cast<double>(
      std::max<uint64_t>(uint64_t{1}, coreUsage));
}

CostModel::PassiveTensorList CostModel::collectPassiveTensor(
    llvm::ArrayRef<mlir::Operation *> passiveTensorProducers,
    const CandidateOutputSizesMap &storedCandidateOutputSizes) const {
  PassiveTensorList passiveTensorList;
  passiveTensorList.reserve(passiveTensorProducers.size());

  for (mlir::Operation *passiveProducerOp : passiveTensorProducers) {
    const auto cacheIt =
        storedCandidateOutputSizes.find(passiveProducerOp);

    if (cacheIt == storedCandidateOutputSizes.end()) {
      continue;
    }

    CandidateSizeList candidateSizes;
    candidateSizes.reserve(cacheIt->second.size());

    for (const std::optional<uint64_t> &cachedBytes : cacheIt->second) {
      if (!cachedBytes || *cachedBytes == 0) {
        continue;
      }

      candidateSizes.push_back(*cachedBytes);
    }

    if (candidateSizes.empty()) {
      continue;
    }

    llvm::sort(candidateSizes,
               [](uint64_t lhs, uint64_t rhs) { return lhs > rhs; });

    passiveTensorList.push_back(std::move(candidateSizes));
  }

  return passiveTensorList;
}

op_constraint_validation::ValidationResult CostModel::validateOpConfig(
    mlir::Operation *op, const OpConfigCandidate &candidate,
    const PassiveTensorList &passiveTensorList,
    uint64_t &selectedAdditionalL1Usage,
    size_t &selectedSpillCount) const {
  const auto validateForUsage = [&](uint64_t additionalL1Usage) {
    return op_constraint_validation::validateOperation(
        op, candidate.inputLayouts, candidate.opConfig, additionalL1Usage);
  };

  op_constraint_validation::ValidationResult lastOOMResult =
      op_constraint_validation::ValidationResult::outOfMemoryError(
          "Passive combinations exhausted");

  op_constraint_validation::ValidationResult result = lastOOMResult;

  if (passiveTensorList.empty()) {
    selectedAdditionalL1Usage = 0;
    selectedSpillCount = 0;

    return validateForUsage(/*additionalL1Usage=*/0);
  }

  bool foundResult = false;

  const uint64_t additionalUsageThreshold =
      static_cast<uint64_t>(getEffectiveL1CapacityBytes(op));

  const double candidateCores =
      getLayoutCores(candidate.opConfig.outputLayout);

  const auto toPerCoreAdditionalUsage =
      [candidateCores](uint64_t totalBytes) -> uint64_t {
    const double perCore =
        static_cast<double>(totalBytes) / candidateCores;

    return static_cast<uint64_t>(std::ceil(perCore));
  };

  const size_t passiveCount = passiveTensorList.size();

  const auto getOptionCount = [&](size_t dim,
                                  size_t spillCount) -> size_t {
    return dim < spillCount ? size_t{1}
                            : passiveTensorList[dim].size();
  };

  const auto computeTotalBytes =
      [&](llvm::ArrayRef<size_t> indices,
          size_t spillCount) -> uint64_t {
    uint64_t sum = 0;

    for (size_t dim = 0; dim < indices.size(); ++dim) {
      if (dim < spillCount) {
        continue;
      }

      sum += passiveTensorList[dim][indices[dim]];
    }

    return sum;
  };

  for (size_t spillCount = 0;
       spillCount <= passiveCount && !foundResult; ++spillCount) {
    auto lessByTotalBytes =
        [](const HeapEntry &lhs, const HeapEntry &rhs) {
          return lhs.totalBytes < rhs.totalBytes;
        };

    const llvm::SmallVector<size_t> startIndices(passiveCount, 0);

    std::priority_queue<HeapEntry, std::vector<HeapEntry>,
                        decltype(lessByTotalBytes)>
        maxHeap(lessByTotalBytes);

    maxHeap.push(
        HeapEntry{computeTotalBytes(startIndices, spillCount), startIndices});

    std::unordered_set<std::string> visited;
    visited.insert(buildCombinationKey(startIndices));

    while (!maxHeap.empty() && !foundResult) {
      const HeapEntry current = maxHeap.top();
      maxHeap.pop();

      selectedAdditionalL1Usage =
          toPerCoreAdditionalUsage(current.totalBytes);

      selectedSpillCount = spillCount;

      if (selectedAdditionalL1Usage > additionalUsageThreshold) {
        lastOOMResult =
            op_constraint_validation::ValidationResult::outOfMemoryError(
                "Passive additionalL1Usage exceeds threshold");
      } else {
        const op_constraint_validation::ValidationResult validationResult =
            validateForUsage(selectedAdditionalL1Usage);

        if (validationResult.isSuccess()) {
          result = validationResult;
          foundResult = true;

          if (selectedSpillCount > 0) {
            TTMLIR_DEBUG(
                ttmlir::LogComponent::ViterbiOptimizer,
                "Passive spill required: op={} spillCount={} "
                "additionalL1PerCore={} outputLayout={}",
                op->getName().getStringRef(), selectedSpillCount,
                selectedAdditionalL1Usage,
                optimizer_utils::layoutToString(
                    candidate.opConfig.outputLayout));
          }
        } else {
          foundResult = updateValidationFailure(
              validationResult, result, lastOOMResult);
        }
      }

      if (foundResult) {
        break;
      }

      for (size_t dim = 0; dim < current.indices.size(); ++dim) {
        llvm::SmallVector<size_t> nextIndices = current.indices;

        ++nextIndices[dim];

        if (nextIndices[dim] >= getOptionCount(dim, spillCount)) {
          continue;
        }

        const std::string key = buildCombinationKey(nextIndices);

        if (!visited.insert(key).second) {
          continue;
        }

        maxHeap.push(HeapEntry{
            computeTotalBytes(nextIndices, spillCount), nextIndices});
      }
    }
  }

  if (!foundResult) {
    result = lastOOMResult;
  }

  return result;
}

} // namespace mlir::tt::ttnn