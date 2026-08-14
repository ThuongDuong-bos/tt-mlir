// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ReallocationAnalysis.h"

#include "ttmlir/Dialect/TTNN/Interfaces/TTNNOpModelInterface.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace mlir::tt::ttnn {

namespace {

bool isDeviceBuffer(Value value) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(value.getType());
  return !tensorType || utils::isTensorOnDevice(tensorType);
}

bool isDeviceBufferOp(Operation *op) {
  if (!op) {
    return false;
  }

  for (Value operand : op->getOperands()) {
    if (mlir::isa<RankedTensorType>(operand.getType()) &&
        !isDeviceBuffer(operand)) {
      return false;
    }
  }

  for (Value result : op->getResults()) {
    if (mlir::isa<RankedTensorType>(result.getType()) &&
        !isDeviceBuffer(result)) {
      return false;
    }
  }

  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// ShardAllocationModel
//===----------------------------------------------------------------------===//

bool ShardAllocationModel::allocate(TensorMemoryEntry &tensor) {
  assert(!tensor.isAllocated());

  for (std::size_t blockIndex = freeBlocks.size(); blockIndex > 0;
       --blockIndex) {
    FreeBlock &block = freeBlocks[blockIndex - 1];
    const std::size_t blockSize = block.endAddress - block.startAddress;
    if (blockSize < tensor.getBufferSize()) {
      continue;
    }

    const std::size_t allocAddress = block.endAddress - tensor.getBufferSize();
    tensor.allocate(allocAddress);
    shardedTensors.push_back(&tensor);

    if (blockSize == tensor.getBufferSize()) {
      freeBlocks.erase(freeBlocks.begin() + (blockIndex - 1));
    } else {
      block.endAddress -= tensor.getBufferSize();
    }

    return true;
  }

  return false;
}

bool ShardAllocationModel::deallocate(TensorMemoryEntry &tensor) {
  auto it = llvm::find(shardedTensors, &tensor);
  if (it == shardedTensors.end()) {
    return false;
  }

  const std::size_t startAddress = tensor.start();
  const std::size_t endAddress = tensor.end();

  shardedTensors.erase(it);
  tensor.deallocate();
  freeBlocks.push_back({startAddress, endAddress});

  llvm::sort(freeBlocks, [](const FreeBlock &lhs, const FreeBlock &rhs) {
    return lhs.startAddress < rhs.startAddress;
  });

  llvm::SmallVector<FreeBlock, 16> merged;
  for (const FreeBlock &block : freeBlocks) {
    if (merged.empty() || merged.back().endAddress < block.startAddress) {
      merged.push_back(block);
      continue;
    }

    merged.back().endAddress =
        std::max(merged.back().endAddress, block.endAddress);
  }

  freeBlocks = std::move(merged);
  return true;
}

bool ShardAllocationModel::reallocate(TensorMemoryEntry &tensor) {
  if (!tensor.isAllocated() || !llvm::is_contained(shardedTensors, &tensor)) {
    return false;
  }

  const std::size_t oldAddress = tensor.getLogicalAddress();
  if (!deallocate(tensor) || !allocate(tensor)) {
    return false;
  }

  // The model compacts toward the top of L1. A successful reallocation should
  // therefore never move a tensor toward the CB region.
  assert(tensor.getLogicalAddress() >= oldAddress);
  return true;
}

std::size_t ShardAllocationModel::getLowestAllocatedAddress() const {
  std::size_t lowestAddress = totalL1Size;

  for (const TensorMemoryEntry *entry : shardedTensors) {
    if (entry && entry->isAllocated()) {
      lowestAddress = std::min(lowestAddress, entry->start());
    }
  }

  return lowestAddress;
}

std::size_t
ShardAllocationModel::getHighestCBAddress(std::size_t peakCBMemorySize) const {
  const std::size_t offset = static_cast<std::size_t>(totalL1Size * offsetCap);
  return peakCBMemorySize + offset;
}

bool ShardAllocationModel::isAllocated(const TensorMemoryEntry &tensor) const {
  return tensor.isAllocated() && llvm::is_contained(shardedTensors, &tensor);
}

bool ShardAllocationModel::isSameState(
    const ShardAllocationModel &other) const {
  if (freeBlocks.size() != other.freeBlocks.size()) {
    return false;
  }

  for (std::size_t blockIndex = 0; blockIndex < freeBlocks.size();
       ++blockIndex) {
    if (freeBlocks[blockIndex].startAddress !=
            other.freeBlocks[blockIndex].startAddress ||
        freeBlocks[blockIndex].endAddress !=
            other.freeBlocks[blockIndex].endAddress) {
      return false;
    }
  }

  return true;
}

ShardAllocationModel ShardAllocationModel::cloneWithNewEntries(
    llvm::DenseMap<const TensorMemoryEntry *, TensorMemoryEntry *> &entryMap)
    const {
  ShardAllocationModel copy(totalL1Size, offsetCap);
  copy.freeBlocks = freeBlocks;

  for (const TensorMemoryEntry *oldEntry : shardedTensors) {
    auto *newEntry =
        new TensorMemoryEntry(oldEntry->getBufferSize(), oldEntry->getValue());

    if (oldEntry->isAllocated()) {
      newEntry->allocate(oldEntry->getLogicalAddress());
    }

    copy.shardedTensors.push_back(newEntry);
    entryMap[oldEntry] = newEntry;
  }

  return copy;
}

//===----------------------------------------------------------------------===//
// ReallocationAnalysis
//===----------------------------------------------------------------------===//

bool ReallocationAnalysis::isOpSupportedForReallocate(Operation *op) const {
  if (!op || !mlir::isa<TTNNDialect>(op->getDialect()) ||
      !analysisInput.opConfigMap.contains(op)) {
    return false;
  }

  if (!isDeviceBufferOp(op) || !mlir::dyn_cast<OpModel>(op)) {
    return false;
  }

  // Keep the original limitation: do not model operations with explicit
  // allocation effects as safe reallocation points.
  if (auto memoryEffect = mlir::dyn_cast<MemoryEffectOpInterface>(op)) {
    llvm::SmallVector<MemoryEffects::EffectInstance, 1> effects;
    memoryEffect.getEffects(effects);
    for (const auto &effect : effects) {
      if (mlir::isa<MemoryEffects::Allocate>(effect.getEffect())) {
        return false;
      }
    }
  }

  return true;
}

void ReallocationAnalysis::finishOp(Operation *op) { doneOps.insert(op); }

TensorMemoryEntry &
ReallocationAnalysis::getOrCreateTensorEntry(std::size_t bufferSize,
                                             Value value) {
  for (TensorMemoryEntry *entry : tensorCache) {
    if (entry->equalsByValue(value)) {
      return *entry;
    }
  }

  auto *entry = new TensorMemoryEntry(bufferSize, value);
  tensorCache.insert(entry);
  return *entry;
}

bool ReallocationAnalysis::allocateOutputs(
    ShardAllocationModel &model,
    const llvm::SmallVector<TensorMemoryEntry *> &outputEntries) {
  for (TensorMemoryEntry *entry : outputEntries) {
    if (!entry || !entry->isL1Sharded()) {
      continue;
    }

    if (!model.allocate(*entry)) {
      return false;
    }
  }

  return true;
}

bool ReallocationAnalysis::deallocateInputs(
    ShardAllocationModel &model,
    const llvm::SmallVector<TensorMemoryEntry *> &inputEntries) {
  bool success = true;

  for (TensorMemoryEntry *entry : inputEntries) {
    if (!entry || !entry->isL1Sharded() || !entry->getValue()) {
      continue;
    }

    bool lifetimeEnded = true;
    for (Operation *user : entry->getValue().getUsers()) {
      if (!doneOps.contains(user)) {
        lifetimeEnded = false;
        break;
      }
    }

    if (lifetimeEnded && model.isAllocated(*entry)) {
      success &= model.deallocate(*entry);
    }
  }

  return success;
}

bool ReallocationAnalysis::isNeededReallocation(
    const ShardAllocationModel &allocator, std::size_t peakCBMemorySize,
    std::size_t additionalUsage) const {
  const std::size_t highestCBAddress =
      allocator.getHighestCBAddress(peakCBMemorySize + additionalUsage);
  return highestCBAddress >= allocator.getLowestAllocatedAddress();
}

bool ReallocationAnalysis::insertReallocate(
    ShardAllocationModel &model, Operation *currentOp,
    llvm::ArrayRef<TensorMemoryEntry *> outputEntries) {
  for (TensorMemoryEntry *candidate : model.getShardedTensors()) {
    if (!candidate || !model.isAllocated(*candidate) ||
        llvm::is_contained(outputEntries, candidate)) {
      continue;
    }

    Value value = candidate->getValue();
    if (!value || !value.getDefiningOp()) {
      continue;
    }

    // A tensor with no remaining use cannot contribute to future allocation
    // pressure and does not need to be materialized through ReallocateOp.
    const bool hasRemainingUse =
        llvm::any_of(value.getUsers(),
                     [&](Operation *user) { return !doneOps.contains(user); });
    if (!hasRemainingUse) {
      continue;
    }

    if (!model.reallocate(*candidate)) {
      continue;
    }

    // Preserve exact SSA identity. The original PR converted this decision into
    // Edge(producerOp, consumerOp, operandIdx), which loses result identity for
    // multi-result producers.
    analysisResult.memReallocateValuesMap[currentOp].insert(value);
    return true;
  }

  return false;
}

bool ReallocationAnalysis::applyOverrides() { return false; }

void ReallocationAnalysis::analysisImplementation() {
  doneOps.clear();

  op->walk([&](func::FuncOp func) {
    auto scheduleIt = analysisInput.schedule.find(func);
    if (!ttmlir::utils::isForwardDeviceFunc(func) ||
        scheduleIt == analysisInput.schedule.end()) {
      return;
    }

    ShardAllocationModel allocModel(analysisInput.usableL1CacheSize,
                                    analysisInput.offsetCap);

    for (Operation *currentOp : scheduleIt->second) {
      if (!isOpSupportedForReallocate(currentOp)) {
        finishOp(currentOp);
        continue;
      }

      auto configIt = analysisInput.opConfigMap.find(currentOp);
      assert(configIt != analysisInput.opConfigMap.end());

      const std::vector<TTNNLayoutAttr> inputLayouts =
          utils::extractInputLayouts(currentOp);

      const op_constraint_validation::ValidationResult validation =
          op_constraint_validation::validateOperation(currentOp, inputLayouts,
                                                      configIt->second);

      if (!validation.isSuccess()) {
        finishOp(currentOp);
        continue;
      }

      const std::size_t cbPeakUsage = validation.cbPeakUsage;
      const std::size_t outputTensorUsagePerCore = validation.outputL1Usage;

      // The current BOS validation result exposes the backend tensor-buffer
      // peak separately from CB usage. Keep the same approximation used by the
      // original analysis for input tensors.
      const std::size_t l1BuffersPeakUsage = validation.l1BuffersPeakUsage;
      const std::size_t inputTensorUsagePerCore =
          l1BuffersPeakUsage > outputTensorUsagePerCore
              ? l1BuffersPeakUsage - outputTensorUsagePerCore
              : 0;

      llvm::SmallVector<TensorMemoryEntry *> inputEntries;
      llvm::SmallVector<TensorMemoryEntry *> outputEntries;
      std::size_t additionalUsage = l1BuffersPeakUsage;

      for (Value outputValue : analysisInput.getOutputValues(currentOp)) {
        TensorMemoryEntry &entry =
            getOrCreateTensorEntry(outputTensorUsagePerCore, outputValue);
        if (!entry.isL1Sharded()) {
          continue;
        }

        outputEntries.push_back(&entry);
        additionalUsage -= std::min(additionalUsage, entry.getBufferSize());
      }

      for (Value inputValue : analysisInput.getInputValues(currentOp)) {
        TensorMemoryEntry &entry =
            getOrCreateTensorEntry(inputTensorUsagePerCore, inputValue);
        if (!entry.isL1Sharded()) {
          continue;
        }

        inputEntries.push_back(&entry);
        additionalUsage -= std::min(additionalUsage, entry.getBufferSize());
      }

      ShardAllocationModel workingModel = allocModel;
      ShardAllocationModel previousModel = workingModel;

      bool needReallocation = false;
      bool reallocationSucceeded = false;

      while (true) {
        llvm::SmallVector<TensorMemoryEntry> temporaryOutputs;
        llvm::SmallVector<TensorMemoryEntry *> temporaryOutputPtrs;
        temporaryOutputs.reserve(outputEntries.size());
        temporaryOutputPtrs.reserve(outputEntries.size());

        for (TensorMemoryEntry *entry : outputEntries) {
          temporaryOutputs.emplace_back(entry->getBufferSize(),
                                        entry->getValue());
          temporaryOutputPtrs.push_back(&temporaryOutputs.back());
        }

        llvm::DenseMap<const TensorMemoryEntry *, TensorMemoryEntry *> entryMap;
        ShardAllocationModel testModel =
            workingModel.cloneWithNewEntries(entryMap);

        const bool canAllocateOutputs =
            allocateOutputs(testModel, temporaryOutputPtrs);

        needReallocation =
            !canAllocateOutputs ||
            isNeededReallocation(testModel, cbPeakUsage, additionalUsage);

        if (!needReallocation) {
          allocModel = std::move(workingModel);
          break;
        }

        if (!insertReallocate(workingModel, currentOp, outputEntries)) {
          break;
        }

        reallocationSucceeded = true;

        if (previousModel.isSameState(workingModel)) {
          allocModel = std::move(workingModel);
          break;
        }

        previousModel = workingModel;
      }

      if (needReallocation && !reallocationSucceeded) {
        TTMLIR_DEBUG(
            ttmlir::LogComponent::ViterbiOptimizer,
            "ReallocationAnalysis: unable to resolve fragmentation before {}",
            currentOp->getName().getStringRef());
      }

      if (!allocateOutputs(allocModel, outputEntries)) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "ReallocationAnalysis: output allocation failed for {}",
                     currentOp->getName().getStringRef());
      }

      deallocateInputs(allocModel, inputEntries);
      finishOp(currentOp);
    }
  });
}

} // namespace mlir::tt::ttnn
