// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_REALLOCATIONANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_REALLOCATIONANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <utility>

namespace mlir::tt::ttnn {

struct ReallocationAnalysisInput {
  llvm::DenseMap<Operation *, OpConfig> opConfigMap;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> schedule;
  llvm::DenseMap<Operation *, llvm::DenseSet<Value>> inputValueMap;
  llvm::DenseMap<Operation *, llvm::DenseSet<Value>> outputValueMap;
  std::size_t usableL1CacheSize = 0;
  /// Max allowed offset as percentage of total L1 size, e.g. 0.5 for 50%
  double offsetCap = 0.10;

  ReallocationAnalysisInput() = default;

  ReallocationAnalysisInput(
      llvm::DenseMap<Operation *, OpConfig> opConfigMap,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> schedule,
      std::size_t usableL1CacheSize, double offsetCap)
      : opConfigMap(std::move(opConfigMap)), schedule(std::move(schedule)),
        usableL1CacheSize(usableL1CacheSize), offsetCap(offsetCap) {
    for (const auto &[_, operations] : this->schedule) {
      for (Operation *operation : operations) {
        llvm::DenseSet<Value> inputValues;
        llvm::DenseSet<Value> outputValues;

        for (Value operand : operation->getOperands()) {
          if (operand.getDefiningOp()) {
            inputValues.insert(operand);
          }
        }

        for (Value result : operation->getResults()) {
          if (!result.use_empty()) {
            outputValues.insert(result);
          }
        }

        inputValueMap[operation] = std::move(inputValues);
        outputValueMap[operation] = std::move(outputValues);
      }
    }
  }

  llvm::DenseSet<Value> getInputValues(Operation *operation) const {
    auto inputIt = inputValueMap.find(operation);
    if (inputIt == inputValueMap.end()) {
      return {};
    }

    return inputIt->second;
  }

  llvm::DenseSet<Value> getOutputValues(Operation *operation) const {
    auto outputIt = outputValueMap.find(operation);
    if (outputIt == outputValueMap.end()) {
      return {};
    }

    return outputIt->second;
  }

  bool operator==(const ReallocationAnalysisInput &other) const {
    return opConfigMap == other.opConfigMap && schedule == other.schedule;
  }

  bool operator!=(const ReallocationAnalysisInput &other) const {
    return !(*this == other);
  }
};

/// Result of reallocation analysis.
///
/// Contains the set of graph values where a reallocation operation
/// must be inserted to avoid L1 memory fragmentation or CB overlap.
struct ReallocationAnalysisResult {
  llvm::DenseMap<Operation *, llvm::DenseSet<Value>> memReallocateValuesMap;

  bool operator==(const ReallocationAnalysisResult &other) const {
    return memReallocateValuesMap == other.memReallocateValuesMap;
  }

  bool operator!=(const ReallocationAnalysisResult &other) const {
    return !(*this == other);
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &output,
                                     const ReallocationAnalysisResult &result) {
  output << "ReallocationAnalysisResult {\n";

  if (result.memReallocateValuesMap.empty()) {
    output << "  <empty>\n";
  }

  for (const auto &[insertBeforeOp, values] : result.memReallocateValuesMap) {
    output << "  Insert before op: ";
    if (insertBeforeOp) {
      insertBeforeOp->print(output);
    } else {
      output << "<null>";
    }
    output << "\n";

    for (Value value : values) {
      output << "    - ";
      value.printAsOperand(output, mlir::OpPrintingFlags{});
      output << "\n";
    }
  }

  output << "}";
  return output;
}

struct FreeBlock {
  std::size_t startAddress = 0;
  std::size_t endAddress = 0;
};

/// Logical representation of a tensor allocation in L1.
///
/// This is an approximate model used for static analysis only.
class TensorMemoryEntry {
public:
  TensorMemoryEntry(std::size_t bufferSize, Value value)
      : bufferSize(bufferSize), value(value) {}

  bool isAllocated() const { return logicalAddress != unallocatedAddress; }

  void allocate(std::size_t address) {
    assert(!isAllocated());
    logicalAddress = address;
  }

  void deallocate() {
    assert(isAllocated());
    logicalAddress = unallocatedAddress;
  }

  std::size_t getLogicalAddress() const { return logicalAddress; }

  std::size_t start() const { return logicalAddress; }

  std::size_t end() const { return logicalAddress + bufferSize; }

  std::size_t getBufferSize() const { return bufferSize; }

  Value getValue() const { return value; }

  bool isL1Sharded() const {
    if (!value) {
      return false;
    }

    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType) {
      return false;
    }

    auto layout = dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    return layout && isL1BufferType(layout.getBufferType()) &&
           layout.hasShardedTensorMemoryLayout();
  }

  bool isL1() const {
    if (!value) {
      return false;
    }

    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType) {
      return false;
    }

    auto layout = dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    return layout && isL1BufferType(layout.getBufferType()) &&
           layout.getMemLayout();
  }

  bool equalsByValue(Value otherValue) const { return value == otherValue; }

  bool operator==(const TensorMemoryEntry &other) const {
    return logicalAddress == other.logicalAddress &&
           bufferSize == other.bufferSize && value == other.value;
  }

  bool operator!=(const TensorMemoryEntry &other) const {
    return !(*this == other);
  }

private:
  static constexpr std::size_t unallocatedAddress =
      static_cast<std::size_t>(-1);

  std::size_t logicalAddress = unallocatedAddress;
  std::size_t bufferSize = 0;
  Value value;
};

/// Simulated L1 allocator for sharded tensors.
///
/// Models a runtime free-list allocator using:
//        first-fit + top-down placement.
/// Used to detect overlapping allocations between tensor buffers
/// and circular buffers (CBs).
class ShardAllocationModel {
public:
  explicit ShardAllocationModel(std::size_t totalL1Size)
      : totalL1Size(totalL1Size) {
    freeBlocks.push_back({0, totalL1Size});
  }

  ShardAllocationModel(std::size_t totalL1Size, double offsetCap)
      : totalL1Size(totalL1Size), offsetCap(offsetCap) {
    freeBlocks.push_back({0, totalL1Size});
  }

  /// Allocate a tensor using first-fit strategy.
  bool allocate(TensorMemoryEntry &tensor);

  /// Reallocate a tensor, this tensor must be exist in tensors
  bool deallocate(TensorMemoryEntry &tensor);

  /// Deallocate a tensor entry.
  bool reallocate(TensorMemoryEntry &tensor);

  /// Return the lowest allocated logical address in L1.
  std::size_t getLowestAllocatedAddress() const;

  /// Return highest address occupied by CB given the peak CB memory usage
  std::size_t getHighestCBAddress(std::size_t peakCBMemorySize) const;

  /// Is exist in the allocation
  bool isAllocated(const TensorMemoryEntry &tensor) const;

  /// Check 2 models has same state by free blocks address
  bool isSameState(const ShardAllocationModel &other) const;

  ShardAllocationModel cloneWithNewEntries(
      llvm::DenseMap<const TensorMemoryEntry *, TensorMemoryEntry *> &entryMap)
      const;

  /// Get sharded tensors
  const llvm::SmallVector<TensorMemoryEntry *, 16> &getShardedTensors() const {
    return shardedTensors;
  }

private:
  std::size_t totalL1Size = 0;
  /// Max allowed offset for reallocation, as a percentage of total L1 size
  double offsetCap = 0.10;
  llvm::SmallVector<TensorMemoryEntry *, 16> shardedTensors;
  llvm::SmallVector<FreeBlock, 16> freeBlocks;
};

/// Analyze and determine where reallocation ops must be inserted
/// to prevent L1 fragmentation and circular buffer overlap.
class ReallocationAnalysis : public TTNNAnalysis<ReallocationAnalysisInput,
                                                 ReallocationAnalysisResult> {
public:
  ReallocationAnalysis(Operation *op) : TTNNAnalysis(op) {}

private:
  /// Check whether an op is supported by reallocation analysis.
  ///
  bool isOpSupportedForReallocate(Operation *op) const;

  /// Finish op
  void finishOp(Operation *op);

  /// Get tensor memory entry, if available in cache then return the entry
  TensorMemoryEntry &getOrCreateTensorEntry(std::size_t bufferSize,
                                            Value value);

  /// Update L1 memory state by allocation of current op
  ///
  /// \return true if allocation succeeds without overlap
  bool
  allocateOutputs(ShardAllocationModel &model,
                  const llvm::SmallVector<TensorMemoryEntry *> &outputEntries);

  /// Update L1 memory state by deallocate of current op
  ///
  /// \return true if allocation succeeds without overlap
  bool
  deallocateInputs(ShardAllocationModel &model,
                   const llvm::SmallVector<TensorMemoryEntry *> &inputEntries);

  /// Determine whether reallocation is required before this op.
  ///
  /// \param allocator        Current simulated L1 allocator state
  /// \param peakCBMemorySize Peak circular buffer memory for the op
  ///
  /// \return true if reallocation must be inserted
  bool isNeededReallocation(const ShardAllocationModel &allocator,
                            std::size_t peakCBMemorySize,
                            std::size_t additionalUsage) const;

  /// Record a reallocation insertion on a problematic edge.
  ///
  /// \param allocator Current allocator state
  ///
  /// \return true if reallocation insertion succeeds
  bool insertReallocate(ShardAllocationModel &model, Operation *currentOp,
                        llvm::ArrayRef<TensorMemoryEntry *> outputEntries);

  /// Apply pass-level overrides.
  bool applyOverrides() override;

  /// Main analysis implementation.
  void analysisImplementation() override;

  llvm::DenseSet<Operation *> doneOps;
  llvm::DenseSet<TensorMemoryEntry *> tensorCache;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_REALLOCATIONANALYSIS_H
