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

  bool allocate(TensorMemoryEntry &tensor);
  bool deallocate(TensorMemoryEntry &tensor);
  bool reallocate(TensorMemoryEntry &tensor);

  std::size_t getLowestAllocatedAddress() const;
  std::size_t getHighestCBAddress(std::size_t peakCBMemorySize) const;

  bool isAllocated(const TensorMemoryEntry &tensor) const;
  bool isSameState(const ShardAllocationModel &other) const;

  ShardAllocationModel cloneWithNewEntries(
      llvm::DenseMap<const TensorMemoryEntry *, TensorMemoryEntry *> &entryMap)
      const;

  const llvm::SmallVector<TensorMemoryEntry *, 16> &getShardedTensors() const {
    return shardedTensors;
  }

private:
  std::size_t totalL1Size = 0;
  double offsetCap = 0.10;
  llvm::SmallVector<TensorMemoryEntry *, 16> shardedTensors;
  llvm::SmallVector<FreeBlock, 16> freeBlocks;
};

class ReallocationAnalysis : public TTNNAnalysis<ReallocationAnalysisInput,
                                                 ReallocationAnalysisResult> {
public:
  ReallocationAnalysis(Operation *op) : TTNNAnalysis(op) {}

private:
  bool isOpSupportedForReallocate(Operation *op) const;
  void finishOp(Operation *op);

  TensorMemoryEntry &getOrCreateTensorEntry(std::size_t bufferSize,
                                            Value value);

  bool
  allocateOutputs(ShardAllocationModel &model,
                  const llvm::SmallVector<TensorMemoryEntry *> &outputEntries);

  bool
  deallocateInputs(ShardAllocationModel &model,
                   const llvm::SmallVector<TensorMemoryEntry *> &inputEntries);

  bool isNeededReallocation(const ShardAllocationModel &allocator,
                            std::size_t peakCBMemorySize,
                            std::size_t additionalUsage) const;

  bool insertReallocate(ShardAllocationModel &model, Operation *currentOp,
                        llvm::ArrayRef<TensorMemoryEntry *> outputEntries);

  bool applyOverrides() override;
  void analysisImplementation() override;

  llvm::DenseSet<Operation *> doneOps;
  llvm::DenseSet<TensorMemoryEntry *> tensorCache;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_REALLOCATIONANALYSIS_H
