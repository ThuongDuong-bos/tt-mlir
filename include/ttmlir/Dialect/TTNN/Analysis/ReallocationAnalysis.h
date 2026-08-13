// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_REALLOCATIONANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_REALLOCATIONANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <string>
#include <utility>

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// Reallocation analysis input / output
//===----------------------------------------------------------------------===//

struct ReallocationAnalysisInput {
  llvm::DenseMap<Operation *, OpConfig> opConfigMap;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> schedule;

  llvm::DenseMap<Operation *, llvm::DenseSet<Value>> inputValueMap;
  llvm::DenseMap<Operation *, llvm::DenseSet<Value>> outputValueMap;

  uint64_t usableL1CacheSize = 0;
  double offsetCap = 0.10;

  ReallocationAnalysisInput() = default;

  ReallocationAnalysisInput(
      llvm::DenseMap<Operation *, OpConfig> opConfigMap,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> schedule,
      uint64_t usableL1CacheSize, double offsetCap)
      : opConfigMap(std::move(opConfigMap)), schedule(std::move(schedule)),
        usableL1CacheSize(usableL1CacheSize), offsetCap(offsetCap) {
    for (const auto &[_, operations] : this->schedule) {
      for (Operation *op : operations) {
        extractInputValues(op);
        extractOutputValues(op);
      }
    }
  }

  llvm::DenseSet<Value> getInputValues(Operation *op) const {
    auto it = inputValueMap.find(op);
    return it == inputValueMap.end() ? llvm::DenseSet<Value>{} : it->second;
  }

  llvm::DenseSet<Value> getOutputValues(Operation *op) const {
    auto it = outputValueMap.find(op);
    return it == outputValueMap.end() ? llvm::DenseSet<Value>{} : it->second;
  }

  bool operator==(const ReallocationAnalysisInput &rhs) const {
    return opConfigMap == rhs.opConfigMap && schedule == rhs.schedule &&
           usableL1CacheSize == rhs.usableL1CacheSize &&
           offsetCap == rhs.offsetCap;
  }

  bool operator!=(const ReallocationAnalysisInput &rhs) const {
    return !(*this == rhs);
  }

private:
  void extractInputValues(Operation *op) {
    llvm::DenseSet<Value> values;
    for (Value operand : op->getOperands()) {
      if (mlir::isa<RankedTensorType>(operand.getType())) {
        values.insert(operand);
      }
    }
    inputValueMap[op] = std::move(values);
  }

  void extractOutputValues(Operation *op) {
    llvm::DenseSet<Value> values;
    for (Value result : op->getResults()) {
      if (mlir::isa<RankedTensorType>(result.getType())) {
        values.insert(result);
      }
    }
    outputValueMap[op] = std::move(values);
  }
};

/// Exact SSA values that should be materialized through ttnn.reallocate.
/// The map key is the operation whose allocation pressure triggered the
/// decision; the values retain exact multi-result identity.
struct ReallocationAnalysisResult {
  llvm::DenseMap<Operation *, llvm::DenseSet<Value>> memReallocateValuesMap;

  bool operator==(const ReallocationAnalysisResult &rhs) const {
    return memReallocateValuesMap == rhs.memReallocateValuesMap;
  }

  bool operator!=(const ReallocationAnalysisResult &rhs) const {
    return !(*this == rhs);
  }
};

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const ReallocationAnalysisResult &result) {
  os << "ReallocationAnalysisResult {\n";
  if (result.memReallocateValuesMap.empty()) {
    os << "  <empty>\n";
  }

  for (const auto &[triggerOp, values] : result.memReallocateValuesMap) {
    os << "  Trigger: ";
    if (triggerOp) {
      triggerOp->print(os);
    } else {
      os << "<null>";
    }
    os << "\n";

    for (Value value : values) {
      os << "    - ";
      value.printAsOperand(os, mlir::OpPrintingFlags{});
      os << "\n";
    }
  }

  os << "}";
  return os;
}

//===----------------------------------------------------------------------===//
// Tensor memory model
//===----------------------------------------------------------------------===//

struct FreeBlock {
  size_t startAddress = 0;
  size_t endAddress = 0;
};

class TensorMemoryEntry {
public:
  TensorMemoryEntry(size_t bufferSize, Value value)
      : bufferSize(bufferSize), value(value) {}

  bool isAllocated() const {
    return logicalAddress != static_cast<size_t>(-1);
  }

  void allocate(size_t address) {
    assert(!isAllocated());
    logicalAddress = address;
  }

  void deallocate() {
    assert(isAllocated());
    logicalAddress = static_cast<size_t>(-1);
  }

  size_t getLogicalAddress() const { return logicalAddress; }
  size_t start() const { return logicalAddress; }
  size_t end() const { return logicalAddress + bufferSize; }
  size_t getBufferSize() const { return bufferSize; }
  Value getValue() const { return value; }

  bool isL1Sharded() const {
    auto tensorType = mlir::dyn_cast_if_present<RankedTensorType>(
        value ? value.getType() : Type());
    if (!tensorType) {
      return false;
    }

    auto layout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    return layout && layout.hasShardedL1TensorMemoryLayout();
  }

  bool equalsByValue(Value rhs) const { return value == rhs; }

private:
  size_t logicalAddress = static_cast<size_t>(-1);
  size_t bufferSize = 0;
  Value value;
};

class ShardAllocationModel {
public:
  explicit ShardAllocationModel(size_t totalL1Size)
      : totalL1Size(totalL1Size) {
    freeBlocks.push_back({0, totalL1Size});
  }

  ShardAllocationModel(size_t totalL1Size, double offsetCap)
      : totalL1Size(totalL1Size), offsetCap(offsetCap) {
    freeBlocks.push_back({0, totalL1Size});
  }

  bool allocate(TensorMemoryEntry &tensor);
  bool deallocate(TensorMemoryEntry &tensor);
  bool reallocate(TensorMemoryEntry &tensor);

  size_t getLowestAllocatedAddress() const;
  size_t getHighestCBAddress(size_t peakCBMemorySize) const;

  bool isAllocated(const TensorMemoryEntry &tensor) const;
  bool isSameState(const ShardAllocationModel &other) const;

  ShardAllocationModel cloneWithNewEntries(
      llvm::DenseMap<const TensorMemoryEntry *, TensorMemoryEntry *> &entryMap)
      const;

  llvm::SmallVector<TensorMemoryEntry *, 16> getShardedTensors() const {
    return shardedTensors;
  }

private:
  size_t totalL1Size = 0;
  double offsetCap = 0.10;
  llvm::SmallVector<TensorMemoryEntry *, 16> shardedTensors;
  llvm::SmallVector<FreeBlock, 16> freeBlocks;
};

//===----------------------------------------------------------------------===//
// Reallocation analysis
//===----------------------------------------------------------------------===//

class ReallocationAnalysis
    : public TTNNAnalysis<ReallocationAnalysisInput,
                          ReallocationAnalysisResult> {
public:
  explicit ReallocationAnalysis(Operation *op) : TTNNAnalysis(op) {}

  ~ReallocationAnalysis() {
    for (TensorMemoryEntry *entry : tensorCache) {
      delete entry;
    }
  }

private:
  void analysisImplementation() override;
  bool applyOverrides() override;

  bool isOpSupportedForReallocate(Operation *op) const;
  void finishOp(Operation *op);

  TensorMemoryEntry &getOrCreateTensorEntry(size_t bufferSize, Value value);

  bool allocateOutputs(
      ShardAllocationModel &model,
      const llvm::SmallVector<TensorMemoryEntry *> &outputEntries);

  bool deallocateInputs(
      ShardAllocationModel &model,
      const llvm::SmallVector<TensorMemoryEntry *> &inputEntries);

  bool isNeededReallocation(const ShardAllocationModel &allocator,
                            size_t peakCBMemorySize,
                            size_t additionalUsage) const;

  bool insertReallocate(
      ShardAllocationModel &model, Operation *currentOp,
      llvm::ArrayRef<TensorMemoryEntry *> outputEntries);

  llvm::DenseSet<Operation *> doneOps;
  llvm::DenseSet<TensorMemoryEntry *> tensorCache;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_REALLOCATIONANALYSIS_H
