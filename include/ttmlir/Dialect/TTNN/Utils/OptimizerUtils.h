// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir::tt::ttnn::optimizer_utils {

/// Returns true if the op produces a ranked tensor result that the layout
/// optimizer can assign a memory layout to.
inline bool opHasTensorResult(mlir::Operation *op) {
  if (op->getNumResults() == 0) {
    return false;
  }

  if (!llvm::isa<mlir::RankedTensorType>(op->getResult(0).getType())) {
    return false;
  }

  if (llvm::isa<EmptyOp>(op)) {
    return false;
  }

  return true;
}

/// Returns true for in-place ops with no tensor result that still need input
/// layout validation and upstream reshard propagation in the beam search.
inline bool isSinkOp(mlir::Operation *op) {
  return llvm::isa<FillCacheOp, PagedFillCacheOp, PagedUpdateCacheOp>(op);
}

/// Returns true if this op should participate in the greedy beam search,
/// either because it produces a tensor output or because it is a sink that
/// drives upstream input layout decisions.
inline bool isBeamSearchTarget(mlir::Operation *op) {
  return opHasTensorResult(op) || isSinkOp(op);
}

/// Formats an MLIR SSA value into a compact string for debug logging.
std::string formatValueShort(mlir::Value value);

/// Returns the TTNN layout of an operation's first result, if present.
std::optional<TTNNLayoutAttr> extractOutputLayoutFromIR(mlir::Operation *op);

/// Returns the TTNN layout encoded on an SSA value, if present.
std::optional<TTNNLayoutAttr> extractLayoutFromValue(mlir::Value value);

/// Formats a list of TTNN layouts into a printable string.
std::string layoutsToString(const std::vector<TTNNLayoutAttr> &layouts);

/// Formats an optional TTNN layout into a printable string.
std::string layoutToString(const std::optional<TTNNLayoutAttr> &layout);

/// Formats a list of parent operations into a compact string for debug logs.
std::string parentOpsToString(
    llvm::ArrayRef<mlir::Operation *> parentOps);

/// Returns the tensor memory layout, defaulting to Interleaved when unset.
TensorMemoryLayout getLayout(TTNNLayoutAttr layout);

/// Returns true if the value represents an intermediate tensor activation.
bool isTensorActivationOperand(mlir::Value value);

/// Returns the tensor-input layout index corresponding to an exact operand.
///
/// Non-tensor operands are not counted in the returned layout index.
std::optional<unsigned>
findTensorInputLayoutIndex(mlir::Operation *op, mlir::Value value);

/// Returns the position of an operation in an operation schedule.
std::optional<size_t> getScheduledOpIndex(
    const llvm::DenseMap<
        mlir::func::FuncOp, llvm::SmallVector<mlir::Operation *>> &schedule,
    mlir::Operation *op);

// Returns unique op-specific attributes from a list of OpConfigs.
// Deduplicates by comparing OpConfig::OpSpecificAttrs values.
std::vector<mlir::tt::ttnn::OpConfig::OpSpecificAttrs>
getUniqueOpSpecificAttrs(
    const std::vector<mlir::tt::ttnn::OpConfig> &configs);

// Returns unique test configs for Matmul/Linear ops.
// Generates unique (bufferType, memLayout, opSpecificAttrs) combinations using
// ignorePhysicalLayout.
llvm::SmallVector<mlir::tt::ttnn::OpConfig>
getUniqueTestConfigsForMatmulLinear(
    const std::vector<mlir::tt::ttnn::OpConfig> &consumerConfigs);

// Returns unique test configs for validation.
// - For non-Matmul/Linear ops: Only unique op-specific attrs are needed.
// - For Matmul/Linear ops: Generate unique layout and op-specific attribute
//   combinations using ignorePhysicalLayout.
llvm::SmallVector<mlir::tt::ttnn::OpConfig> getUniqueTestConfigs(
    const std::vector<mlir::tt::ttnn::OpConfig> &consumerConfigs,
    bool isMatmulOrLinear);

} // namespace mlir::tt::ttnn::optimizer_utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H