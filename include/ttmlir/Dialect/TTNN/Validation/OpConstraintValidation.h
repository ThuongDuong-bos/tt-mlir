// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATION_H
#define TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATION_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mlir::tt::ttnn {

// Context-agnostic utility functions for systematic validation of TTNN
// operation configurations.
namespace op_constraint_validation {

enum class ValidationStatus {
  Success,
  NotImplemented,
  MetalBackendError,
  UnmatchedReferenceConfig,
  OutOfMemoryError
};

// Convert ValidationStatus to string for error messages.
llvm::StringRef validationStatusToString(ValidationStatus status);

// Result of a single constraint validation test.
struct ValidationResult {
  ValidationStatus status = ValidationStatus::Success;

  // Index in reference configs vector.
  size_t configIndex = 0;

  // What the backend actually returned. Only valid when validation succeeds.
  llvm::SmallVector<TTNNLayoutAttr> actualOutputLayouts;

  // Per-core L1 footprint of the output tensor.
  uint64_t outputL1Usage = 0;

  // Circular-buffer peak L1 usage reported by OpModel.
  uint64_t cbPeakUsage = 0;

  // Tensor-buffer peak L1 usage reported by OpModel.
  uint64_t l1BuffersPeakUsage = 0;

  // Overall peak L1 usage reported by OpModel.
  uint64_t overallPeakL1Usage = 0;

  // Error message if validation failed.
  std::string errorMessage;

  ValidationResult() = default;

  explicit ValidationResult(
      size_t configIndex,
      llvm::SmallVector<TTNNLayoutAttr> actualOutputLayouts,
      uint64_t outputL1Usage = 0, uint64_t cbPeakUsage = 0,
      uint64_t l1BuffersPeakUsage = 0, uint64_t overallPeakL1Usage = 0)
      : configIndex(configIndex),
        actualOutputLayouts(std::move(actualOutputLayouts)),
        outputL1Usage(outputL1Usage), cbPeakUsage(cbPeakUsage),
        l1BuffersPeakUsage(l1BuffersPeakUsage),
        overallPeakL1Usage(overallPeakL1Usage) {}

  // Accessors for the first actual output layout. These are convenience
  // helpers for single-output operations.
  TTNNLayoutAttr getFirstActualOutputLayout() const {
    return actualOutputLayouts.empty() ? nullptr : actualOutputLayouts[0];
  }

  TTNNLayoutAttr checkAndGetFirstActualOutputLayout() const {
    assert(!actualOutputLayouts.empty());
    return actualOutputLayouts.front();
  }

  static ValidationResult success(
      size_t configIndex, TTNNLayoutAttr actualOutputLayout,
      uint64_t outputL1Usage = 0, uint64_t cbPeakUsage = 0,
      uint64_t l1BuffersPeakUsage = 0, uint64_t overallPeakL1Usage = 0) {
    return ValidationResult(
        configIndex, llvm::SmallVector<TTNNLayoutAttr>{actualOutputLayout},
        outputL1Usage, cbPeakUsage, l1BuffersPeakUsage, overallPeakL1Usage);
  }

  static ValidationResult success(
      size_t configIndex,
      llvm::SmallVector<TTNNLayoutAttr> actualOutputLayouts,
      uint64_t outputL1Usage = 0, uint64_t cbPeakUsage = 0,
      uint64_t l1BuffersPeakUsage = 0, uint64_t overallPeakL1Usage = 0) {
    return ValidationResult(configIndex, std::move(actualOutputLayouts),
                            outputL1Usage, cbPeakUsage, l1BuffersPeakUsage,
                            overallPeakL1Usage);
  }

  static ValidationResult error(ValidationStatus status, std::string message) {
    ValidationResult result;
    result.status = status;
    result.errorMessage = std::move(message);
    return result;
  }

  static ValidationResult notImplemented(std::string message) {
    return error(ValidationStatus::NotImplemented, std::move(message));
  }

  static ValidationResult metalBackendError(std::string message) {
    return error(ValidationStatus::MetalBackendError, std::move(message));
  }

  static ValidationResult unmatchedReferenceConfig(std::string message) {
    return error(ValidationStatus::UnmatchedReferenceConfig,
                 std::move(message));
  }

  static ValidationResult outOfMemoryError(std::string message) {
    return error(ValidationStatus::OutOfMemoryError, std::move(message));
  }

  bool isSuccess() const { return status == ValidationStatus::Success; }

  bool isNotImplemented() const {
    return status == ValidationStatus::NotImplemented;
  }

  bool isError() const { return status != ValidationStatus::Success; }

  bool isMetalBackendError() const {
    return status == ValidationStatus::MetalBackendError;
  }
};

// Validate one operation configuration.
ValidationResult validateOperation(Operation *op,
                                   llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                                   const OpConfig &config,
                                   uint64_t additionalL1Usage = 0);

// Test multiple operation configurations against a set of reference configs.
std::vector<ValidationResult>
validateWithMultipleAttributes(Operation *op,
                               llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                               llvm::ArrayRef<OpConfig> opConfigs,
                               llvm::ArrayRef<OpConfig> referenceConfigs);

// Validate an OpConstraints result against the L1 memory budget derived from
// the given context operation.
ValidationResult
checkConstraintsResult(Operation *contextOp,
                       llvm::Expected<op_model::OpConstraints> constraints,
                       uint64_t additionalL1Usage = 0);

// Op-less validation. Calls OpModel<OpType>::getOpConstraints directly and
// validates the result against the L1 budget of contextOp.
template <typename OpType, typename... Args>
ValidationResult validateOperation(Operation *contextOp,
                                   uint64_t additionalL1Usage, Args &&...args) {
  auto constraints =
      op_model::OpModel<OpType>::getOpConstraints(std::forward<Args>(args)...);

  return checkConstraintsResult(contextOp, std::move(constraints),
                                additionalL1Usage);
}

} // namespace op_constraint_validation

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_VALIDATION_OPCONSTRAINTVALIDATION_H
