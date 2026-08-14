// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_COSTMODEL_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_COSTMODEL_H

#include "ttmlir/Dialect/TTNN/Analysis/OpCandidatesBuilder.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn {

struct EmissionCostParams {
  double wCore = 0.0167;
  double wRisk = 10.0;
  double wCb = 25.0;
  double wBuf = 1.0;
  double wOut = 1.0;
  double wShard = 15.0;
  double wSpill = 50.0;

  double tauSoft = 0.85;
  double tauHard = 1.0;
};

struct TransitionCostParams {
  double wToLayout = 1.0;
  double wReshape = 1.0;
  double wPermute = 1.0;
  double wPad = 1.0;
  double wTypecast = 1.0;

  double wSwitch = 1.0;
  double wSwitchDramToL1 = 7.0;
  double wSwitchL1ToDram = 10.0;
  double wSwitchL1InterleavedSharded = 5.0;
  double wSwitchL1ShardedKind = 5.0;
};

class CostModel {
public:
  using CandidateOutputSizesMap =
      llvm::DenseMap<mlir::Operation *,
                     llvm::SmallVector<std::optional<uint64_t>>>;
  using CandidateSizeList = llvm::SmallVector<uint64_t>;
  using PassiveTensorList = llvm::SmallVector<CandidateSizeList>;

  struct LocalCostResult {
    double cost = 0.0;
    bool skipGroup = false;
    std::optional<uint64_t> outputSizeBytes;
    uint64_t additionalL1Usage = 0;
    std::size_t selectedSpillCount = 0;
  };

  struct TransitionEdgeCostInput {
    mlir::Operation *producerOp = nullptr;
    mlir::Operation *consumerOp = nullptr;
    mlir::Value consumerOperand;
    const OpConfigCandidate *producerCandidate = nullptr;
    const OpConfigCandidate *consumerCandidate = nullptr;
    const llvm::DenseMap<mlir::Operation *, OpConfig> *opConfigMap = nullptr;
    uint64_t additionalL1Usage = 0;
  };

  CostModel(EmissionCostParams emissionParams = EmissionCostParams(),
            TransitionCostParams transitionParams = TransitionCostParams());

  virtual ~CostModel() = default;

  virtual LocalCostResult
  getLocalCost(mlir::Operation *op, const OpConfigCandidate &candidate,
               llvm::ArrayRef<mlir::Operation *> passiveTensorProducers,
               bool shouldComputeOutputSize,
               const CandidateOutputSizesMap &storedCandidateOutputSizes) const;

  virtual double getTransitionCost(const TransitionEdgeCostInput &input) const;

  virtual std::optional<uint64_t> computeOutputSize(
      std::optional<TTNNLayoutAttr> outputLayout,
      const op_constraint_validation::ValidationResult &validationResult) const;

protected:
  virtual double getEmissionCost(
      mlir::Operation *op, const OpConfigCandidate &candidate,
      const op_constraint_validation::ValidationResult &validationResult,
      uint64_t additionalL1Usage = 0) const;

  virtual double calculateEmission(
      mlir::Operation *op, const OpConfigCandidate &candidate,
      const op_constraint_validation::ValidationResult &validationResult,
      uint64_t additionalL1Usage = 0) const;

  virtual double
  calculateTransition(const TransitionEdgeCostInput &input) const;

  double computeSpillCost(uint64_t selectedSpillBytes,
                          std::optional<TTNNLayoutAttr> outputLayout,
                          mlir::Operation *op) const;

  uint64_t getTensorTotalSize(uint64_t outputTensorUsagePerCore,
                              TTNNLayoutAttr outputLayout) const;

  double getLayoutCores(TTNNLayoutAttr layout) const;

  PassiveTensorList collectPassiveTensor(
      llvm::ArrayRef<mlir::Operation *> passiveTensorProducers,
      const CandidateOutputSizesMap &storedCandidateOutputSizes) const;

  op_constraint_validation::ValidationResult
  validateOpConfig(mlir::Operation *op, const OpConfigCandidate &candidate,
                   const PassiveTensorList &passiveTensorList,
                   uint64_t &selectedAdditionalL1Usage,
                   std::size_t &selectedSpillCount) const;

  EmissionCostParams emissionCostParams;
  TransitionCostParams transitionCostParams;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_COSTMODEL_H
