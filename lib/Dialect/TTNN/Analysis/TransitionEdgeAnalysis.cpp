// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/TransitionEdgeAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>
#include <utility>

namespace mlir::tt::ttnn::analysis {

namespace {

//===----------------------------------------------------------------------===//
// Transition helpers
//===----------------------------------------------------------------------===//

static TransitionOpIndex getTransitionOpIndex(Operation *op) {
  if (!op) {
    return TransitionOpIndex::Unknown;
  }

  if (isa<ToLayoutOp>(op)) {
    return TransitionOpIndex::ToLayout;
  }

  if (isa<TypecastOp>(op)) {
    return TransitionOpIndex::Typecast;
  }

  if (isa<ReshapeOp>(op)) {
    return TransitionOpIndex::Reshape;
  }

  if (isa<PermuteOp>(op)) {
    return TransitionOpIndex::Permute;
  }

  if (isa<PadOp>(op)) {
    return TransitionOpIndex::Pad;
  }

  return TransitionOpIndex::Unknown;
}

static bool isConversionOp(Operation *op) {
  if (!op) {
    return false;
  }

  return isa<ToLayoutOp, ToMemoryConfigOp, ReshapeOp, PermuteOp, TransposeOp,
             PadOp, TypecastOp>(op);
}

// Current uplift only explicitly supports Q/K/V-style multi-result producers.
// Keep this local until there is a shared non-BOS utility for the same rule.
static bool isQKVHeadsOp(Operation *op) {
  if (!op) {
    return false;
  }

  const StringRef opName = op->getName().getStringRef();

  return opName == "ttnn.split_query_key_value_and_split_heads" ||
         opName == "ttnn.nlp_create_qkv_heads";
}

static std::pair<TTNNLayoutAttr, TTNNLayoutAttr>
getLayoutsFromOp(Operation *op) {
  if (!op || op->getNumOperands() == 0 || op->getNumResults() == 0) {
    return {nullptr, nullptr};
  }

  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());

  if (!inputType || !outputType) {
    return {nullptr, nullptr};
  }

  auto inputLayout = dyn_cast_or_null<TTNNLayoutAttr>(inputType.getEncoding());
  auto outputLayout =
      dyn_cast_or_null<TTNNLayoutAttr>(outputType.getEncoding());

  return {inputLayout, outputLayout};
}

struct TransitionOpsInfo {
  llvm::SmallVector<std::pair<TransitionOpInfo, Operation *>> ops;
  bool reachedProducer = false;
};

// Collect conversion operations that already exist between consumerOperand and
// the exact producer op.
static TransitionOpsInfo collectTransitionOps(Value consumerOperand,
                                              Operation *producerOp) {
  TransitionOpsInfo info;

  Value currentValue = consumerOperand;
  Operation *currentOp = currentValue.getDefiningOp();

  while (currentOp) {
    if (currentOp == producerOp) {
      info.reachedProducer = true;
      break;
    }

    if (isConversionOp(currentOp)) {
      auto [inputLayout, outputLayout] = getLayoutsFromOp(currentOp);

      if (inputLayout && outputLayout) {
        const TransitionOpIndex opIndex = getTransitionOpIndex(currentOp);

        if (opIndex != TransitionOpIndex::Unknown) {
          TransitionOpInfo opInfo{
              currentOp->getName().getStringRef(),
              inputLayout,
              outputLayout,
              opIndex,
          };

          info.ops.push_back({opInfo, currentOp});
        }
      }
    }

    if (currentOp->getNumOperands() != 1) {
      break;
    }

    currentValue = currentOp->getOperand(0);
    currentOp = currentValue.getDefiningOp();
  }

  // A null producer represents a block argument / graph input.
  if (!producerOp && !currentOp) {
    info.reachedProducer = true;
  }

  if (info.reachedProducer) {
    std::reverse(info.ops.begin(), info.ops.end());
  }

  return info;
}

static llvm::SmallVector<TransitionOpInfo>
constructTransitionOps(TTNNLayoutAttr producerOutputLayout,
                       TTNNLayoutAttr consumerInputLayout) {
  if (!producerOutputLayout || !consumerInputLayout ||
      producerOutputLayout == consumerInputLayout) {
    return {};
  }

  return {{
      "ToLayoutOp",
      producerOutputLayout,
      consumerInputLayout,
      TransitionOpIndex::ToLayout,
  }};
}

// Validate only transition ops that are newly planned. Existing conversion ops
// are already present in the module.
static op_constraint_validation::ValidationResult
validateTransitionOp(const TransitionOpInfo &transitionOpInfo, Value inputValue,
                     Operation *contextOp, uint64_t additionalL1Usage) {
  auto validationResult =
      op_constraint_validation::ValidationResult::notImplemented(
          "Validation for transition op is not supported");

  switch (transitionOpInfo.opIndex) {
  case TransitionOpIndex::ToLayout: {
    auto inputType = dyn_cast<RankedTensorType>(inputValue.getType());

    if (!inputType) {
      return validationResult;
    }

    const TTNNLayoutAttr inputLayout = transitionOpInfo.inputLayout;
    const TTNNLayoutAttr outputLayout = transitionOpInfo.outputLayout;

    if (!inputLayout || !outputLayout) {
      return validationResult;
    }

    std::optional<ttcore::DataType> outputDtype =
        outputLayout.getLayout() != Layout::RowMajor
            ? std::optional<ttcore::DataType>(outputLayout.getDataType())
            : std::nullopt;

    // Current branch OpModel<ToLayoutOp>::getOpConstraints takes:
    //   inputShape, inputLayout, outputDtype, outputLayout.
    //
    // Do NOT pass deviceGrid here.
    validationResult = op_constraint_validation::validateOperation<ToLayoutOp>(
        contextOp, additionalL1Usage, inputType.getShape(), inputLayout,
        outputDtype, outputLayout);

    break;
  }

  case TransitionOpIndex::Typecast:
    // Planned Typecast validation is not implemented yet.
    break;

  case TransitionOpIndex::Reshape:
  case TransitionOpIndex::Permute:
  case TransitionOpIndex::Pad:
  case TransitionOpIndex::Unknown:
    // These operations are only expected as existing conversion operations in
    // the current implementation.
    break;
  }

  return validationResult;
}

} // namespace

//===----------------------------------------------------------------------===//
// Main analysis
//===----------------------------------------------------------------------===//

LogicalResult TransitionEdgeAnalysis::run() {
  transitionEdges.clear();
  emittedUses.clear();
  valueLayoutMap.clear();

  resolveOpLayouts();
  emitEdges();

  return validateTransitions();
}

void TransitionEdgeAnalysis::resolveOpLayouts() {
  for (const auto &[funcRef, operations] : opSchedule) {
    func::FuncOp func = funcRef;

    if (!func.empty()) {
      Block &entryBlock = func.getBody().front();

      for (Value arg : entryBlock.getArguments()) {
        auto tensorType = dyn_cast<RankedTensorType>(arg.getType());

        if (!tensorType) {
          continue;
        }

        if (auto layout =
                dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding())) {
          valueLayoutMap[arg] = layout;
        }
      }
    }

    for (Operation *op : operations) {
      if (!op || op->getNumResults() == 0) {
        continue;
      }

      auto configIt = opConfigMap.find(op);

      // Policy-selected compute operation.
      if (configIt != opConfigMap.end()) {
        const OpConfig &config = configIt->second;

        if (op->getNumResults() > 1 && !isQKVHeadsOp(op)) {
          TTMLIR_FATAL(
              ttmlir::LogComponent::ViterbiOptimizer,
              "Only Q/K/V multi-result producer ops are currently supported "
              "in TransitionEdgeAnalysis: {}",
              op->getName().getStringRef());
        }

        for (OpResult result : op->getResults()) {
          TTNNLayoutAttr layout = config.outputLayout;

          // Preserve exact SSA result identity. The current OpConfig still
          // stores a single selected output layout, so supported Q/K/V
          // multi-result ops initially share that selected layout.
          if (!layout) {
            if (auto tensorType =
                    dyn_cast<RankedTensorType>(result.getType())) {
              layout =
                  dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
            }
          }

          if (layout) {
            valueLayoutMap[result] = layout;
          }
        }

        continue;
      }

      // Non-policy operation: prefer the result type encoding.
      for (OpResult result : op->getResults()) {
        TTNNLayoutAttr resolvedLayout = nullptr;

        if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
          resolvedLayout =
              dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
        }

        // Safe fallback for shape/layout-preserving single-result operations.
        if (!resolvedLayout && op->getNumOperands() > 0) {
          if (op->getNumResults() == 1 &&
              op->getOperand(0).getType() == result.getType()) {
            Value input = op->getOperand(0);

            if (auto layoutIt = valueLayoutMap.find(input);
                layoutIt != valueLayoutMap.end()) {
              resolvedLayout = layoutIt->second;
            }
          }
        }

        if (resolvedLayout) {
          valueLayoutMap[result] = resolvedLayout;
        }
      }
    }
  }
}

void TransitionEdgeAnalysis::emitEdges() {
  for (const auto &[_, operations] : opSchedule) {
    for (Operation *consumerOp : operations) {
      if (!consumerOp) {
        continue;
      }

      for (unsigned operandIndex = 0;
           operandIndex < consumerOp->getNumOperands(); ++operandIndex) {
        OpOperand &consumerUse = consumerOp->getOpOperand(operandIndex);

        Value consumerOperand = consumerUse.get();

        if (!isa<RankedTensorType>(consumerOperand.getType())) {
          continue;
        }

        auto resolved = resolveProducerAndLayout(consumerOperand);

        if (!resolved) {
          TTMLIR_DEBUG(
              ttmlir::LogComponent::ViterbiOptimizer,
              "TransitionEdgeAnalysis: failed to resolve producer/layout for "
              "consumer op {} operand {}",
              consumerOp->getName().getStringRef(), operandIndex);
          continue;
        }

        Value producerValue = resolved->first;
        TTNNLayoutAttr producerLayout = resolved->second;

        std::optional<TTNNLayoutAttr> requiredLayout =
            getRequiredLayout(consumerOp, operandIndex);

        if (!requiredLayout || !*requiredLayout) {
          continue;
        }

        TTNNLayoutAttr consumerLayout = *requiredLayout;

        if (producerLayout == consumerLayout) {
          continue;
        }

        // Deduplicate by exact SSA use, not by producer operation.
        if (!emittedUses.insert(&consumerUse).second) {
          continue;
        }

        Operation *producerOp = producerValue.getDefiningOp();

        GetTransitionOpsResult transitionResult =
            getTransitionOps(producerOp, consumerOperand, consumerOp,
                             producerLayout, consumerLayout,
                             /*additionalL1Usage=*/0);

        if (!transitionResult.isSuccess) {
          consumerOp->emitError()
              << "Unable to construct a valid layout transition for operand "
              << operandIndex;
          continue;
        }

        transitionEdges.push_back(TransitionEdge{
            producerValue,
            consumerOp,
            operandIndex,
            producerLayout,
            consumerLayout,
            std::move(transitionResult.transitionOps),
        });
      }
    }
  }
}

LogicalResult TransitionEdgeAnalysis::validateTransitions() {
  for (const TransitionEdge &transitionEdge : transitionEdges) {
    if (!transitionEdge.consumerOp || !transitionEdge.producerLayout ||
        !transitionEdge.consumerLayout) {
      if (transitionEdge.consumerOp) {
        transitionEdge.consumerOp->emitError()
            << "Invalid transition edge: missing producer or consumer layout";
      }

      return failure();
    }

    if (transitionEdge.producerLayout == transitionEdge.consumerLayout) {
      transitionEdge.consumerOp->emitError()
          << "Invalid transition edge: identical producer and consumer layouts";

      return failure();
    }

    if (transitionEdge.transitionOps.empty()) {
      transitionEdge.consumerOp->emitError()
          << "Invalid transition edge: layout mismatch has no transition ops";

      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Layout resolution
//===----------------------------------------------------------------------===//

std::optional<TTNNLayoutAttr>
TransitionEdgeAnalysis::getRequiredLayout(Operation *op,
                                          unsigned operandIndex) const {
  auto inputLayoutsIt = inputLayoutsMap.find(op);

  if (inputLayoutsIt == inputLayoutsMap.end()) {
    return std::nullopt;
  }

  // inputLayoutsMap contains tensor operands only. Translate the MLIR operand
  // index into its tensor-input-layout index.
  unsigned tensorInputIndex = 0;

  for (unsigned currentOperandIndex = 0;
       currentOperandIndex < op->getNumOperands(); ++currentOperandIndex) {
    Value operand = op->getOperand(currentOperandIndex);

    if (!isa<RankedTensorType>(operand.getType())) {
      continue;
    }

    if (currentOperandIndex == operandIndex) {
      if (tensorInputIndex >= inputLayoutsIt->second.size()) {
        return std::nullopt;
      }

      TTNNLayoutAttr layout = inputLayoutsIt->second[tensorInputIndex];

      return layout ? std::optional<TTNNLayoutAttr>(layout) : std::nullopt;
    }

    ++tensorInputIndex;
  }

  return std::nullopt;
}

std::optional<std::pair<Value, TTNNLayoutAttr>>
TransitionEdgeAnalysis::resolveProducerAndLayout(Value value) {
  llvm::SmallPtrSet<Value, 8> visited;

  auto getLayoutFromValue = [&](Value currentValue) -> TTNNLayoutAttr {
    if (auto layoutIt = valueLayoutMap.find(currentValue);
        layoutIt != valueLayoutMap.end()) {
      return layoutIt->second;
    }

    auto tensorType = dyn_cast<RankedTensorType>(currentValue.getType());

    if (!tensorType) {
      return nullptr;
    }

    auto layout = dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());

    if (layout) {
      valueLayoutMap[currentValue] = layout;
    }

    return layout;
  };

  TTNNLayoutAttr actualLayout = getLayoutFromValue(value);

  if (!actualLayout) {
    return std::nullopt;
  }

  Value currentValue = value;

  while (true) {
    if (!visited.insert(currentValue).second) {
      return std::nullopt;
    }

    if (!isa<RankedTensorType>(currentValue.getType())) {
      return std::nullopt;
    }

    Operation *defOp = currentValue.getDefiningOp();

    // Function argument.
    if (!defOp) {
      return std::make_pair(currentValue, actualLayout);
    }

    // Stop at policy-selected producer while preserving exact result Value.
    if (opConfigMap.contains(defOp)) {
      return std::make_pair(currentValue, actualLayout);
    }

    // Existing conversion operations are traversed, but the layout presented
    // to the consumer remains the encoding of the original consumer operand.
    if (isConversionOp(defOp)) {
      if (defOp->getNumOperands() != 1) {
        return std::nullopt;
      }

      currentValue = defOp->getOperand(0);
      continue;
    }

    // Pass through simple one-input non-policy operations.
    if (defOp->getNumOperands() != 1) {
      return std::nullopt;
    }

    Value nextValue = defOp->getOperand(0);

    if (!isa<RankedTensorType>(nextValue.getType())) {
      return std::nullopt;
    }

    currentValue = nextValue;
  }
}

//===----------------------------------------------------------------------===//
// Transition planning
//===----------------------------------------------------------------------===//

GetTransitionOpsResult TransitionEdgeAnalysis::getTransitionOps(
    Operation *producerOp, Value consumerOperand, Operation *consumerOp,
    TTNNLayoutAttr producerOutputLayout, TTNNLayoutAttr consumerInputLayout,
    uint64_t additionalL1Usage) const {
  if (!consumerOperand || !consumerOp || !producerOutputLayout ||
      !consumerInputLayout) {
    return {false, {}};
  }

  // Q/K/V operations can expose result-specific layouts through the exact SSA
  // result type. Prefer that encoding when available.
  if (producerOp && producerOp->getNumResults() > 1 &&
      isQKVHeadsOp(producerOp)) {
    Value currentValue = consumerOperand;

    while (Operation *defOp = currentValue.getDefiningOp()) {
      if (defOp == producerOp) {
        if (auto tensorType =
                dyn_cast<RankedTensorType>(currentValue.getType())) {
          if (auto exactLayout =
                  dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding())) {
            producerOutputLayout = exactLayout;
          }
        }

        break;
      }

      if (defOp->getNumOperands() != 1) {
        break;
      }

      currentValue = defOp->getOperand(0);
    }
  }

  TransitionOpsInfo existingTransOps =
      collectTransitionOps(consumerOperand, producerOp);

  if (!existingTransOps.reachedProducer) {
    TTMLIR_DEBUG(
        ttmlir::LogComponent::ViterbiOptimizer,
        "Transition planning failed: operand does not trace to producer. "
        "consumer={} producer={}",
        consumerOp->getName().getStringRef(),
        producerOp ? producerOp->getName().getStringRef()
                   : StringRef("<block-arg>"));

    return {false, {}};
  }

  GetTransitionOpsResult result;

  for (const auto &[opInfo, _] : existingTransOps.ops) {
    result.transitionOps.push_back(opInfo);
  }

  TTNNLayoutAttr finalProducerOutputLayout =
      existingTransOps.ops.empty()
          ? producerOutputLayout
          : existingTransOps.ops.back().first.outputLayout;

  llvm::SmallVector<TransitionOpInfo> plannedTransitionOps =
      constructTransitionOps(finalProducerOutputLayout, consumerInputLayout);

  result.transitionOps.append(plannedTransitionOps.begin(),
                              plannedTransitionOps.end());

  // Existing conversion operations already exist in the module and are not
  // revalidated here. Only newly planned transition operations are validated.
  const size_t existingOpsCount = existingTransOps.ops.size();

  for (size_t transitionIndex = existingOpsCount;
       transitionIndex < result.transitionOps.size(); ++transitionIndex) {
    const TransitionOpInfo &transitionOpInfo =
        result.transitionOps[transitionIndex];

    op_constraint_validation::ValidationResult validationResult =
        validateTransitionOp(transitionOpInfo, consumerOperand, consumerOp,
                             additionalL1Usage);

    if (!validationResult.isSuccess()) {
      TTMLIR_DEBUG(
          ttmlir::LogComponent::ViterbiOptimizer,
          "Transition validation failed: op={} status={} error={} "
          "inputLayout={} outputLayout={}",
          transitionOpInfo.opName,
          op_constraint_validation::validationStatusToString(
              validationResult.status),
          validationResult.errorMessage.empty() ? std::string("<none>")
                                                : validationResult.errorMessage,
          optimizer_utils::layoutToString(transitionOpInfo.inputLayout),
          optimizer_utils::layoutToString(transitionOpInfo.outputLayout));

      return {false, {}};
    }
  }

  // Multi-result Q/K/V producer:
  //
  // The exact result edge may resolve to a layout different from the
  // consumer's original candidate input layout. Revalidate the consumer using
  // the final resolved layout on exactly that tensor operand.
  if (producerOp && producerOp->getNumResults() > 1) {
    if (!isQKVHeadsOp(producerOp)) {
      TTMLIR_FATAL(
          ttmlir::LogComponent::ViterbiOptimizer,
          "Only Q/K/V multi-result producer ops are currently supported in "
          "TransitionEdgeAnalysis: {}",
          producerOp->getName().getStringRef());
    }

    auto configIt = opConfigMap.find(consumerOp);
    auto layoutsIt = inputLayoutsMap.find(consumerOp);

    if (configIt == opConfigMap.end() || layoutsIt == inputLayoutsMap.end()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                   "Transition consumer validation missing state for op {}",
                   consumerOp->getName().getStringRef());

      return {false, {}};
    }

    TTNNLayoutAttr finalLayout = result.transitionOps.empty()
                                     ? producerOutputLayout
                                     : result.transitionOps.back().outputLayout;

    llvm::SmallVector<TTNNLayoutAttr> inputLayouts = layoutsIt->second;

    unsigned tensorInputLayoutIndex = 0;
    bool updatedEdgeInputLayout = false;

    for (unsigned operandIndex = 0; operandIndex < consumerOp->getNumOperands();
         ++operandIndex) {
      Value operand = consumerOp->getOperand(operandIndex);

      if (!isa<RankedTensorType>(operand.getType())) {
        continue;
      }

      if (operand == consumerOperand) {
        if (tensorInputLayoutIndex >= inputLayouts.size()) {
          return {false, {}};
        }

        inputLayouts[tensorInputLayoutIndex] = finalLayout;

        updatedEdgeInputLayout = true;
      }

      ++tensorInputLayoutIndex;
    }

    if (!updatedEdgeInputLayout) {
      return {false, {}};
    }

    op_constraint_validation::ValidationResult consumerValidationResult =
        op_constraint_validation::validateOperation(
            consumerOp, inputLayouts, configIt->second, additionalL1Usage);

    if (!consumerValidationResult.isSuccess()) {
      TTMLIR_DEBUG(
          ttmlir::LogComponent::ViterbiOptimizer,
          "Transition consumer validation failed: op={} status={} error={}",
          consumerOp->getName().getStringRef(),
          op_constraint_validation::validationStatusToString(
              consumerValidationResult.status),
          consumerValidationResult.errorMessage.empty()
              ? std::string("<none>")
              : consumerValidationResult.errorMessage);

      return {false, {}};
    }
  }

  result.isSuccess = true;
  return result;
}

} // namespace mlir::tt::ttnn::analysis