// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/TransitionEdgeAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace mlir::tt::ttnn::analysis {

LogicalResult TransitionEdgeAnalysis::run() {
  // Reset internal state for re-runs
  transitionEdges.clear();
  emittedUses.clear();
  valueLayoutMap.clear();

  resolveOpLayouts();

  emitEdges();

  if (failed(validateTransitions())) {
    return failure();
  }

  return success();
}

void TransitionEdgeAnalysis::resolveOpLayouts() {
  // Assign a concrete TTNNLayoutAttr to every Value in the graph
  // and store it in valueLayoutMap.

  for (auto &[funcRef, ops] : opSchedule) {
    func::FuncOp func = funcRef;

    // Handle block arguments (function inputs)
    // These Values have no defining op.
    //
    if (!func.empty()) {
      Block &entryBlock = func.getBody().front();

      for (Value arg : entryBlock.getArguments()) {
        if (auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType())) {
          if (auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
                  tensorType.getEncoding())) {
            valueLayoutMap[arg] = layout;
          }
        }
      }
    }

    for (Operation *op : ops) {

      if (op->getNumResults() == 0) {
        continue;
      }

      if (!opConfigMap.count(op)) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "Missing OpConfig for op: {}",
                     op->getName().getStringRef());
      }

      // Assign layout for compute op (has policy-selected config)
      if (auto it = opConfigMap.find(op); it != opConfigMap.end()) {
        const OpConfig &config = it->second;

        for (OpResult result : op->getResults()) {
          TTNNLayoutAttr layout = config.outputLayout;

          assert(op->getNumResults() == 1 &&
                 "Current implementation assumes single outputLayout per op");

          // Fallback: extract from result type if config is missing
          if (!layout) {
            if (auto tensorType =
                    mlir::dyn_cast<RankedTensorType>(result.getType())) {
              layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
                  tensorType.getEncoding());
            }
          }

          if (layout) {
            valueLayoutMap[result] = layout;
          }
          if (!config.outputLayout) {
            TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                         "Missing outputLayout in config for op: {}",
                         op->getName().getStringRef());
          }
        }
        continue;
      }

      // Assign layout for non-compute op (no config)
      for (OpResult result : op->getResults()) {
        TTNNLayoutAttr resolvedLayout = nullptr;

        if (auto tensorType =
                mlir::dyn_cast<RankedTensorType>(result.getType())) {
          resolvedLayout =
              mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
        }

        // Propagate from first operand (fallback)
        if (!resolvedLayout && op->getNumOperands() > 0) {
          assert(op->getOperand(0).getType() == result.getType() &&
                 op->getNumResults() == 1 &&
                 "Unsafe layout propagation: only allowed for single-result "
                 "ops with identical types");

          Value input = op->getOperand(0);
          auto it = valueLayoutMap.find(input);
          if (it != valueLayoutMap.end()) {
            resolvedLayout = it->second;
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
  // Identify edges where:
  // producer layout != expected layout

  for (const auto &[func, ops] : opSchedule) {
    for (Operation *consumerOp : ops) {

      if (consumerOp->getNumOperands() == 0) {
        continue;
      }

      for (unsigned operandIndex = 0;
           operandIndex < consumerOp->getNumOperands(); ++operandIndex) {

        Value v = consumerOp->getOperand(operandIndex);

        if (!mlir::isa<RankedTensorType>(v.getType())) {
          continue;
        }

        auto resolved = resolveProducerAndLayout(v);
        if (!resolved) {
          TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                       "Failed to resolve producer/layout in op: {}",
                       consumerOp->getName().getStringRef());
          continue;
        }

        auto [producerValue, actualLayout] = *resolved;

        // Resolve expected layout
        auto required = getRequiredLayout(consumerOp, operandIndex);
        if (!required) {
          continue;
        }
        TTNNLayoutAttr expectedLayout = *required;

        if (!expectedLayout) {
          continue;
        }

        // Check mismatch
        if (actualLayout == expectedLayout) {
          continue;
        }

        // Each consumer operand is already an SSA use. Keep that use as the
        // transition identity instead of creating a synthetic Edge.
        OpOperand &consumerUse = consumerOp->getOpOperand(operandIndex);
        if (!emittedUses.insert(&consumerUse).second) {
          continue;
        }

        GetTransitionOpsInput input{v, actualLayout, expectedLayout};
        auto transition = getTransitionOps(input);

        TransitionEdge transitionEdge{producerValue, consumerOp, operandIndex,
                                      actualLayout, expectedLayout,
                                      std::move(transition)};

        transitionEdges.push_back(std::move(transitionEdge));
      }
    }
  }
}

LogicalResult TransitionEdgeAnalysis::validateTransitions() {
  // Ensure that all generated TransitionEdge entries are valid
  // from an analysis perspective.
  // Only validate internally consistent, not the execution valid.

  for (const TransitionEdge &te : transitionEdges) {

    if (te.producerLayout == te.consumerLayout) {
      if (te.consumerOp) {
        te.consumerOp->emitError()
            << "Invalid TransitionEdge: producer and consumer layouts are "
               "identical, but a transition was generated.\n"
            << "Layout: " << te.producerLayout;
      }
      return failure();
    }

    if (te.transitionOps.empty() && te.producerLayout != te.consumerLayout) {
      if (te.consumerOp) {
        te.consumerOp->emitError()
            << "Invalid TransitionEdge: no transition ops generated for "
               "layout mismatch.\n"
            << "Producer layout: " << te.producerLayout << "\n"
            << "Consumer layout: " << te.consumerLayout;
      }
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Helper
//===----------------------------------------------------------------------===//

std::optional<TTNNLayoutAttr>
TransitionEdgeAnalysis::getRequiredLayout(mlir::Operation *op,
                                          unsigned operandIndex) const {
  // Return layout constraint for a specific operand of an op

  auto it = inputLayoutsMap.find(op);
  if (it != inputLayoutsMap.end()) {
    const auto &inputLayouts = it->second;

    if (operandIndex < inputLayouts.size()) {
      TTNNLayoutAttr layout = inputLayouts[operandIndex];
      if (layout) {
        return layout;
      }
    }
  }

  return std::nullopt;
}

static std::pair<TTNNLayoutAttr, TTNNLayoutAttr>
getLayoutsFromOp(Operation *op) {
  if (op->getNumOperands() == 0 || op->getNumResults() == 0) {
    return {nullptr, nullptr};
  }

  auto inType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto outType = dyn_cast<RankedTensorType>(op->getResult(0).getType());

  if (!inType || !outType) {
    return {nullptr, nullptr};
  }

  auto inLayout = dyn_cast_or_null<TTNNLayoutAttr>(inType.getEncoding());
  auto outLayout = dyn_cast_or_null<TTNNLayoutAttr>(outType.getEncoding());

  return {inLayout, outLayout};
}

llvm::SmallVector<TransitionOpInfo>
TransitionEdgeAnalysis::collectExistingOps(Value v) const {
  llvm::SmallVector<TransitionOpInfo> ops;

  Operation *curr = v.getDefiningOp();

  while (curr) {

    // Stop at compute op (has config)
    if (opConfigMap.count(curr)) {
      break;
    }

    if (isa<ttnn::ToLayoutOp, ttnn::ReshapeOp, ttnn::PermuteOp, ttnn::PadOp,
            ttnn::TypecastOp>(curr)) {

      auto [inLayout, outLayout] = getLayoutsFromOp(curr);

      if (inLayout && outLayout) {
        ops.push_back({curr->getName().getStringRef(), inLayout, outLayout});
      }
    }

    if (curr->getNumOperands() == 0) {
      break;
    }
    curr = curr->getOperand(0).getDefiningOp();
  }

  std::reverse(ops.begin(), ops.end());
  return ops;
}

llvm::SmallVector<TransitionOpInfo> TransitionEdgeAnalysis::getTransitionOps(
    const GetTransitionOpsInput &input) const {
  // Describe the sequence of conversion operations
  // between these 2 compute ops.

  llvm::SmallVector<TransitionOpInfo> result;

  auto existingOps = collectExistingOps(input.consumerOperand);
  for (auto &op : existingOps) {
    result.push_back(op);
  }

  if (input.producerOutputLayout != input.consumerInputLayout) {
    result.push_back(
        {"ToLayoutOp", input.producerOutputLayout, input.consumerInputLayout});
  }

  return result;
}

std::optional<std::pair<Value, TTNNLayoutAttr>>
TransitionEdgeAnalysis::resolveProducerAndLayout(Value v) {
  SmallPtrSet<Value, 8> visited;

  auto getLayoutFromValue = [&](Value value) -> TTNNLayoutAttr {
    if (auto it = valueLayoutMap.find(value); it != valueLayoutMap.end()) {
      return it->second;
    }

    auto tensorType = mlir::dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType) {
      return nullptr;
    }

    auto layout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    if (layout) {
      valueLayoutMap[value] = layout;
    }
    return layout;
  };

  auto getLayoutFromToLayout = [&](ToLayoutOp toLayout) -> TTNNLayoutAttr {
    auto resultType =
        mlir::dyn_cast<RankedTensorType>(toLayout.getResult().getType());
    if (!resultType) {
      return nullptr;
    }

    auto memConfig = toLayout.getMemoryConfig();
    if (!memConfig) {
      return nullptr;
    }

    auto bufferTypeAttr = memConfig->getBufferType();
    auto memoryLayoutAttr = memConfig->getTensorMemoryLayout();
    if (!bufferTypeAttr || !memoryLayoutAttr) {
      return nullptr;
    }

    auto elementType =
        mlir::tt::ttcore::TileType::get(resultType.getElementType());

    return TTNNLayoutAttr::Builder(v.getContext(), resultType.getShape(),
                                  elementType)
        .setBufferType(bufferTypeAttr.getValue())
        .setGridShape({1, 1})
        .setMemoryLayout(memoryLayoutAttr)
        .build();
  };

  TTNNLayoutAttr actualLayout = getLayoutFromValue(v);
  if (!actualLayout) {
    return std::nullopt;
  }

  while (true) {
    if (!visited.insert(v).second) {
      return std::nullopt;
    }

    if (!mlir::isa<RankedTensorType>(v.getType())) {
      return std::nullopt;
    }

    Operation *defOp = v.getDefiningOp();

    // A block argument is already an exact SSA producer value.
    if (!defOp) {
      return std::make_pair(v, actualLayout);
    }

    // Handle ToLayoutOp in producer chain
    if (auto toLayout = mlir::dyn_cast<ToLayoutOp>(defOp)) {
      actualLayout = getLayoutFromToLayout(toLayout);
      if (!actualLayout) {
        return std::nullopt;
      }
      valueLayoutMap[v] = actualLayout;
      v = toLayout->getOperand(0);
      continue;
    }

    // Stop at the exact result Value of the compute producer.
    if (opConfigMap.count(defOp)) {
      return std::make_pair(v, actualLayout);
    }

    // Otherwise, continue tracing back the producer chain
    if (defOp->getNumOperands() != 1) {
      return std::nullopt;
    }

    v = defOp->getOperand(0);
  }
}

} // namespace mlir::tt::ttnn::analysis