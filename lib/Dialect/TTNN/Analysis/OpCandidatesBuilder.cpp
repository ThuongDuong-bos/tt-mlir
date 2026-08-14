// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpCandidatesBuilder.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <optional>

namespace mlir::tt::ttnn {

namespace {

struct GroupState {
  llvm::DenseMap<std::size_t, llvm::SmallVector<OpConfigCandidate>>
      candidatesByGroup;
  llvm::SmallVector<TTNNLayoutAttr> keptDefaults;
};

// Group emission order: tiled first, then row-major.
static constexpr std::array<CandidateGroup, 3> kTileGroups = {
    CandidateGroup::TileHeight, CandidateGroup::TileBlock,
    CandidateGroup::TileWidth};

static constexpr std::array<CandidateGroup, 3> kRowMajorGroups = {
    CandidateGroup::RowMajorHeight, CandidateGroup::RowMajorBlock,
    CandidateGroup::RowMajorWidth};

static std::size_t asIndex(CandidateGroup group) {
  return static_cast<std::size_t>(group);
}

static TensorMemoryLayout getMemoryLayoutOrInterleaved(TTNNLayoutAttr layout) {
  if (TensorMemoryLayoutAttr memLayout = layout.getMemLayout()) {
    return memLayout.getValue();
  }

  return TensorMemoryLayout::Interleaved;
}

// Conversion-like operations are removed from the graph optimized by Viterbi.
// TransitionEdgeAnalysis later materializes the required conversions between
// the selected layouts of the remaining compute operations.
static bool isConversionOp(mlir::Operation *op) {
  if (!op) {
    return false;
  }

  return llvm::isa<ttnn::ToLayoutOp, ttnn::ToMemoryConfigOp, ttnn::ReshapeOp,
                   ttnn::PermuteOp, ttnn::TransposeOp, ttnn::PadOp,
                   ttnn::TypecastOp>(op);
}

static bool isTensorActivationOperand(mlir::Value value) {
  if (!mlir::isa<mlir::TensorType>(value.getType())) {
    return false;
  }

  if (llvm::isa<mlir::BlockArgument>(value)) {
    return false;
  }

  mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return false;
  }

  if (mlir::isa<mlir::tt::ttcore::LoadCachedOp>(definingOp)) {
    return false;
  }

  return true;
}

static llvm::SmallVector<mlir::Value>
collectActivationOperands(mlir::Operation *op, bool isFirstOpInFunction) {
  llvm::SmallVector<mlir::Value> activationOperands;

  if (!op) {
    return activationOperands;
  }

  // The first scheduled compute op may consume a function argument. Treat its
  // first tensor operand as the activation even though block arguments are not
  // intermediate activation values.
  if (isFirstOpInFunction) {
    if (op->getNumOperands() == 0) {
      return activationOperands;
    }

    mlir::Value operand = op->getOperand(0);
    if (mlir::isa<mlir::TensorType>(operand.getType())) {
      activationOperands.push_back(operand);
    }

    return activationOperands;
  }

  for (mlir::Value operand : op->getOperands()) {
    if (isTensorActivationOperand(operand)) {
      activationOperands.push_back(operand);
    }
  }

  return activationOperands;
}

static bool hasTensorOperand(mlir::Operation *op) {
  if (!op) {
    return false;
  }

  return llvm::any_of(op->getOperands(), [](mlir::Value operand) {
    return mlir::isa<mlir::TensorType>(operand.getType());
  });
}

static bool hasSameLayoutClass(TTNNLayoutAttr lhs, TTNNLayoutAttr rhs) {
  if (!lhs || !rhs) {
    return false;
  }

  if (lhs.getBufferType() != rhs.getBufferType() ||
      getMemoryLayoutOrInterleaved(lhs) != getMemoryLayoutOrInterleaved(rhs) ||
      lhs.getLayout() != rhs.getLayout()) {
    return false;
  }

  if (lhs.hasShardedTensorMemoryLayout() &&
      rhs.hasShardedTensorMemoryLayout()) {
    return llvm::equal(lhs.getGridShape(), rhs.getGridShape());
  }

  return true;
}

static TTNNLayoutAttr pickActivationLayoutForOutputClass(
    mlir::RankedTensorType tensorType,
    const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
    std::optional<TTNNLayoutAttr> targetOutputLayout) {
  auto layoutsByTypeIt = tensorTypePossibleLayouts.find(tensorType);
  if (layoutsByTypeIt == tensorTypePossibleLayouts.end() ||
      layoutsByTypeIt->second.empty()) {
    return TTNNLayoutAttr{};
  }

  const auto &scalarLayoutsMap = layoutsByTypeIt->second;
  mlir::Type scalarType = tensorType.getElementType();
  auto scalarLayoutsIt = scalarLayoutsMap.find(scalarType);
  if (scalarLayoutsIt == scalarLayoutsMap.end()) {
    scalarLayoutsIt = scalarLayoutsMap.begin();
  }

  for (const auto &layoutsByMemoryLayout : scalarLayoutsIt->second) {
    for (const auto &layouts : layoutsByMemoryLayout) {
      for (TTNNLayoutAttr legalLayout : layouts) {
        if (targetOutputLayout &&
            !hasSameLayoutClass(legalLayout, *targetOutputLayout)) {
          continue;
        }

        return legalLayout;
      }
    }
  }

  return TTNNLayoutAttr{};
}

static llvm::SmallVector<TTNNLayoutAttr>
extractInputLayouts(mlir::Operation *op,
                    llvm::ArrayRef<mlir::Value> activationOperands,
                    const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
                    std::optional<TTNNLayoutAttr> targetOutputLayout) {
  llvm::SmallVector<TTNNLayoutAttr> inputLayouts;

  if (!op) {
    return inputLayouts;
  }

  const auto isActivationOperand = [&](mlir::Value operand) {
    return llvm::is_contained(activationOperands, operand);
  };

  for (mlir::Value operand : op->getOperands()) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
    if (!tensorType) {
      continue;
    }

    TTNNLayoutAttr chosenLayout;
    TTNNLayoutAttr currentLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());

    if (isActivationOperand(operand)) {
      chosenLayout = pickActivationLayoutForOutputClass(
          tensorType, tensorTypePossibleLayouts, targetOutputLayout);

      // A candidate is invalid if an activation cannot be represented using
      // the same layout class as the candidate output.
      if (!chosenLayout && targetOutputLayout) {
        return {};
      }
    } else {
      // Weights, biases, and other non-activation tensors retain the layout
      // already encoded in the IR.
      chosenLayout = currentLayout;
    }

    if (chosenLayout) {
      inputLayouts.push_back(chosenLayout);
    }
  }

  return inputLayouts;
}

static bool isDefaultLayout(TTNNLayoutAttr layout) {
  return getMemoryLayoutOrInterleaved(layout) ==
         TensorMemoryLayout::Interleaved;
}

static bool keepDefaultLayout(TTNNLayoutAttr layout,
                              llvm::SmallVectorImpl<TTNNLayoutAttr> &kept) {
  if (!isDefaultLayout(layout)) {
    return true;
  }

  if (llvm::any_of(kept, [&](TTNNLayoutAttr keptLayout) {
        return keptLayout.getBufferType() == layout.getBufferType() &&
               keptLayout.getLayout() == layout.getLayout();
      })) {
    return false;
  }

  kept.push_back(layout);
  return true;
}

static bool isLayoutMatching(llvm::ArrayRef<mlir::Value> activationOperands,
                             llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                             TTNNLayoutAttr outputLayout) {
  std::size_t inputLayoutIndex = 0;

  for (mlir::Value operand : activationOperands) {
    if (!mlir::isa<mlir::RankedTensorType>(operand.getType())) {
      continue;
    }

    if (inputLayoutIndex >= inputLayouts.size()) {
      return false;
    }

    TTNNLayoutAttr inputLayout = inputLayouts[inputLayoutIndex];

    if (inputLayout.getLayout() != outputLayout.getLayout() ||
        getMemoryLayoutOrInterleaved(inputLayout) !=
            getMemoryLayoutOrInterleaved(outputLayout)) {
      return false;
    }

    if (inputLayout.hasShardedTensorMemoryLayout() &&
        outputLayout.hasShardedTensorMemoryLayout() &&
        !llvm::equal(inputLayout.getGridShape(), outputLayout.getGridShape())) {
      return false;
    }

    ++inputLayoutIndex;
  }

  return true;
}

static std::optional<LayoutFamily> getLayoutFamily(TTNNLayoutAttr layout) {
  const TensorMemoryLayout memoryLayout = getMemoryLayoutOrInterleaved(layout);
  const bool tiled = layout.isTiled();

  switch (memoryLayout) {
  case TensorMemoryLayout::HeightSharded:
    return tiled ? LayoutFamily::TileHeight : LayoutFamily::RowMajorHeight;
  case TensorMemoryLayout::BlockSharded:
    return tiled ? LayoutFamily::TileBlock : LayoutFamily::RowMajorBlock;
  case TensorMemoryLayout::WidthSharded:
    return tiled ? LayoutFamily::TileWidth : LayoutFamily::RowMajorWidth;
  default:
    return std::nullopt;
  }
}

static CandidateGroup getCandidateGroup(LayoutFamily family) {
  switch (family) {
  case LayoutFamily::TileHeight:
    return CandidateGroup::TileHeight;
  case LayoutFamily::TileBlock:
    return CandidateGroup::TileBlock;
  case LayoutFamily::TileWidth:
    return CandidateGroup::TileWidth;
  case LayoutFamily::RowMajorHeight:
    return CandidateGroup::RowMajorHeight;
  case LayoutFamily::RowMajorBlock:
    return CandidateGroup::RowMajorBlock;
  case LayoutFamily::RowMajorWidth:
    return CandidateGroup::RowMajorWidth;
  }

  llvm_unreachable("unsupported layout family");
}

static std::optional<std::size_t> classifyGroup(mlir::Operation *op,
                                                TTNNLayoutAttr layout) {
  if (!op || !layout) {
    return std::nullopt;
  }

  const TensorMemoryLayout memoryLayout = getMemoryLayoutOrInterleaved(layout);

  if (memoryLayout == TensorMemoryLayout::Interleaved) {
    if (layout.getBufferType() == BufferType::DRAM) {
      return asIndex(CandidateGroup::DefaultDRAM);
    }

    if (layout.getBufferType() == BufferType::L1) {
      return asIndex(CandidateGroup::DefaultL1);
    }

    return std::nullopt;
  }

  std::optional<LayoutFamily> family = getLayoutFamily(layout);
  if (!family) {
    return std::nullopt;
  }

  return asIndex(getCandidateGroup(*family));
}

static void
appendGroupedCandidates(llvm::SmallVectorImpl<OpConfigCandidate> &destination,
                        GroupState &state) {
  const auto appendGroupSet = [&](llvm::ArrayRef<CandidateGroup> groups,
                                  std::size_t &nextGroupIndex) {
    for (CandidateGroup group : groups) {
      auto groupIt = state.candidatesByGroup.find(asIndex(group));
      if (groupIt == state.candidatesByGroup.end() || groupIt->second.empty()) {
        continue;
      }

      for (OpConfigCandidate &candidate : groupIt->second) {
        candidate.groupIndex = nextGroupIndex;
        destination.push_back(std::move(candidate));
      }

      ++nextGroupIndex;
    }
  };

  std::size_t nextGroupIndex = 0;

  static constexpr std::array<CandidateGroup, 2> kDefaultGroups = {
      CandidateGroup::DefaultDRAM, CandidateGroup::DefaultL1};

  appendGroupSet(kDefaultGroups, nextGroupIndex);
  appendGroupSet(kTileGroups, nextGroupIndex);
  appendGroupSet(kRowMajorGroups, nextGroupIndex);
}

static void printScheduleOrder(
    const llvm::DenseMap<mlir::func::FuncOp,
                         llvm::SmallVector<mlir::Operation *>> &schedule,
    llvm::StringRef graphKind) {
  TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
               "=== {} Operation Schedule Order ===", graphKind);

  for (const auto &[func, operations] : schedule) {
    (void)func;

    for (std::size_t index = 0; index < operations.size(); ++index) {
      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer, "Op[{}]: {}", index,
                   operations[index]->getName().getStringRef());
    }
  }
}

} // namespace

PrunedGraphInfo OpCandidatesBuilder::makePrunedSubgraph(
    const llvm::DenseMap<mlir::func::FuncOp,
                         llvm::SmallVector<mlir::Operation *>> &schedule)
    const {
  PrunedGraphInfo info;

  for (const auto &[func, operations] : schedule) {
    llvm::SmallVector<mlir::Operation *> &prunedOps = info.prunedSchedule[func];
    mlir::Operation *lastKeptOp = nullptr;

    for (std::size_t fullIndex = 0; fullIndex < operations.size();
         ++fullIndex) {
      mlir::Operation *op = operations[fullIndex];
      info.fullOpIndex[op] = fullIndex;

      if (isConversionOp(op)) {
        if (lastKeptOp) {
          info.removedOpsAfterPrunedOp[lastKeptOp].push_back(op);
        }
        continue;
      }

      info.prunedOpIndex[op] = prunedOps.size();
      prunedOps.push_back(op);
      lastKeptOp = op;
    }
  }

  return info;
}

OpCandidateBuilderResult OpCandidatesBuilder::buildCandidatesFromSchedule(
    const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
    const llvm::DenseMap<mlir::func::FuncOp,
                         llvm::SmallVector<mlir::Operation *>> &schedule,
    const llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>>
        &legalOpConfigs) const {
  OpCandidateBuilderResult generatedCandidates;

  if (tensorTypePossibleLayouts.empty() || schedule.empty() ||
      legalOpConfigs.empty()) {
    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                 "Cannot build Viterbi candidates: tensorLayoutsEmpty={} "
                 "scheduleEmpty={} legalConfigsEmpty={}",
                 tensorTypePossibleLayouts.empty(), schedule.empty(),
                 legalOpConfigs.empty());
    return generatedCandidates;
  }

  for (const auto &[func, operations] : schedule) {
    (void)func;

    for (std::size_t opIndex = 0; opIndex < operations.size(); ++opIndex) {
      mlir::Operation *op = operations[opIndex];

      if (op->getNumResults() == 0) {
        continue;
      }

      auto resultType =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
      if (!resultType) {
        continue;
      }

      auto opConfigIt = legalOpConfigs.find(op);
      if (opConfigIt == legalOpConfigs.end()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                     "No legal op configs for scheduled op {}",
                     op->getName().getStringRef());
        continue;
      }

      const std::vector<OpConfig> &selectedOpConfigs = opConfigIt->second;
      if (selectedOpConfigs.empty()) {
        continue;
      }

      const bool isFirstOpInFunction = opIndex == 0;
      const llvm::SmallVector<mlir::Value> activationOperands =
          collectActivationOperands(op, isFirstOpInFunction);
      const bool opHasTensorOperand = hasTensorOperand(op);

      llvm::SmallVector<OpConfigCandidate> opCandidates;
      opCandidates.reserve(selectedOpConfigs.size());

      llvm::SmallVector<TTNNLayoutAttr> keptOutputLayouts;
      GroupState state;

      for (const OpConfig &opConfig : selectedOpConfigs) {
        const TTNNLayoutAttr outputLayout = opConfig.outputLayout;
        if (!outputLayout) {
          continue;
        }

        if (llvm::is_contained(keptOutputLayouts, outputLayout)) {
          continue;
        }

        llvm::SmallVector<TTNNLayoutAttr> inputLayouts = extractInputLayouts(
            op, activationOperands, tensorTypePossibleLayouts, outputLayout);

        if (inputLayouts.empty() && opHasTensorOperand) {
          continue;
        }

        if (!isLayoutMatching(activationOperands, inputLayouts, outputLayout)) {
          continue;
        }

        if (!keepDefaultLayout(outputLayout, state.keptDefaults)) {
          continue;
        }

        std::optional<std::size_t> groupIndex = classifyGroup(op, outputLayout);
        if (!groupIndex) {
          continue;
        }

        OpConfigCandidate candidate;
        candidate.inputLayouts.assign(inputLayouts.begin(), inputLayouts.end());
        candidate.opConfig = opConfig;

        keptOutputLayouts.push_back(outputLayout);
        state.candidatesByGroup[*groupIndex].push_back(std::move(candidate));
      }

      appendGroupedCandidates(opCandidates, state);

      if (!opCandidates.empty()) {
        generatedCandidates.candidateMap[op] = std::move(opCandidates);
      }
    }
  }

  return generatedCandidates;
}

void OpCandidatesBuilder::buildFullCandidates(
    const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
    const llvm::DenseMap<mlir::func::FuncOp,
                         llvm::SmallVector<mlir::Operation *>> &schedule,
    const llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>>
        &legalOpConfigs) {
  printScheduleOrder(schedule, "Full Graph");
  fullCandidates = buildCandidatesFromSchedule(tensorTypePossibleLayouts,
                                               schedule, legalOpConfigs);
}

void OpCandidatesBuilder::buildPrunedCandidates(
    const TensorTypeLayoutsMap &tensorTypePossibleLayouts,
    const llvm::DenseMap<mlir::func::FuncOp,
                         llvm::SmallVector<mlir::Operation *>> &schedule,
    const llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>>
        &legalOpConfigs) {
  prunedGraphInfo = makePrunedSubgraph(schedule);
  printScheduleOrder(prunedGraphInfo.prunedSchedule, "Pruned Graph");
  prunedCandidates = buildCandidatesFromSchedule(tensorTypePossibleLayouts,
                                                 prunedGraphInfo.prunedSchedule,
                                                 legalOpConfigs);
}

} // namespace mlir::tt::ttnn
