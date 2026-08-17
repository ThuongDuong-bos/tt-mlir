// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace mlir::tt::ttnn::optimizer_utils {

std::string formatValueShort(mlir::Value value) {
  std::string result;
  llvm::raw_string_ostream os(result);

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    os << "%arg" << blockArg.getArgNumber();
  } else if (auto opResult = mlir::dyn_cast<mlir::OpResult>(value)) {
    os << "%" << opResult.getResultNumber();

    if (Operation *defOp = value.getDefiningOp()) {
      os << " (" << defOp->getName().getStringRef() << ")";
    }
  } else {
    os << "<unknown-value>";
  }

  return os.str();
}

std::optional<TTNNLayoutAttr> extractOutputLayoutFromIR(Operation *op) {
  if (!op || op->getNumResults() == 0) {
    return std::nullopt;
  }

  auto outputType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!outputType) {
    return std::nullopt;
  }

  if (auto layout =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(outputType.getEncoding())) {
    return layout;
  }

  return std::nullopt;
}

std::optional<TTNNLayoutAttr> extractLayoutFromValue(mlir::Value value) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType) {
    return std::nullopt;
  }

  if (auto layout =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding())) {
    return layout;
  }

  return std::nullopt;
}

std::string layoutsToString(const std::vector<TTNNLayoutAttr> &layouts) {
  std::string result;
  llvm::raw_string_ostream os(result);

  os << "[";
  for (size_t i = 0; i < layouts.size(); ++i) {
    if (i > 0) {
      os << "\n ";
    }

    mlir::Attribute(layouts[i]).print(os);
  }
  os << "]";

  return os.str();
}

std::string layoutToString(const std::optional<TTNNLayoutAttr> &layout) {
  if (!layout || !*layout) {
    return "<none>";
  }

  std::string result;
  llvm::raw_string_ostream os(result);

  mlir::Attribute(*layout).print(os);

  return os.str();
}

std::string
parentOpsToString(llvm::ArrayRef<mlir::Operation *> parentOps) {
  std::string result;
  llvm::raw_string_ostream os(result);

  for (size_t i = 0; i < parentOps.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }

    Operation *op = parentOps[i];
    if (!op) {
      os << "<null>";
      continue;
    }

    os << op->getName().getStringRef();
  }

  return os.str();
}

TensorMemoryLayout getLayout(TTNNLayoutAttr layout) {
  if (TensorMemoryLayoutAttr memoryLayout = layout.getMemLayout()) {
    return memoryLayout.getValue();
  }

  return TensorMemoryLayout::Interleaved;
}

bool isTensorActivationOperand(mlir::Value value) {
  if (!mlir::isa<mlir::TensorType>(value.getType())) {
    return false;
  }

  if (mlir::isa<mlir::BlockArgument>(value)) {
    return false;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }

  // Cached values are model parameters rather than intermediate activations.
  if (mlir::isa<mlir::tt::ttcore::LoadCachedOp>(defOp)) {
    return false;
  }

  return true;
}

std::optional<unsigned>
findTensorInputLayoutIndex(mlir::Operation *op, mlir::Value value) {
  if (!op || !value) {
    return std::nullopt;
  }

  unsigned tensorLayoutIndex = 0;

  for (unsigned operandIndex = 0; operandIndex < op->getNumOperands();
       ++operandIndex) {
    mlir::Value operand = op->getOperand(operandIndex);

    if (!mlir::isa<mlir::RankedTensorType>(operand.getType())) {
      continue;
    }

    if (operand == value) {
      return tensorLayoutIndex;
    }

    ++tensorLayoutIndex;
  }

  return std::nullopt;
}

std::optional<size_t> getScheduledOpIndex(
    const llvm::DenseMap<
        mlir::func::FuncOp, llvm::SmallVector<mlir::Operation *>> &schedule,
    mlir::Operation *op) {
  if (!op) {
    return std::nullopt;
  }

  for (const auto &[func, operations] : schedule) {
    (void)func;

    for (size_t opIndex = 0; opIndex < operations.size(); ++opIndex) {
      if (operations[opIndex] == op) {
        return opIndex;
      }
    }
  }

  return std::nullopt;
}

std::vector<OpConfig::OpSpecificAttrs>
getUniqueOpSpecificAttrs(const std::vector<OpConfig> &configs) {
  llvm::DenseSet<OpConfig::OpSpecificAttrs> uniqueAttrs;
  std::vector<OpConfig::OpSpecificAttrs> attrVec;

  for (const OpConfig &config : configs) {
    if (uniqueAttrs.insert(config.opSpecificAttrs).second) {
      attrVec.push_back(config.opSpecificAttrs);
    }
  }
  return attrVec;
}

llvm::SmallVector<OpConfig> getUniqueTestConfigsForMatmulLinear(
    const std::vector<OpConfig> &consumerConfigs) {
  struct BufferMemLayoutKey {
    BufferType bufferType;
    TensorMemoryLayout memLayout;
    bool operator<(const BufferMemLayoutKey &other) const {
      if (bufferType != other.bufferType) {
        return bufferType < other.bufferType;
      }
      return memLayout < other.memLayout;
    }
  };

  // For each unique (bufferType, memLayout), collect:
  //   - A representative partial layout (with ignorePhysicalLayout=true)
  //   - The unique opSpecificAttrs from configs with that same memLayout
  //
  // MatmulProgramConfig depends on the tensor memory layout type
  // (width_sharded uses mcast_in0=true, height_sharded uses mcast_in0=false,
  // block_sharded uses a 2D config). Pairing a program config generated for
  // one memLayout type with a different memLayout would produce invalid
  // configs.
  struct LayoutGroup {
    TTNNLayoutAttr partialLayout;
    std::vector<OpConfig::OpSpecificAttrs> uniqueAttrs;
    llvm::DenseSet<OpConfig::OpSpecificAttrs> seenAttrs;
  };

  // Iteration order must be deterministic: downstream tie-breaks
  // (e.g., L1-spill's first-fit walk over fallback configs) commit to
  // the first entry, so a shuffled order produces different IR across
  // processes. std::unordered_map iteration depends on bucket layout
  // and varies between processes; std::map iterates in key order
  // (bufferType, then memLayout).
  std::map<BufferMemLayoutKey, LayoutGroup> groups;

  for (const OpConfig &config : consumerConfigs) {
    assert(config.outputLayout &&
           "Matmul/Linear configs must have valid output layout");

    BufferMemLayoutKey key{config.outputLayout.getBufferType(),
                           config.outputLayout.getMemLayout().getValue()};

    LayoutGroup &group = groups[key];
    if (!group.partialLayout) {
      TTNNLayoutAttr layout = config.outputLayout;
      group.partialLayout = layout.withIgnorePhysicalLayout(true);
    }
    if (group.seenAttrs.insert(config.opSpecificAttrs).second) {
      group.uniqueAttrs.push_back(config.opSpecificAttrs);
    }
  }

  // Build test configs: each partial layout is paired only with
  // opSpecificAttrs from configs of the same (bufferType, memLayout) group.
  llvm::SmallVector<OpConfig> testConfigs;
  for (const auto &[layoutKey, group] : groups) {
    for (const OpConfig::OpSpecificAttrs &attrs : group.uniqueAttrs) {
      testConfigs.push_back(OpConfig(group.partialLayout, attrs));
    }
  }

  return testConfigs;
}

llvm::SmallVector<OpConfig>
getUniqueTestConfigs(const std::vector<OpConfig> &consumerConfigs,
                     bool isMatmulOrLinear) {
  if (isMatmulOrLinear) {
    return getUniqueTestConfigsForMatmulLinear(consumerConfigs);
  }

  // For non-Matmul/Linear: only op-specific attrs matter, no output layout
  // needed.
  std::vector<OpConfig::OpSpecificAttrs> uniqueAttrs =
      getUniqueOpSpecificAttrs(consumerConfigs);
  llvm::SmallVector<OpConfig> testConfigs;
  for (const OpConfig::OpSpecificAttrs &attrs : uniqueAttrs) {
    testConfigs.push_back(OpConfig(/*outputLayout=*/nullptr, attrs));
  }

  return testConfigs;
}

} // namespace mlir::tt::ttnn::optimizer_utils
