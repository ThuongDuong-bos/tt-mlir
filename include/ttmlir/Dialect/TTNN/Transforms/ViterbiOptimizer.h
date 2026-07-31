// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_VITERBIOPTIMIZER_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_VITERBIOPTIMIZER_H

#include "mlir/Pass/PassRegistry.h"

#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"

namespace tt::tt_metal::distributed {
class MeshDevice;
} // namespace tt::tt_metal::distributed

namespace mlir::tt::ttnn {

struct TTIRToTTNNCommonPipelineOptions;

//===----------------------------------------------------------------------===//
// ViterbiOptimizer
//===----------------------------------------------------------------------===//
struct ViterbiOptimizerOptions {
  llvm::StringMap<OutputLayoutOverrideParams> overrideOutputLayout;
  llvm::StringMap<Conv2dConfigOverrideParams> overrideConv2dConfig;
  int64_t maxLegalLayouts = 64;
  bool rowMajorEnabled = false;

  ViterbiOptimizerOptions() = default;

  explicit ViterbiOptimizerOptions(
      const TTIRToTTNNCommonPipelineOptions &pipelineOptions);
};

std::unique_ptr<::mlir::Pass> createViterbiOptimizer();
std::unique_ptr<::mlir::Pass>
createViterbiOptimizer(ViterbiOptimizerOptions options);

//===----------------------------------------------------------------------===//
// ViterbiOptimizer Registration
//===----------------------------------------------------------------------===//
inline void registerViterbiOptimizer() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createViterbiOptimizer();
  });
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_VITERBIOPTIMIZER_H
