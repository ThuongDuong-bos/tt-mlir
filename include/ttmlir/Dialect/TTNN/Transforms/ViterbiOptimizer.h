// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_VITERBIOPTIMIZER_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_VITERBIOPTIMIZER_H

#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"

#include "mlir/Pass/PassRegistry.h"

#include <cstdint>
#include <memory>

namespace mlir::tt::ttnn {

struct TTIRToTTNNCommonPipelineOptions;

struct ViterbiOptimizerOptions {
  llvm::StringMap<OutputLayoutOverrideParams> overrideOutputLayout;
  llvm::StringMap<Conv2dConfigOverrideParams> overrideConv2dConfig;
  bool memoryLayoutAnalysisEnabled = false;
  int64_t maxLegalLayouts = 64;
  bool rowMajorEnabled = false;
  bool reallocationAnalysisEnabled = false;
  double reallocationOffsetCapacity = 0.10;

  ViterbiOptimizerOptions() = default;

  explicit ViterbiOptimizerOptions(
      const TTIRToTTNNCommonPipelineOptions &pipelineOptions);
};

std::unique_ptr<::mlir::Pass> createViterbiOptimizer();
std::unique_ptr<::mlir::Pass>
createViterbiOptimizer(ViterbiOptimizerOptions options);

inline void registerViterbiOptimizer() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createViterbiOptimizer();
  });
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_VITERBIOPTIMIZER_H
