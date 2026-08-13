// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/ReallocationAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/ConvRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/ScalarDataTypeAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpCandidatesBuilder.h"
#include "ttmlir/Dialect/TTNN/Analysis/OperationScheduler.h"
#include "ttmlir/Dialect/TTNN/Analysis/ViterbiPolicy.h"
#include "ttmlir/Dialect/TTNN/Analysis/TransitionEdgeAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNTraits.h"
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

namespace mlir::tt::ttnn {

ViterbiOptimizerOptions::ViterbiOptimizerOptions(
    const TTIRToTTNNCommonPipelineOptions &pipelineOptions)
    : overrideOutputLayout(pipelineOptions.overrideOutputLayout),
      overrideConv2dConfig(pipelineOptions.overrideConv2dConfig),
      memoryLayoutAnalysisEnabled(pipelineOptions.memoryLayoutAnalysisEnabled),
      maxLegalLayouts(pipelineOptions.maxLegalLayouts),
      rowMajorEnabled(pipelineOptions.rowMajorEnabled),
      reallocationAnalysisEnabled(pipelineOptions.reallocationAnalysisEnabled),
      reallocationOffsetCapacity(pipelineOptions.reallocationOffsetCapacity) {}

namespace impl {

std::unique_ptr<::mlir::Pass> createViterbiOptimizer();
std::unique_ptr<::mlir::Pass>
createViterbiOptimizer(ViterbiOptimizerOptions options);

template <typename DerivedT>
class ViterbiOptimizerBase : public ::mlir::OperationPass<::mlir::ModuleOp> {
public:
  using Base = ViterbiOptimizerBase;

  ViterbiOptimizerBase()
      : ::mlir::OperationPass<::mlir::ModuleOp>(
            ::mlir::TypeID::get<DerivedT>()) {}

  ViterbiOptimizerBase(const ViterbiOptimizerBase &other)
      : ::mlir::OperationPass<::mlir::ModuleOp>(other) {}

  ViterbiOptimizerBase &operator=(const ViterbiOptimizerBase &) = delete;
  ViterbiOptimizerBase(ViterbiOptimizerBase &&) = delete;
  ViterbiOptimizerBase &operator=(ViterbiOptimizerBase &&) = delete;

  ~ViterbiOptimizerBase() override = default;

  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("viterbi-optimizer");
  }

  ::llvm::StringRef getArgument() const override { return "viterbi-optimizer"; }

  ::llvm::StringRef getDescription() const override {
    return "Determine TTNN operation configurations using Viterbi "
           "optimization.";
  }

  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ViterbiOptimizer");
  }

  ::llvm::StringRef getName() const override { return "ViterbiOptimizer"; }

  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ViterbiOptimizerBase<DerivedT>)

  ViterbiOptimizerBase(ViterbiOptimizerOptions options)
      : ViterbiOptimizerBase() {
    overrideOutputLayout = std::move(options.overrideOutputLayout);
    overrideConv2dConfig = std::move(options.overrideConv2dConfig);
    memoryLayoutAnalysisEnabled = options.memoryLayoutAnalysisEnabled;
    maxLegalLayouts = options.maxLegalLayouts;
    rowMajorEnabled = options.rowMajorEnabled;
    reallocationAnalysisEnabled = options.reallocationAnalysisEnabled;
    reallocationOffsetCapacity = options.reallocationOffsetCapacity;
  }

protected:
  ::mlir::Pass::Option<llvm::StringMap<OutputLayoutOverrideParams>,
                       OutputLayoutOverrideParser>
      overrideOutputLayout{
          *this, OptionNames::overrideOutputLayout,
          ::llvm::cl::desc(
              "Override output tensor layout for specific operations."),
          ::llvm::cl::init(llvm::StringMap<OutputLayoutOverrideParams>())};

  ::mlir::Pass::Option<llvm::StringMap<Conv2dConfigOverrideParams>,
                       Conv2dConfigOverrideParser>
      overrideConv2dConfig{
          *this, OptionNames::overrideConv2dConfig,
          ::llvm::cl::desc(
              "Override Conv2d configuration for specific operations."),
          ::llvm::cl::init(llvm::StringMap<Conv2dConfigOverrideParams>())};

  ::mlir::Pass::Option<bool> memoryLayoutAnalysisEnabled{
    *this, OptionNames::memoryLayoutAnalysisEnabled,
    ::llvm::cl::desc("Enable memory layout optimization."),
    ::llvm::cl::init(false)};

  ::mlir::Pass::Option<int64_t> maxLegalLayouts{
      *this, OptionNames::maxLegalLayouts,
      ::llvm::cl::desc(
          "Maximum number of sharded layouts retained by legal layout "
          "analysis."),
      ::llvm::cl::init(64)};

  ::mlir::Pass::Option<bool> rowMajorEnabled{
      *this, "row-major-enabled",
      ::llvm::cl::desc(
          "Enable row-major layout generation in legal layout analysis."),
      ::llvm::cl::init(false)};

  ::mlir::Pass::Option<bool> reallocationAnalysisEnabled{
      *this, "reallocation-analysis-enabled",
      ::llvm::cl::desc("Enable L1 reallocation analysis."),
      ::llvm::cl::init(false)};

  ::mlir::Pass::Option<double> reallocationOffsetCapacity{
      *this, "reallocation-offset-capacity",
      ::llvm::cl::desc(
          "Reserved L1 fraction between CB and tensor allocations."),
      ::llvm::cl::init(0.10)};

private:
  friend std::unique_ptr<::mlir::Pass> createViterbiOptimizer() {
    return std::make_unique<DerivedT>();
  }

  friend std::unique_ptr<::mlir::Pass>
  createViterbiOptimizer(ViterbiOptimizerOptions options) {
    return std::make_unique<DerivedT>(std::move(options));
  }
};

} // namespace impl

std::unique_ptr<::mlir::Pass> createViterbiOptimizer() {
  return impl::createViterbiOptimizer();
}

std::unique_ptr<::mlir::Pass>
createViterbiOptimizer(ViterbiOptimizerOptions options) {
  return impl::createViterbiOptimizer(std::move(options));
}

namespace {

class ViterbiOptimizer : public impl::ViterbiOptimizerBase<ViterbiOptimizer> {
public:
  using impl::ViterbiOptimizerBase<ViterbiOptimizer>::ViterbiOptimizerBase;

  void runOnOperation() final {
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal(
        "ViterbiOptimizer pass requires OpModel support to be enabled.");
#else
    ModuleOp moduleOp = getOperation();
    op_model::ScopedSingletonDeviceGuard deviceGuard(moduleOp);

    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                 "Running ViterbiOptimizer.");

    // Set the default Conv2d slice configuration before candidate validation.
    applyConvSliceConfig(moduleOp);

    assertOverridesValid();

    ttcore::GridAttr deviceGrid =
        ttcore::lookupDevice(moduleOp).getWorkerGrid();

    llvm::DenseMap<Operation *, std::vector<OpConfig>> legalConfigs;
    llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> opSchedule;
    llvm::DenseMap<Operation *, OpConfig> opConfigMap;
    llvm::DenseMap<Operation *, llvm::SmallVector<TTNNLayoutAttr>>
        inputLayoutsMap;
    llvm::SmallVector<analysis::TransitionEdge> transitionEdges;
    llvm::SmallVector<SpillRequest> spillRequests;

    // Step 1: Run legal analyses.

    ScalarDataTypeAnalysis scalarDataTypeAnalysis =
        getAnalysis<ScalarDataTypeAnalysis>();
    scalarDataTypeAnalysis.init(
        ScalarDataTypeAnalysisInput(&overrideOutputLayout));

    auto scalarTypes = scalarDataTypeAnalysis.getResult();

    TTMLIR_TRACE(ttmlir::LogComponent::ViterbiOptimizer,
                 "ScalarDataTypeAnalysis found {0} unique scalar types.",
                 scalarTypes.size());

    LegalTensorLayoutAnalysis legalTensorLayoutAnalysis =
        getAnalysis<LegalTensorLayoutAnalysis>();
    legalTensorLayoutAnalysis.init(LegalTensorLayoutAnalysisInput(
        deviceGrid, &scalarTypes, rowMajorEnabled));
    TensorTypeLayoutsMap tensorTypePossibleLayouts =
        legalTensorLayoutAnalysis.getResult();

    moduleOp.walk([&](func::FuncOp func) {
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
        return;
      }

      func.walk([&](Operation *op) {
        if (!optimizer_utils::opHasTensorResult(op)) {
          // Constraint sink operations have no tensor result but still need
          // input layout validation.
          if (mlir::dyn_cast<OpModel>(op) && optimizer_utils::isSinkOp(op)) {
            legalConfigs[op] = {OpConfig{TTNNLayoutAttr()}};
          }
          return;
        }

        // Operations without OpModel support cannot be validated by the
        // backend.
        if (!mlir::dyn_cast<OpModel>(op)) {
          return;
        }

        // Existing ToLayout operations are managed by other layout passes.
        if (mlir::isa<ToLayoutOp>(op)) {
          return;
        }

        RankedTensorType tensorType =
            mlir::cast<RankedTensorType>(op->getResult(0).getType());

        auto tensorLayouts = tensorTypePossibleLayouts.find(tensorType);
        assert(tensorLayouts != tensorTypePossibleLayouts.end() &&
               "No layouts found for tensor type");

        LegalOpLayoutAnalysis legalOpLayoutAnalysis =
            getChildAnalysis<LegalOpLayoutAnalysis>(op);
        legalOpLayoutAnalysis.init(LegalOpLayoutAnalysisInput(
            &tensorLayouts->getSecond(), maxLegalLayouts, &overrideOutputLayout,
            rowMajorEnabled));

        LegalOpConfigAnalysis legalOpConfigAnalysis =
            getChildAnalysis<LegalOpConfigAnalysis>(op);
        legalOpConfigAnalysis.init(LegalOpConfigAnalysisInput(
            legalOpLayoutAnalysis.getResult(), &overrideConv2dConfig,
            /*overrideConv3dConfig=*/nullptr));

        legalConfigs[op] = legalOpConfigAnalysis.getResult();
      });
    });

    // Step 2: Run optimal layout/config analysis.
    if (memoryLayoutAnalysisEnabled) {
      // Build the full operation schedule first. Conversion-like operations are
      // kept here because transition analysis still needs the original graph.
      analysis::OperationScheduler opScheduleAnalysis(getOperation());
      opScheduleAnalysis.init(analysis::OperationSchedulerInput(
          analysis::SchedulePolicy::DepthFirstSearch));
      opSchedule = opScheduleAnalysis.getResult().schedule;

      // Build candidates from the legal OpConfig set, not from an already
      // selected OpConfig map. Viterbi runs only on the pruned graph, where
      // reshape/permute/to_layout/typecast/pad-like conversion operations have
      // been removed.
      OpCandidatesBuilder opCandidatesBuilder;
      opCandidatesBuilder.buildPrunedCandidates(
          tensorTypePossibleLayouts, opSchedule, legalConfigs);

      const OpCandidateBuilderResult &opCandidateResult =
          opCandidatesBuilder.getPrunedCandidates();
      const PrunedGraphInfo &prunedGraphInfo =
          opCandidatesBuilder.getPrunedGraphInfo();

      ViterbiPolicy viterbiPolicy(opCandidateResult,
                                  prunedGraphInfo.prunedSchedule,
                                  prunedGraphInfo);
      ViterbiResult result = viterbiPolicy.run();

      if (!isSolverOk(result.status)) {
        getOperation()->emitError()
            << "Viterbi memory selection failed with status "
            << getSolverStatusString(result.status);
        signalPassFailure();
        return;
      }

      opConfigMap = result.optimalConfigurations;
      inputLayoutsMap = result.inputLayouts;
      spillRequests = result.spillRequests;

      // Transition analysis uses the full schedule so it can reconstruct the
      // conversion operations removed from the Viterbi graph.
      analysis::TransitionEdgeAnalysis transitionEdgeAnalysis(
          opConfigMap, inputLayoutsMap, opSchedule);
      if (failed(transitionEdgeAnalysis.run())) {
        signalPassFailure();
        return;
      }
      transitionEdges = transitionEdgeAnalysis.getTransitionEdges();
    } else {
      // Fall back to the existing per-op configuration selection when memory
      // layout optimization is disabled.
      OpConfigAnalysis opConfigAnalysis = getAnalysis<OpConfigAnalysis>();
      opConfigAnalysis.init(OpConfigAnalysisInput(std::move(legalConfigs)));
      opConfigMap = opConfigAnalysis.getResult();
    }


    // Step 3: Apply transformations based on analysis results.
    //
    // Apply the selected output layout and operation-specific configuration,
    // then materialize transition and spill decisions. Reallocation analysis
    // runs on the resulting SSA graph after these rewrites.
    moduleOp.walk([&](func::FuncOp func) {
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
        return;
      }

      func.walk([&](Operation *op) {
        if (op->getNumResults() == 0) {
          return;
        }

        // EmptyOp is handled through the destination operand update.
        if (mlir::isa<EmptyOp>(op)) {
          return;
        }

        if (!mlir::isa<RankedTensorType>(op->getResult(0).getType())) {
          return;
        }

        if (!opConfigMap.contains(op)) {
          return;
        }

        RankedTensorType tensorType =
            mlir::cast<RankedTensorType>(op->getResult(0).getType());
        llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

        const OpConfig &selectedConfig = opConfigMap.at(op);
        TTNNLayoutAttr chosenLayout = selectedConfig.outputLayout;

        Type originalElementType = tensorType.getElementType();
        Type newElementType = originalElementType;

        // Preserve quantized element types. For non-quantized tensors, use the
        // scalar element type selected by the chosen layout.
        if (!mlir::isa<mlir::quant::QuantizedType>(originalElementType)) {
          newElementType = chosenLayout.getScalarElementType();
        } else {
          auto quantizedType =
              mlir::cast<mlir::quant::QuantizedType>(originalElementType);

          assert(
              quantizedType.getStorageType() ==
                  chosenLayout.getScalarElementType() &&
              "Layout scalar element type must match quantized storage type");
        }

        RankedTensorType newTensorType =
            RankedTensorType::get(tensorShape, newElementType, chosenLayout);

        // Update the operation result type with the selected output layout.
        op->getResult(0).setType(newTensorType);

        // Destination-style operations require the destination operand type to
        // match the operation result type.
        if (mlir::isa<DestinationStyleOpInterface>(op)) {
          op->getOperands().back().setType(newTensorType);
        }

        // Apply operation-specific configuration attributes.
        llvm::TypeSwitch<Operation *, void>(op)
            .Case<Conv2dOp>([&](Conv2dOp convOp) {
              if (!std::holds_alternative<Conv2dAttrs>(
                      selectedConfig.opSpecificAttrs)) {
                return;
              }

              const Conv2dAttrs &conv2dAttrs =
                  std::get<Conv2dAttrs>(selectedConfig.opSpecificAttrs);

              if (conv2dAttrs.conv2dConfig.has_value()) {
                convOp.setConv2dConfigAttr(conv2dAttrs.conv2dConfig.value());
              }

              if (conv2dAttrs.deviceComputeKernelConfig.has_value()) {
                convOp.setComputeConfigAttr(
                    conv2dAttrs.deviceComputeKernelConfig.value());
              }
            })
            .Case<ConvTranspose2dOp>([&](ConvTranspose2dOp convOp) {
              if (!std::holds_alternative<Conv2dAttrs>(
                      selectedConfig.opSpecificAttrs)) {
                return;
              }

              const Conv2dAttrs &conv2dAttrs =
                  std::get<Conv2dAttrs>(selectedConfig.opSpecificAttrs);

              if (conv2dAttrs.conv2dConfig.has_value()) {
                convOp.setConv2dConfigAttr(conv2dAttrs.conv2dConfig.value());
              }
            })
            .Case<MatmulOp, LinearOp>([&](auto matmulOp) {
              if (!std::holds_alternative<MatmulAttrs>(
                      selectedConfig.opSpecificAttrs)) {
                return;
              }

              const MatmulAttrs &matmulAttrs =
                  std::get<MatmulAttrs>(selectedConfig.opSpecificAttrs);

              if (matmulAttrs.computeKernelConfig.has_value()) {
                matmulOp.setComputeConfigAttr(
                    matmulAttrs.computeKernelConfig.value());
              }

              if (!matmulAttrs.matmulProgramConfig.has_value()) {
                return;
              }

              auto programConfig = matmulAttrs.matmulProgramConfig.value();

              matmulOp.setMatmulProgramConfigAttr(programConfig);

              // Remove the standalone activation when the selected program
              // configuration already contains a fused activation.
              bool hasFusedActivation =
                  llvm::TypeSwitch<Attribute, bool>(programConfig)
                      .Case<
                          MatmulMultiCoreReuseMultiCastProgramConfigAttr,
                          MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
                          MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
                          [](auto config) {
                            return config.getFusedActivation() != nullptr;
                          })
                      .Default([](Attribute) { return false; });

              if (hasFusedActivation) {
                matmulOp.removeActivationAttr();
              }
            });
      });

      // Materialize layout changes after selected configurations have been
      // applied. Transition rewrites are consumer-local; spill rewrites use
      // exact SSA values selected by Viterbi.
      processTransitionEdges(transitionEdges, func);
      processSpillRequests(spillRequests, func);

      // Update the function type to reflect the updated return operand types.
      SmallVector<Type> funcResultTypes;

      func.walk([&](Operation *op) {
        if (op->getNumResults() != 0) {
          return;
        }

        func::ReturnOp returnOp = mlir::dyn_cast<func::ReturnOp>(op);
        if (!returnOp) {
          return;
        }

        funcResultTypes.append(returnOp.getOperandTypes().begin(),
                               returnOp.getOperandTypes().end());
      });

      FunctionType funcType = func.getFunctionType();
      FunctionType newFuncType = FunctionType::get(
          func.getContext(), funcType.getInputs(), funcResultTypes);

      func.setType(newFuncType);
    });

    // Step 4: Run reallocation analysis on the final layout/materialization
    // graph. Rebuild the schedule and fill configs for conversion ops inserted
    // during transition/spill materialization.
    if (memoryLayoutAnalysisEnabled && reallocationAnalysisEnabled) {
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>>
          reallocationSchedule;
      llvm::DenseMap<Operation *, OpConfig> reallocationOpConfigMap =
          opConfigMap;

      moduleOp.walk([&](func::FuncOp func) {
        if (!ttmlir::utils::isForwardDeviceFunc(func)) {
          return;
        }

        auto &schedule = reallocationSchedule[func];
        func.walk([&](Operation *op) {
          schedule.push_back(op);

          if (reallocationOpConfigMap.contains(op)) {
            return;
          }

          std::optional<TTNNLayoutAttr> outputLayout =
              optimizer_utils::extractOutputLayoutFromIR(op);
          if (outputLayout) {
            reallocationOpConfigMap[op] = OpConfig{*outputLayout};
          }
        });
      });

      ReallocationAnalysis reallocationAnalysis =
          getAnalysis<ReallocationAnalysis>();
      reallocationAnalysis.init(ReallocationAnalysisInput(
          std::move(reallocationOpConfigMap),
          std::move(reallocationSchedule),
          utils::getUsableL1PerCore(moduleOp),
          reallocationOffsetCapacity));

      const ReallocationAnalysisResult &reallocationResult =
          reallocationAnalysis.getResult();

      TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer,
                   "Reallocation result:\n{}", reallocationResult);

      OpBuilder builder(&getContext());
      processReallocationValues(
          reallocationResult.memReallocateValuesMap, builder);
    }

#endif
  }

private:
  static RankedTensorType
  getTensorTypeWithLayout(RankedTensorType inputType,
                          TTNNLayoutAttr outputLayout) {
    Type elementType = inputType.getElementType();

    if (!mlir::isa<mlir::quant::QuantizedType>(elementType)) {
      elementType = outputLayout.getScalarElementType();
    } else {
      auto quantizedType =
          mlir::cast<mlir::quant::QuantizedType>(elementType);
      assert(quantizedType.getStorageType() ==
                 outputLayout.getScalarElementType() &&
             "Layout scalar element type must match quantized storage type");
    }

    return RankedTensorType::get(inputType.getShape(), elementType,
                                 outputLayout);
  }

  static Value materializeToLayout(Value input, TTNNLayoutAttr outputLayout,
                                   Operation *insertBefore,
                                   llvm::StringRef locSuffix) {
    if (!input || !outputLayout || !insertBefore) {
      return {};
    }

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return {};
    }

    RankedTensorType outputType =
        getTensorTypeWithLayout(inputType, outputLayout);

    OpBuilder builder(insertBefore);
    builder.setInsertionPoint(insertBefore);

    Location loc = ttmlir::utils::appendLocationSuffix(
        insertBefore->getLoc(), locSuffix);

    // In this branch ToLayoutOp derives layout/dtype/memory configuration from
    // the result tensor encoding. Its builder only takes result type + input.
    return builder.create<ToLayoutOp>(loc, outputType, input).getResult();
  }

  static void processTransitionEdges(
      llvm::ArrayRef<analysis::TransitionEdge> transitionEdges,
      func::FuncOp func) {
    for (const analysis::TransitionEdge &transitionEdge : transitionEdges) {
      Operation *consumerOp = transitionEdge.consumerOp;
      if (!consumerOp ||
          consumerOp->getParentOfType<func::FuncOp>() != func ||
          transitionEdge.consumerOperandIndex >= consumerOp->getNumOperands()) {
        continue;
      }

      Value input =
          consumerOp->getOperand(transitionEdge.consumerOperandIndex);
      auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
      if (!inputType || !transitionEdge.consumerLayout) {
        continue;
      }

      TTNNLayoutAttr inputLayout =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(inputType.getEncoding());
      if (inputLayout == transitionEdge.consumerLayout) {
        continue;
      }

      Value converted =
          materializeToLayout(input, transitionEdge.consumerLayout, consumerOp,
                              "_viterbi_transition");
      if (!converted) {
        consumerOp->emitError()
            << "Failed to materialize Viterbi layout transition";
        continue;
      }

      // Rewrite only the exact SSA use described by TransitionEdgeAnalysis.
      consumerOp->setOperand(transitionEdge.consumerOperandIndex, converted);
    }
  }

  static void processSpillRequests(
      llvm::ArrayRef<SpillRequest> spillRequests, func::FuncOp func) {
    auto isUseAfterTrigger = [](Operation *useOp, Operation *triggerOp) {
      return useOp && triggerOp &&
             useOp->getBlock() == triggerOp->getBlock() &&
             triggerOp->isBeforeInBlock(useOp);
    };

    for (const SpillRequest &spillRequest : spillRequests) {
      Value value = spillRequest.value;
      Operation *triggerOp = spillRequest.triggerOp;

      if (!value || !triggerOp ||
          triggerOp->getParentOfType<func::FuncOp>() != func) {
        continue;
      }

      auto valueType = mlir::dyn_cast<RankedTensorType>(value.getType());
      if (!valueType) {
        continue;
      }

      TTNNLayoutAttr valueLayout =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(valueType.getEncoding());
      if (!valueLayout || valueLayout.getBufferType() != BufferType::L1) {
        continue;
      }

      // Snapshot later uses before creating the spill op so the spill op's own
      // input cannot become part of the rewrite set.
      llvm::SmallVector<OpOperand *> laterUses;
      for (OpOperand &use : value.getUses()) {
        if (isUseAfterTrigger(use.getOwner(), triggerOp)) {
          laterUses.push_back(&use);
        }
      }

      if (laterUses.empty()) {
        continue;
      }

      TTNNLayoutAttr dramLayout =
          TTNNLayoutAttr::Builder(valueLayout, valueType.getShape())
              .setBufferType(BufferType::DRAM)
              .setMemoryLayout(TensorMemoryLayout::Interleaved)
              .build();

      Value spillValue =
          materializeToLayout(value, dramLayout, triggerOp, "_spill_to_dram");
      if (!spillValue) {
        continue;
      }

      for (OpOperand *use : laterUses) {
        if (!use) {
          continue;
        }

        Operation *userOp = use->getOwner();
        if (!userOp || userOp->getBlock() != triggerOp->getBlock()) {
          continue;
        }

        // If a later conversion already exists, let it consume the spilled
        // DRAM value directly. That conversion becomes the reload.
        if (mlir::isa<ToLayoutOp, ToMemoryConfigOp, TypecastOp>(userOp)) {
          use->set(spillValue);
          continue;
        }

        // Compute users still expect the selected pre-spill layout. Insert a
        // reload immediately before that exact use.
        Value reloadValue =
            materializeToLayout(spillValue, valueLayout, userOp,
                                "_reload_from_dram");
        if (!reloadValue) {
          continue;
        }

        use->set(reloadValue);
      }
    }
  }

  static void processReallocationValues(
      const llvm::DenseMap<Operation *, llvm::DenseSet<Value>>
          &memReallocateValuesMap,
      OpBuilder &builder) {
    llvm::DenseSet<Value> alreadyReallocated;

    for (const auto &[_, values] : memReallocateValuesMap) {
      for (Value value : values) {
        if (!value || alreadyReallocated.contains(value)) {
          continue;
        }

        Operation *producerOp = value.getDefiningOp();
        if (!producerOp) {
          continue;
        }

        alreadyReallocated.insert(value);

        // Snapshot uses before creating ReallocateOp so its own input is not
        // rewritten to the newly created result.
        llvm::SmallVector<OpOperand *> uses;
        for (OpOperand &use : value.getUses()) {
          uses.push_back(&use);
        }

        builder.setInsertionPointAfter(producerOp);
        Location loc = ttmlir::utils::appendLocationSuffix(
            producerOp->getLoc(), "_reallocate");

        auto reallocateOp = builder.create<ReallocateOp>(
            loc, value.getType(), value, /*memory_config=*/nullptr);

        for (OpOperand *use : uses) {
          use->set(reallocateOp.getResult());
        }
      }
    }
  }

  void assertOverridesValid() {
    llvm::StringMap<bool> overriddenOpExists;
    llvm::StringMap<bool> overriddenConv2dOps;

    for (const auto &[opLocation, _] : overrideOutputLayout) {
      overriddenOpExists[opLocation] = false;
    }

    for (const auto &[opLocation, _] : overrideConv2dConfig) {
      overriddenOpExists[opLocation] = false;
      overriddenConv2dOps[opLocation] = true;
    }

    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](Operation *op) {
      auto nameLocation = mlir::dyn_cast<NameLoc>(op->getLoc());
      if (!nameLocation) {
        return;
      }

      StringRef opLocationName = nameLocation.getName();

      if (overriddenOpExists.contains(opLocationName)) {
        overriddenOpExists[opLocationName] = true;
      }

      if (overriddenConv2dOps.contains(opLocationName) &&
          !mlir::isa<Conv2dOp>(op)) {
        op->emitRemark()
            << "Trying to apply a Conv2d configuration override to non-Conv2d "
               "operation '"
            << op->getName() << "'. Skipping.";
      }
    });

    for (const auto &[opLocation, exists] : overriddenOpExists) {
      if (exists) {
        continue;
      }

      llvm::report_fatal_error(
          llvm::Twine("Trying to override non-existing operation: ") +
          opLocation + ". Check logs for details.");
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
