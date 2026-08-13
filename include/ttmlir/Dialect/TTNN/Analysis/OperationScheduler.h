// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0
  
#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPERATIONSCHEDULER_H  
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPERATIONSCHEDULER_H  
  
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"  
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/Value.h"  
#include "llvm/ADT/DenseMap.h"  
#include "llvm/ADT/DenseSet.h"  
#include "llvm/ADT/SmallVector.h"  
#include "llvm/Support/FormatVariadic.h"

#include <vector>

namespace mlir::tt::ttnn::analysis {  
  
/// Scheduling policies for operation scheduling  
enum class SchedulePolicy { DepthFirstSearch };  

/// Input specification for operation scheduler  
struct OperationSchedulerInput {  
  SchedulePolicy schedulePolicy = SchedulePolicy::DepthFirstSearch; 
  
  OperationSchedulerInput() = default;
  OperationSchedulerInput(SchedulePolicy policy) :
        schedulePolicy(policy) {}
  
  bool operator==(const OperationSchedulerInput &rhs) const {  
    return schedulePolicy == rhs.schedulePolicy;  
  }
  
  bool operator!=(const OperationSchedulerInput &rhs) const {  
    return !(*this == rhs);  
  }  
};  

/// Result specification for operation scheduler  
struct OperationSchedulerResult {  
  llvm::DenseMap<mlir::func::FuncOp, llvm::SmallVector<mlir::Operation *>> schedule;  
  
  OperationSchedulerResult() = default;  
    
  explicit OperationSchedulerResult(const llvm::DenseMap<mlir::func::FuncOp, llvm::SmallVector<mlir::Operation *>> &schedule)  
      : schedule(schedule) {}  
};  

/// Operation scheduler that maintains dependency graph between TTIR operations  
/// and determines the order in which operations are scheduled.  
class OperationScheduler : public TTNNAnalysis<OperationSchedulerInput,  
                                              OperationSchedulerResult> {  
  
private:  
  
    /// Implementation of the scheduling analysis  
    void analysisImplementation() override;  
    
    /// Apply overrides (no overrides for scheduling)  
    bool applyOverrides() override { return false; }  

public:  
    OperationScheduler(mlir::Operation *op) : TTNNAnalysis(op) {}  
};
  
} // namespace mlir::tt::ttnn::analysis 
  

namespace llvm {

template <>
struct format_provider<mlir::tt::ttnn::analysis::SchedulePolicy> {
  static void format(
      const mlir::tt::ttnn::analysis::SchedulePolicy &policy,
      llvm::raw_ostream &os,
      llvm::StringRef options) {

    using Policy = mlir::tt::ttnn::analysis::SchedulePolicy;

    switch (policy) {
      case Policy::DepthFirstSearch:
        os << "DepthFirstSearch";
        break;
    }
  }
};

} // namespace llvm

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPERATIONSCHEDULER_H