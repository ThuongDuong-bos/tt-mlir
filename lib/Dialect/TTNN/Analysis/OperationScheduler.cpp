// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OperationScheduler.h"

#include "ttmlir/Scheduler/Scheduler.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/FunctionTypes.h"

#include <iostream>

namespace mlir::tt::ttnn::analysis {

void OperationScheduler::analysisImplementation() {

    TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer, 
                "OperationScheduler Policy: {}", 
                analysisInput.schedulePolicy);

    switch (analysisInput.schedulePolicy) { 

    case SchedulePolicy::DepthFirstSearch: {

        op->walk([&](func::FuncOp func) {
        if (!ttmlir::utils::isForwardDeviceFunc(func)) {
            TTMLIR_TRACE(ttmlir::LogComponent::ViterbiOptimizer, 
                        "Skipped, not forward device func\n: {}", 
                        func->getName().getStringRef());
            return;
        }

        mlir::tt::scheduler::Scheduler scheduler(&func);
        llvm::SmallVector<mlir::Operation *> schedulableOps;
        mlir::Operation *currentOp = nullptr;

        while (scheduler.hasUnscheduledOps()) {

            // tt::scheduler::Scheduler already is DF search
            schedulableOps = scheduler.getSchedulableOps();
            currentOp = schedulableOps[0];
            assert(currentOp != nullptr && "Opereration from scheduler is nullptr!");

            // Schedule the operation
            scheduler.scheduleOp(currentOp);

            TTMLIR_DEBUG(ttmlir::LogComponent::ViterbiOptimizer, 
                "Scheduled op: {}", 
                currentOp->getName().getStringRef());
        }

        (analysisResult.schedule)[func] = scheduler.getSchedule();
        });

        break;
    }

    llvm_unreachable("Unhandled SchedulePolicy");

    }
}
} // namespace mlir::tt::ttnn::analysis