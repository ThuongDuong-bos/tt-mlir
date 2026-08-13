// SPDX-FileCopyrightText: (c) 2026 BOS Semiconductors
//  
// SPDX-License-Identifier: Apache-2.0  
  
#include "operations/data_movement/reallocate.h"  
#include "tt/runtime/detail/common/logger.h"  
#include "tt/runtime/detail/ttnn/ttnn.h"  
  
#include "tt/runtime/detail/ttnn/operations/utils.h"  
#include "tt/runtime/detail/ttnn/utils.h"  
  
namespace tt::runtime::ttnn::operations::data_movement {  
void run(const ::tt::target::ttnn::ReallocateOp *op, ProgramContext &context) {  
  ProgramTensorPool &tensorPool = context.getTensorPool();  
  
  const ::ttnn::Tensor &inputTensor =  
      tensorPool.getTTNNTensorAndValidate(op->in());  
  
  std::optional<::ttnn::MemoryConfig> memoryConfig =  
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memory_config());  
  
  ::ttnn::Tensor out = ::ttnn::reallocate(inputTensor, memoryConfig);  
  
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);  
}  
} // namespace tt::runtime::ttnn::operations::data_movement
