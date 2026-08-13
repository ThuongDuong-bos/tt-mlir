// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline -o %t %s  
// RUN: FileCheck %s --input-file=%t  
// RUN: ttmlir-opt --split-input-file --ttnn-to-emitc -o %t.emitc %t  
// RUN: FileCheck %s --input-file=%t.emitc --check-prefix=EMITC  
// RUN: ttmlir-translate --emitc-to-cpp -o %t.cpp %t.emitc  
  
#dram = #ttnn.buffer_type<dram>  
#l1 = #ttnn.buffer_type<l1>  
  
#dram_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #dram>, <interleaved>>  
#l1_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #l1>, <interleaved>>  
  
// -----  
// Test basic reallocate operation  
module attributes {} {  
  func.func @test_reallocate(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout> {  
    // CHECK: "ttnn.reallocate"  
    %0 = "ttnn.reallocate"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>  
    return %0 : tensor<32x32xf32, #l1_layout>  
  }  
}  
  
// -----  
// Test reallocate without memory config  
module attributes {} {  
  func.func @test_reallocate_no_config(%arg0: tensor<64x64xbf16, #l1_layout>) -> tensor<64x64xbf16, #dram_layout> {  
    // CHECK: "ttnn.reallocate"  
    %0 = "ttnn.reallocate"(%arg0) : (tensor<64x64xbf16, #l1_layout>) -> tensor<64x64xbf16, #dram_layout>  
    return %0 : tensor<64x64xbf16, #dram_layout>  
  }  
}  
  
// EMITC-LABEL: @test_reallocate  
// EMITC: emitc.call_opaque "ttnn::reallocate"  
// EMITC-SAME: args = [0 : index, #emitc.opaque<"std::optional<::ttnn::MemoryConfig>{::ttnn::MemoryConfig{.memory_layout = ::ttnn::TensorMemoryLayout::Interleaved, .buffer_type = ::ttnn::BufferType::L1}}">]  
  
// EMITC-LABEL: @test_reallocate_no_config    
// EMITC: emitc.call_opaque "ttnn::reallocate"  
// EMITC-SAME: args = [0 : index, #emitc.opaque<"std::nullopt">]
