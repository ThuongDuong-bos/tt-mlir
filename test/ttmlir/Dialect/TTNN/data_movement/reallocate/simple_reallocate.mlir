// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline -o %t %s  
// RUN: FileCheck %s --input-file=%t  
  
#dram = #ttnn.buffer_type<dram>  
#l1 = #ttnn.buffer_type<l1>  
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #dram>, <interleaved>>  
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #l1>, <interleaved>>  
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64xbf16, #dram>, <interleaved>>  
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x128xf32, #l1>, <interleaved>>

// -----  
// Test basic reallocate operation  
module attributes {} {  
  func.func @test_reallocate(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1> {  
    // CHECK: "ttnn.reallocate"  
    %0 = "ttnn.reallocate"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>}> : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>  
    return %0 : tensor<32x32xf32, #ttnn_layout1>  
  }  
}  
  
// -----  
// Test reallocate without memory config (should use output's layout)  
module attributes {} {  
  func.func @test_reallocate_no_config(%arg0: tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout2> {  
    // CHECK: "ttnn.reallocate"  
    %0 = "ttnn.reallocate"(%arg0) : (tensor<64x64xbf16, #ttnn_layout>) -> tensor<64x64xbf16, #ttnn_layout2>  
    return %0 : tensor<64x64xbf16, #ttnn_layout2>  
  }  
}  
  
// -----  
// Test reallocate in optimization context  
module attributes {} {  
  func.func @test_reallocate_optimization(%arg0: tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout3> {  
    %0 = "ttnn.relu"(%arg0) : (tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout>  
    // CHECK: "ttnn.reallocate"  
    %1 = "ttnn.reallocate"(%0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout3>  
    return %1 : tensor<128x128xf32, #ttnn_layout3>  
  }  
}
