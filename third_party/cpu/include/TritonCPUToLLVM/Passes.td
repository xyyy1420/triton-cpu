#ifndef TRITONCPU_CONVERSION_PASSES
#define TRITONCPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FuncOpToLLVM : Pass<"triton-cpu-func-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert FuncOp to LLVM for CPU.";
    let description = [{}];
    let constructor = "mlir::triton::cpu::createFuncOpToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def MemoryOpToLLVM : Pass<"triton-cpu-memory-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton memory operations to LLVM for CPU.";
    let description = [{}];
    let constructor = "mlir::triton::cpu::createMemoryOpToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def GetProgramIdOpToLLVM : Pass<"triton-cpu-get-program-id-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton GetProgramId to LLVM for CPU.";
    let description = [{}];
    let constructor = "mlir::triton::cpu::createGetProgramIdOpToLLVMPass()";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect"];
}

def LowerMultiReduction : Pass<"triton-cpu-lower-multi-reduction", "mlir::triton::FuncOp"> {
    let summary = "Convert multi-dimensional reductions.";
    let description = [{}];
    let constructor = "mlir::triton::cpu::createLowerMultiReductionPass()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::triton::cpu::TritonCPUDialect",
                             "mlir::triton::TritonDialect"];
}

def AtomicOpsToLLVM : Pass<"triton-cpu-atomic-ops-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton atomic operations to LLVM.";
    let description = [{}];
    let constructor = "mlir::triton::cpu::createAtomicOpsToLLVMPass()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::triton::cpu::TritonCPUDialect",
                             "mlir::triton::TritonDialect"];
}

def DebugOpsToLLVM : Pass<"triton-cpu-debug-ops-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton debug operations (prints and asserts) to LLVM.";
    let description = [{}];
    let constructor = "mlir::triton::cpu::createDebugOpsToLLVMPass()";

    let dependentDialects = ["mlir::triton::cpu::TritonCPUDialect",
                             "mlir::triton::TritonDialect"];
}

def UkernelOpsToOneDNNLLVM : Pass<"triton-cpu-ukernels-to-onednn-llvm", "mlir::ModuleOp"> {
    let summary = "Convert ukernel operations to OneDNN LLVM runtime calls.";
    let description = [{}];
    let constructor = "mlir::triton::cpu::createUkernelOpsToOneDNNLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::cpu::TritonCPUDialect",
                             "mlir::triton::TritonDialect"];
}

def UkernelOpsToXSMMLLVM : Pass<"triton-cpu-ukernels-to-xsmm-llvm", "mlir::ModuleOp"> {
    let summary = "Convert ukernel operations to XSMM LLVM runtime calls.";
    let description = [{}];
    let constructor = "mlir::triton::cpu::createUkernelOpsToXSMMLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::cpu::TritonCPUDialect",
                             "mlir::triton::TritonDialect"];
}

def MathToVecLib : Pass<"triton-cpu-math-to-vec-lib", "mlir::ModuleOp"> {
    let summary = "Convert vector math operations to vector libm or sleef calls.";
    let description = [{
    }];
    let constructor = "mlir::triton::cpu::createMathToVecLibPass()";

    let options = [
        Option<"lib", "lib",
               "mlir::triton::cpu::VecLib", /*default*/"mlir::triton::cpu::VecLib::Sleef",
               "Library to use for vector math (libsleef or libmvec).",
               [{::llvm::cl::values(
               clEnumValN(mlir::triton::cpu::VecLib::Sleef, "sleef",
                "Use Sleef as mm lib"),
               clEnumValN(mlir::triton::cpu::VecLib::Mvec, "mvec",
                "Use Mvec as mm lib")
              )}]>,
        ListOption<"cpu_features", "cpu_features", "std::string",
             "A list of available CPU features to choose proper vector functions">,
    ];

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::triton::cpu::TritonCPUDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::func::FuncDialect",
                             "mlir::LLVM::LLVMDialect"];
}

#endif
