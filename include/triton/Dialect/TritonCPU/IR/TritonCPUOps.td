#ifndef TRITONCPU_OPS
#define TRITONCPU_OPS

include "triton/Dialect/TritonCPU/IR/TritonCPUDialect.td"
include "triton/Dialect/TritonCPU/IR/TritonCPUTypes.td"
include "triton/Dialect/TritonCPU/IR/TritonCPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class TTC_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonCPU_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

//
// External Elementwise op
//
def TTC_ExternElementwiseOp : TTC_Op<"extern_elementwise", [Elementwise,
                                                            SameOperandsAndResultEncoding,
                                                            SameVariadicOperandSize,
                                                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

    let description = [{
        Similar to TT_ExternElementwiseOp, but only supports calls to libsleef at the moment.
        The string "%s(numel)" in $symbol will be interpolated with the number of elements of
        the vector argument(s).
    }];

    let arguments = (ins Variadic<TTC_Type>:$srcs, StrAttr:$symbol, BoolAttr:$pure);

    let results = (outs TTC_Type:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, $result)";
}

def TTC_ExtractMemRefOp : TTC_Op<"extract_memref", [NoMemoryEffect]> {
  let summary = "Extract base memref from a block pointer";

  let description = [{
    Extract base memref from a block pointer. It covers whole base tensor memory,
    not only the block referenced. Base pointer, shape, and strides are used
    in the resulting memref. Offsets and block shape are ignored.

  }];

  let arguments = (ins TT_TensorPtr:$src);

  let results = (outs AnyRankedOrUnrankedMemRef:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTC_ExtractIndicesOp : TTC_Op<"extract_indices", [NoMemoryEffect]> {
  let summary = "Extract indices from a block pointer.";

  let description = [{
    Extract indices that can be used to access the block using its base memref.
    Indices are supposed to be used for vector loads/stores with the base
    memref extracted from the same block pointer.
  }];

  let arguments = (ins TT_TensorPtr:$src);

  let results = (outs Variadic<Index>:$result);

  let builders = [
    OpBuilder<(ins "Value":$src)>
  ];

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTC_PtrToMemRefOp : TTC_Op<"ptr_to_memref", [NoMemoryEffect]> {
  let summary = "Build a memref for a pointer.";

  let description = [{
    Build memref with static shape, offset, strides, and specified base pointer.
  }];

  let arguments = (ins TT_Ptr:$src);

  let results = (outs AnyStaticShapeMemRef:$result);

  let hasCanonicalizer = 0;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;


def TTC_LoadOp : TTC_Op<"load", [
  MemoryEffects<[MemRead<GlobalMemory>]>,
]> {
  let summary = "Load from a memref to triton tensor";

  let description = [{
    Operation to allow load from allocated temporary buffer to triton tensor.
  }];

  let arguments = (ins AnyMemRef:$src);

  let results = (outs TT_Type:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTC_StoreOp : TTC_Op<"store", [
  MemoryEffects<[MemWrite<GlobalMemory>]>,
]> {
  let summary = "Store triton tensor to memref";

  let description = [{
    Operation to allow store triton tensor to allocated temporary buffer.
  }];

  let arguments = (
    ins
    TT_Type:$src,
    AnyMemRef:$dst
  );

  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

def TTC_PrintOp : TTC_Op<"print", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Print at most a single scalar or vector (converted from tensor) on each line";

  let description = [{
    For converting tensor types to vector types.
    It only takes a single scalar or vector (tensor) element.
  }];

  let arguments = (ins
    StrAttr:$prefix,
    BoolAttr:$hex,
    Variadic<AnyTypeOf<[TT_Float, TT_Int, TT_Ptr, AnyRankedOrUnrankedMemRef]>>:$val,
    DenseI32ArrayAttr:$isSigned
  );

  let assemblyFormat = [{
    $prefix attr-dict (`:` $val^ `:` type($val))?
  }];

  let hasVerifier = 1;
}

def TTC_AssertOp : TTC_Op<"assert", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "For correctness checking";
  let description = [{
    Takes a condition tensor, a message string, a file string, a function string, and a line number.
    If the condition is false, the message is printed, and the program is aborted.
  }];
  let arguments = (ins I1:$condition, StrAttr:$message);
  let assemblyFormat = "$condition `,` $message attr-dict `:` type($condition)";
}

def TTC_DotOp : TTC_Op<"dot", [Pure,
                               DeclareOpInterfaceMethods<InferTypeOpInterface>,
                               TypesMatchWith<"result's type matches accumulator's type",
                                              "d", "c", "$_self">]> {
    let summary = "dot";

    let description = [{Same as tt.dot but on vectors.}];

    let arguments = (
      ins
      TTC_Vector:$a,
      TTC_Vector:$b,
      TTC_Vector:$c,
      DefaultValuedAttr<TT_InputPrecisionAttr, "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
      DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc
    );

    let results = (outs TTC_Vector:$d);

    let assemblyFormat = [{
      $a`,` $b`,` $c (`,` `inputPrecision` `=` $inputPrecision^)? attr-dict `:`
      type($a) `*` type($b) `->` type($d)
    }];
}

def TTC_BrgemmCreate : TTC_Op<"brgemm_create", [NoMemoryEffect]> {
  let summary = "Crete ukernels handles";

  let description = [{For creation of ukernels, that can be used to replace op with dot-like sematnics}];

  // M, N, K_k, batch_size, lda, ldb, ldc, dtypeA, dtypeB, dtypeC
  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$M,
    AnyTypeOf<[AnyInteger, Index]>:$N,
    AnyTypeOf<[AnyInteger, Index]>:$K_k,
    AnyTypeOf<[AnyInteger, Index]>:$batch_size,
    AnyTypeOf<[AnyInteger, Index]>:$lda,
    AnyTypeOf<[AnyInteger, Index]>:$ldb,
    AnyTypeOf<[AnyInteger, Index]>:$ldc,
    AnyTypeOf<[AnyInteger, Index]>:$stepA,
    AnyTypeOf<[AnyInteger, Index]>:$stepB,
    // TODO: Maybe Use properties
    TypeAttr:$dtypeA,
    TypeAttr:$dtypeB,
    TypeAttr:$dtypeC,
    I1:$skipPacking
  );

  let results = (outs Index:$result);
}

def TTC_BrgemmExecute : TTC_Op<"brgemm_execute",
    [MemoryEffects<[MemRead<GlobalMemory>,
                    MemWrite<GlobalMemory>]>]> {
  let summary = "Call ukernel with existing handle for passed ops";

  let arguments = (ins
    Index:$brgemm_kernel_hash,
    Arg<AnyMemRef, "Lhs/In buffer data ptr", [MemRead]>:$A_ptr,
    Arg<AnyMemRef, "Rhs/Weight buffer data ptr", [MemRead]>:$B_ptr,
    Arg<AnyMemRef, "Result/Acc buffer data ptr", [MemWrite]>:$C_ptr,
    AnyTypeOf<[AnyInteger, Index]>:$stepA,
    AnyTypeOf<[AnyInteger, Index]>:$stepB,
    AnyTypeOf<[AnyInteger, Index]>:$blockedBsize,
    AnyTypeOf<[AnyInteger, Index]>:$numBatches,
    I1:$skipPacking
  );
}

#endif
