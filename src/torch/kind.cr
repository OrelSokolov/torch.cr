# Scalar kinds and devices, mirroring at::ScalarType / the shim's int encoding.

module Torch
  # Values must match at::ScalarType.
  enum Kind
    Uint8    =  0
    Int8     =  1
    Int16    =  2
    Int32    =  3
    Int64    =  4
    Float16  =  5
    Float32  =  6
    Float64  =  7
    Complex32  =  8
    Complex64  =  9
    Complex128 = 10
    Bool     = 11
    BFloat16 = 12

    def bytesize : ::Int32
      case self
      in .uint8?, .int8?, .bool?       then 1
      in .int16?, .float16?, .b_float16? then 2
      in .int32?, .float32?, .complex32? then 4
      in .int64?, .float64?, .complex64? then 8
      in .complex128? then 16
      end
    end

    def self.of(type : T.class) : self forall T
      case type
      when ::UInt8.class   then Kind::Uint8
      when ::Int8.class    then Kind::Int8
      when ::Int16.class   then Kind::Int16
      when ::Int32.class   then Kind::Int32
      when ::Int64.class   then Kind::Int64
      when ::Float32.class then Kind::Float32
      when ::Float64.class then Kind::Float64
      when ::Bool.class    then Kind::Bool
      else
        raise Error.new("unsupported Crystal type #{type}")
      end
    end
  end

  struct Device
    getter type : Type
    getter index : Int32

    enum Type
      Cpu
      Cuda
      Mps
      Vulkan
    end

    def initialize(@type : Type, @index : Int32 = 0)
    end

    def self.cpu : Device
      new(Type::Cpu)
    end

    def self.cuda(index : Int32 = 0) : Device
      new(Type::Cuda, index)
    end

    # Wire encoding shared with the shim: CPU = -1, MPS = -2, Vulkan = -3,
    # CUDA:<i> = i.
    def self.from_c_int(value : Int32) : Device
      case value
      when -1 then cpu
      when -2 then new(Type::Mps)
      when -3 then new(Type::Vulkan)
      else         new(Type::Cuda, value)
      end
    end

    def to_c_int : Int32
      case type
      in .cpu?     then -1
      in .mps?     then -2
      in .vulkan?  then -3
      in .cuda?    then index
      end
    end

    def to_s(io : IO) : Nil
      case type
      in .cpu?    then io << "cpu"
      in .cuda?   then io << "cuda:" << index
      in .mps?    then io << "mps"
      in .vulkan? then io << "vulkan"
      end
    end
  end
end

# Per-type kind constants, used by Tensor typed helpers.
struct UInt8
  KIND = Torch::Kind::Uint8
end

struct Int8
  KIND = Torch::Kind::Int8
end

struct Int16
  KIND = Torch::Kind::Int16
end

struct Int32
  KIND = Torch::Kind::Int32
end

struct Int64
  KIND = Torch::Kind::Int64
end

struct Float32
  KIND = Torch::Kind::Float32
end

struct Float64
  KIND = Torch::Kind::Float64
end

struct Bool
  KIND = Torch::Kind::Bool
end
