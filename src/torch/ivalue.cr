# Safe wrapper around torch::jit::IValue — boxed argument/result of
# TorchScript module calls.

class Torch::IValue
  getter c_ivalue : TorchSys::CIValue

  def initialize(@c_ivalue : TorchSys::CIValue)
  end

  def finalize
    TorchSys.ati_free(@c_ivalue)
  end

  enum Tag
    None        =  0
    Tensor      =  1
    Double      =  2
    Int         =  3
    Bool        =  4
    Tuple       =  5
    IntList     =  6
    DoubleList  =  7
    BoolList    =  8
    String      =  9
    TensorList  = 10
    List        = 12
    GenericDict = 13
    Object      = 14
  end

  def self.wrap(c : TorchSys::CIValue) : self
    iv = new(c)
    Torch.check!
    iv
  end

  # -- constructors ----------------------------------------------------------

  def self.none : IValue
    wrap TorchSys.ati_none
  end

  def self.of(tensor : Tensor) : IValue
    wrap TorchSys.ati_tensor(tensor.c_tensor)
  end

  def self.of(value : Int64) : IValue
    wrap TorchSys.ati_int(value)
  end

  def self.of(value : Int32) : IValue
    wrap TorchSys.ati_int(value.to_i64)
  end

  def self.of(value : Float64) : IValue
    wrap TorchSys.ati_double(value)
  end

  def self.of(value : Float32) : IValue
    wrap TorchSys.ati_double(value.to_f64)
  end

  def self.of(value : Bool) : IValue
    wrap TorchSys.ati_bool(value ? 1 : 0)
  end

  def self.of(value : String) : IValue
    wrap TorchSys.ati_string(value.to_unsafe)
  end

  def self.of(value : Array(Int64)) : IValue
    wrap TorchSys.ati_int_list(value.to_unsafe, value.size)
  end

  def self.of(value : Array(Float64)) : IValue
    wrap TorchSys.ati_double_list(value.to_unsafe, value.size)
  end

  def self.of(value : Array(Bool)) : IValue
    buf = Slice(UInt8).new(value.size) { |i| value[i] ? 1_u8 : 0_u8 }
    wrap TorchSys.ati_bool_list(buf.to_unsafe, buf.size)
  end

  def self.of(value : Array(String)) : IValue
    buf = value.map(&.to_unsafe)
    wrap TorchSys.ati_string_list(buf.to_unsafe, buf.size)
  end

  def self.of(value : Array(Tensor)) : IValue
    buf = value.map(&.c_tensor)
    wrap TorchSys.ati_tensor_list(buf.to_unsafe, buf.size)
  end

  def self.of(value : Array(IValue)) : IValue
    buf = value.map(&.c_ivalue)
    wrap TorchSys.ati_generic_list(buf.to_unsafe, buf.size)
  end

  def self.tuple(values : Array(IValue)) : IValue
    buf = values.map(&.c_ivalue)
    wrap TorchSys.ati_tuple(buf.to_unsafe, buf.size)
  end

  def self.device(device : Device) : IValue
    wrap TorchSys.ati_device(device.to_c_int)
  end

  # -- accessors -------------------------------------------------------------

  def tag : Tag
    t = TorchSys.ati_tag(@c_ivalue)
    Torch.check!
    Tag.new(t)
  end

  def as_tensor : Tensor
    Tensor.new(TorchSys.ati_to_tensor(@c_ivalue)).tap { Torch.check! }
  end

  def as_i64 : Int64
    TorchSys.ati_to_int(@c_ivalue).tap { Torch.check! }
  end

  def as_f64 : Float64
    TorchSys.ati_to_double(@c_ivalue).tap { Torch.check! }
  end

  def as_bool : Bool
    (TorchSys.ati_to_bool(@c_ivalue) != 0).tap { Torch.check! }
  end

  def as_string : String
    ptr = TorchSys.ati_to_string(@c_ivalue)
    Torch.check!
    String.new(ptr).tap { LibC.free(ptr.as(Void*)) }
  end

  def length : Int32
    TorchSys.ati_length(@c_ivalue).tap { Torch.check! }
  end

  def as_int_list : Array(Int64)
    buf = Slice(Int64).new(length)
    TorchSys.ati_to_int_list(@c_ivalue, buf.to_unsafe, buf.size)
    Torch.check!
    buf.to_a
  end

  def as_double_list : Array(Float64)
    buf = Slice(Float64).new(length)
    TorchSys.ati_to_double_list(@c_ivalue, buf.to_unsafe, buf.size)
    Torch.check!
    buf.to_a
  end

  def as_bool_list : Array(Bool)
    buf = Slice(UInt8).new(length)
    TorchSys.ati_to_bool_list(@c_ivalue, buf.to_unsafe, buf.size)
    Torch.check!
    buf.map { |b| b != 0 }
  end

  def as_tensor_list : Array(Tensor)
    buf = Pointer(TorchSys::CTensor).malloc(length)
    TorchSys.ati_to_tensor_list(@c_ivalue, buf, length)
    Torch.check!
    buf.to_slice(length).map { |c| Tensor.new(c) }
  end

  # Tuple or generic list of IValues.
  def as_list : Array(IValue)
    n = length
    buf = Pointer(TorchSys::CIValue).malloc(n)
    if tag.tuple?
      TorchSys.ati_to_tuple(@c_ivalue, buf, n)
    else
      TorchSys.ati_to_generic_list(@c_ivalue, buf, n)
    end
    Torch.check!
    buf.to_slice(n).map { |c| IValue.new(c) }.to_a
  end

  # Dict as a flat array of key/value IValue pairs.
  def as_dict_pairs : Array(IValue)
    n = length
    buf = Slice(TorchSys::CIValue).new(n * 2)
    TorchSys.ati_to_generic_dict(@c_ivalue, buf.to_unsafe, n)
    Torch.check!
    buf.to_slice(n).map { |c| IValue.new(c) }.to_a
  end

  def clone : IValue
    IValue.wrap TorchSys.ati_clone(@c_ivalue)
  end
end
