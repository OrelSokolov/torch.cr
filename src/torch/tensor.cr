# Safe wrapper around torch::Tensor.

class Torch::Tensor
  getter c_tensor : TorchSys::CTensor
  getter? owned : Bool

  def initialize(@c_tensor : TorchSys::CTensor, @owned : Bool = true)
  end

  def finalize
    @owned && TorchSys.at_free(@c_tensor)
  end

  # -- construction ----------------------------------------------------------

  # Creates a tensor from a contiguous slice of numbers, inferring the kind
  # from the element type.
  def self.of(values : Slice(T), shape : Array(Int64) | Array(Int32)) : self forall T
    kind = Kind.of(T)
    dims = shape.map(&.to_i64)
    c = TorchSys.at_tensor_of_data(
      values.to_unsafe.as(Void*), dims.to_unsafe, dims.size.to_u64!,
      kind.bytesize.to_u64!, kind.to_i32)
    Torch.check!
    new(c)
  end

  def self.of(values : Array(T), shape) : self forall T
    slice = Slice(T).new(values.size) { |i| values[i] }
    of(slice, shape)
  end

  def self.of(values : Array(Array(T)), shape : Array(Int64) | Array(Int32)) : self forall T
    of(values.flatten, shape)
  end

  def self.of(values : Array(T) | Slice(T)) : self forall T
    of(values, [values.size])
  end

  # Zero-filled tensor of the given kind and shape.
  def self.zeros(shape, kind : Kind = Kind::Float32) : self
    numel = shape.product
    buf = Pointer(UInt8).malloc(numel * kind.bytesize)
    buf.clear(numel * kind.bytesize)
    dims = shape.map(&.to_i64)
    c = TorchSys.at_tensor_of_data(buf, dims.to_unsafe, dims.size.to_u64!,
      kind.bytesize.to_u64!, kind.to_i32)
    Torch.check!
    new(c)
  end

  def self.load(filename : String) : self
    c = TorchSys.at_load(filename.to_unsafe)
    Torch.check!
    new(c)
  end

  # -- introspection ---------------------------------------------------------

  def defined? : Bool
    TorchSys.at_defined(@c_tensor) != 0
  end

  def sparse? : Bool
    TorchSys.at_is_sparse(@c_tensor) != 0
  end

  def contiguous? : Bool
    TorchSys.at_is_contiguous(@c_tensor) != 0
  end

  def dim : Int64
    TorchSys.at_dim(@c_tensor).to_i64
  end

  def shape : Array(Int64)
    buf = Slice(Int64).new(dim.to_i32)
    TorchSys.at_shape(@c_tensor, buf.to_unsafe)
    buf.to_a
  end

  def strides : Array(Int64)
    buf = Slice(Int64).new(dim.to_i32)
    TorchSys.at_stride(@c_tensor, buf.to_unsafe)
    buf.to_a
  end

  def size : Int64
    shape.product
  end

  def kind : Kind
    Kind.new(TorchSys.at_scalar_type(@c_tensor))
  end

  def device : Device
    Device.from_c_int(TorchSys.at_device(@c_tensor))
  end

  def data_ptr : Void*
    TorchSys.at_data_ptr(@c_tensor).tap { Torch.check! }
  end

  # -- element access --------------------------------------------------------

  def [](index : Int) : Tensor
    c = TorchSys.at_get(@c_tensor, index.to_i32)
    Torch.check!
    Tensor.new(c)
  end

  def fill!(value : Float64) : self
    TorchSys.at_fill_double(@c_tensor, value)
    Torch.check!
    self
  end

  def fill!(value : Int64) : self
    TorchSys.at_fill_int64(@c_tensor, value)
    Torch.check!
    self
  end

  # Scalar value of a single-element tensor (or at the given indexes).
  def item : Float64
    value_at(nil)
  end

  def value_at(indexes : Array(Int64) | Nil) : Float64
    ptr = indexes ? indexes.to_unsafe : Pointer(Int64).null
    v = TorchSys.at_double_value_at_indexes(@c_tensor, ptr, indexes ? indexes.size : 0)
    Torch.check!
    v
  end

  def value_i64_at(indexes : Array(Int64) | Nil) : Int64
    ptr = indexes ? indexes.to_unsafe : Pointer(Int64).null
    v = TorchSys.at_int64_value_at_indexes(@c_tensor, ptr, indexes ? indexes.size : 0)
    Torch.check!
    v
  end

  def []=(indexes : Array(Int32), value : Float64) : Nil
    TorchSys.at_set_double_value_at_indexes(@c_tensor, indexes.to_unsafe, indexes.size, value)
    Torch.check!
  end

  def []=(indexes : Array(Int32), value : Int64) : Nil
    TorchSys.at_set_int64_value_at_indexes(@c_tensor, indexes.to_unsafe, indexes.size, value)
    Torch.check!
  end

  # -- data IO ---------------------------------------------------------------

  # Copies the tensor data into `out`; the buffer must be at least `size`
  # elements long and match the tensor's element size. Non-CPU tensors are
  # moved to CPU first.
  def copy_to(buf : Slice(T)) : Nil forall T
    if sizeof(T) != kind.bytesize
      raise Error.new("element size mismatch: tensor #{kind} (#{kind.bytesize} bytes), buffer #{T} (#{sizeof(T)} bytes)")
    end
    TorchSys.at_copy_data(@c_tensor, buf.to_unsafe.as(Void*), buf.size.to_u64!, sizeof(T).to_u64!)
    Torch.check!
  end

  def to_f32 : Array(Float32)
    buf = Slice(Float32).new(size.to_i32!)
    copy_to(buf)
    buf.to_a
  end

  def to_f64 : Array(Float64)
    buf = Slice(Float64).new(size.to_i32!)
    copy_to(buf)
    buf.to_a
  end

  def to_i64 : Array(Int64)
    buf = Slice(Int64).new(size.to_i32!)
    copy_to(buf)
    buf.to_a
  end

  def copy_from(other : Tensor) : self
    TorchSys.at_copy_(@c_tensor, other.c_tensor)
    Torch.check!
    self
  end

  # -- autograd --------------------------------------------------------------

  def requires_grad? : Bool
    TorchSys.at_requires_grad(@c_tensor) != 0
  end

  def backward(keep_graph : Bool = false, create_graph : Bool = false) : Nil
    TorchSys.at_backward(@c_tensor, keep_graph ? 1 : 0, create_graph ? 1 : 0)
    Torch.check!
  end

  # -- misc ------------------------------------------------------------------

  def shallow_clone : Tensor
    c = TorchSys.at_shallow_clone(@c_tensor)
    Torch.check!
    Tensor.new(c)
  end

  def save(filename : String) : Nil
    TorchSys.at_save(@c_tensor, filename.to_unsafe)
    Torch.check!
  end

  def to_s(io : IO) : Nil
    ptr = TorchSys.at_to_string(@c_tensor, 80)
    Torch.check!
    io << String.new(ptr)
    LibC.free(ptr.as(Void*)) if ptr
  end

  def print : Nil
    TorchSys.at_print(@c_tensor)
    Torch.check!
  end

  def inspect(io : IO) : Nil
    io << "Torch::Tensor("
    kind.to_s(io)
    io << ", "
    shape.to_s(io)
    io << ", "
    device.to_s(io)
    io << ") "
    to_s(io)
  end
end
