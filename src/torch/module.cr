# Safe wrapper around torch::jit::script::Module (TorchScript).

class Torch::Module
  getter c_module : TorchSys::CModule

  def initialize(@c_module : TorchSys::CModule)
  end

  def finalize
    TorchSys.atm_free(@c_module)
  end

  def self.load(filename : String, device : Device | Nil = nil) : self
    c = if device
      TorchSys.atm_load_on_device(filename.to_unsafe, device.to_c_int)
    else
      TorchSys.atm_load(filename.to_unsafe)
    end
    Torch.check!
    new(c)
  end

  def eval : self
    TorchSys.atm_eval(@c_module)
    Torch.check!
    self
  end

  def train : self
    TorchSys.atm_train(@c_module)
    Torch.check!
    self
  end

  def to(device : Device, kind : Kind | Nil = nil, non_blocking : Bool = false) : self
    TorchSys.atm_to(@c_module, device.to_c_int,
      (kind || Kind::Float32).to_i32, non_blocking ? 1 : 0)
    Torch.check!
    self
  end

  def save(filename : String) : Nil
    TorchSys.atm_save(@c_module, filename.to_unsafe)
    Torch.check!
  end

  # Convenience path for tensor-only modules: raises unless the module returns
  # a single tensor.
  def forward(tensors : Array(Tensor)) : Tensor
    buf = tensors.map(&.c_tensor)
    c = TorchSys.atm_forward(@c_module, buf.to_unsafe, buf.size)
    Torch.check!
    Tensor.new(c)
  end

  def forward(*tensors : Tensor) : Tensor
    forward(tensors.to_a)
  end

  # General path through IValues, for modules with non-tensor inputs/outputs.
  def forward(values : Array(IValue)) : IValue
    buf = values.map(&.c_ivalue)
    c = TorchSys.atm_forward_(@c_module, buf.to_unsafe, buf.size)
    Torch.check!
    IValue.new(c)
  end

  def call_method(name : String, values : Array(IValue)) : IValue
    buf = values.map(&.c_ivalue)
    c = TorchSys.atm_method_(@c_module, name.to_unsafe, buf.to_unsafe, buf.size)
    Torch.check!
    IValue.new(c)
  end

  # Yields each named parameter (name, tensor). The tensors are owned by the
  # wrapper and freed by GC as usual.
  def each_named_parameter(&block : String, Tensor ->) : Nil
    callback = ->(data : Void*, name : UInt8*, c_tensor : TorchSys::CTensor) {
      b = Box(Proc(String, Tensor, Nil)).unbox(data)
      b.call(String.new(name), Tensor.new(c_tensor))
    }
    TorchSys.atm_named_parameters(@c_module, Box.box(block), callback)
    Torch.check!
  end
end
