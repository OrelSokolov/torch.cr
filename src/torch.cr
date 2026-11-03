# torch.cr — libtorch bindings for Crystal.
require "./torch/sys"
require "./torch/kind"
require "./torch/tensor"
require "./torch/ivalue"
require "./torch/module"

module Torch
  class Error < Exception
  end

  # Returns and clears the last shim error on this thread (set by PROTECT).
  def self.last_error : String?
    ptr = TorchSys.get_and_reset_last_err
    ptr ? String.new(ptr) : nil
  end

  # Raises Torch::Error if the shim recorded an error for the last call.
  def self.check! : Nil
    if msg = last_error
      raise Error.new(msg)
    end
  end

  def self.manual_seed(seed : Int64) : Nil
    TorchSys.at_manual_seed(seed)
  end

  def self.num_threads : Int32
    TorchSys.at_get_num_threads
  end

  def self.num_threads=(n : Int32) : Nil
    TorchSys.at_set_num_threads(n)
  end

  def self.cuda_available? : Bool
    TorchSys.atc_cuda_is_available != 0
  end

  def self.cuda_device_count : Int32
    TorchSys.atc_cuda_device_count
  end

  def self.cuda_manual_seed(seed : UInt64) : Nil
    TorchSys.atc_manual_seed(seed)
  end

  def self.cuda_synchronize(device_index : Int32 = 0) : Nil
    TorchSys.atc_synchronize(device_index.to_i64)
  end
end
