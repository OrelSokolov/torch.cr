require "./spec_helper"

describe Torch::Tensor do
  it "round-trips f32 data" do
    t = Torch::Tensor.of([1.0_f32, 2.5_f32, -3.0_f32], [3])
    t.shape.should eq [3]
    t.kind.float32?.should be_true
    t.contiguous?.should be_true
    t.to_f32.should eq [1.0_f32, 2.5_f32, -3.0_f32]
  end

  it "round-trips i64 data with 2d shape" do
    t = Torch::Tensor.of(Slice[10_i64, 20_i64, 30_i64, 40_i64], [2, 2])
    t.shape.should eq [2, 2]
    t.strides.should eq [2, 1]
    t.to_i64.should eq [10, 20, 30, 40]
    t[1].to_i64.should eq [30, 40]
  end

  it "creates zeros" do
    t = Torch::Tensor.zeros([2, 3], Torch::Kind::Float64)
    t.size.should eq 6
    t.to_f64.should eq Array.new(6, 0.0)
  end

  it "fills and reads scalar values" do
    t = Torch::Tensor.zeros([1], Torch::Kind::Float64)
    t.fill!(2.5)
    t.item.should eq 2.5
    t.value_at([0_i64]).should eq 2.5

    ti = Torch::Tensor.zeros([2, 2], Torch::Kind::Int64)
    ti[[0, 1]] = 42_i64
    ti.value_i64_at([0_i64, 1_i64]).should eq 42
  end

  it "clones and copies" do
    a = Torch::Tensor.of([1.0_f32, 2.0_f32])
    b = a.shallow_clone
    c = Torch::Tensor.zeros([2], Torch::Kind::Float32)
    c.copy_from(b)
    c.to_f32.should eq [1.0_f32, 2.0_f32]
  end

  it "saves and loads" do
    path = "/tmp/torch_cr_spec_tensor.pt"
    a = Torch::Tensor.of([1.0_f64, 2.0_f64, 3.0_f64])
    a.save(path)
    b = Torch::Tensor.load(path)
    b.to_f64.should eq [1.0, 2.0, 3.0]
    File.delete(path)
  end

  it "reports device and threads" do
    t = Torch::Tensor.of([1.0_f32])
    t.device.type.cpu?.should be_true
    Torch.num_threads.should be > 0
  end

  it "prints to a string" do
    t = Torch::Tensor.of([1.0_f32, 2.0_f32])
    t.to_s.should contain "CPUFloatType"
  end
end
