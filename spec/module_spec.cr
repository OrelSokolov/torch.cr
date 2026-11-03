require "./spec_helper"

describe Torch::Module do
  it "loads and runs a traced tensor->tensor module" do
    m = Torch::Module.load(fixture_path("double.pt")).eval
    x = Torch::Tensor.of(GOLDEN["x"].as_a.map(&.as_a.map(&.as_f32)), [2, 3])
    y = m.forward(x)
    expect_grid(y.to_f32.map(&.to_f64).each_slice(3).map(&.to_a).to_a, GOLDEN["double"].as_a)
  end

  it "runs a scripted module with mixed args and tuple result" do
    m = Torch::Module.load(fixture_path("multi.pt")).eval
    x = Torch::Tensor.of(GOLDEN["x"].as_a.map(&.as_a.map(&.as_f32)), [2, 3])
    out = m.forward([Torch::IValue.of(x), Torch::IValue.of(3)])
    out.tag.tuple?.should be_true
    y, rowsum, ret = out.as_list

    expect_grid(y.as_tensor.to_f32.map(&.to_f64).each_slice(3).map(&.to_a).to_a, GOLDEN["multi_y"].as_a)
    rowsum.as_tensor.to_f32.map(&.round(6)).should eq(
      GOLDEN["multi_rowsum"].as_a.map(&.as_f).map(&.round(6)))
    ret.as_i64.should eq GOLDEN["multi_ret_int"].as_i64
  end

  it "exposes named parameters" do
    m = Torch::Module.load(fixture_path("linear.pt")).eval
    params = [] of {String, Array(Int64)}
    m.each_named_parameter do |name, tensor|
      params << {name, tensor.shape}
    end
    params.should eq [{"lin.weight", [2, 3]}]

    x = Torch::Tensor.of(GOLDEN["x"].as_a.map(&.as_a.map(&.as_f32)), [2, 3])
    y = m.forward(x)
    expect_grid(y.to_f32.map(&.to_f64).each_slice(2).map(&.to_a).to_a, GOLDEN["linear"].as_a)
  end

  it "raises Torch::Error on a missing module" do
    expect_raises Torch::Error do
      Torch::Module.load(fixture_path("does_not_exist.pt"))
    end
  end
end
