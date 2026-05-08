defmodule Bumblebee.Multimodal.Qwen3VLTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  @tiny_repo "roulis/tiny-random-Qwen3VLForConditionalGeneration"

  test ":for_conditional_generation" do
    # Tiny model created with /tmp/create_tiny_qwen3vl_v4.py (transformers 4.57.3):
    # - text_config: vocab_size=1024, hidden_size=64, num_hidden_layers=2,
    #                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    #                intermediate_size=128
    # - vision_config: depth=2, hidden_size=32, num_heads=4, intermediate_size=64,
    #                  out_hidden_size=64, patch_size=14, spatial_merge_size=2,
    #                  temporal_patch_size=2
    #
    # Reference values from /tmp/generate_reference_v2.py (seed=0):
    # model = Qwen3VLForConditionalGeneration.from_pretrained(model_path)
    # outputs = model(input_ids=torch.tensor([[10, 20, 30, 40, 50, 60, 0, 0]]),
    #                 attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]))
    # outputs.logits[0, 0:3, 0:5].numpy()

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, @tiny_repo})

    assert %Bumblebee.Multimodal.Qwen3VL{architecture: :for_conditional_generation} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 8, 1024}

    # Reference values from Python (transformers 4.57.3)
    assert_all_close(
      outputs.logits[[.., 0..2, 0..4]],
      Nx.tensor([
        [
          [0.0410, 0.0745, -0.0977, 0.0099, 0.2705],
          [-0.0504, 0.1776, -0.0481, -0.0269, 0.1630],
          [-0.1887, 0.0889, -0.1113, -0.1756, 0.0805]
        ]
      ]),
      atol: 1.0e-4
    )
  end

  test "vision path: non-square grid forwards without crashing" do
    # Configure vision_spec for a 4x6 (h x w) patch grid — exercises the
    # rectangular-grid code path that real Qwen3-VL inputs always hit.
    {:ok, spec} = Bumblebee.load_spec({:hf, @tiny_repo})
    vision_spec = Bumblebee.configure(spec.vision_spec, grid_t: 1, grid_h: 4, grid_w: 6)
    spec = %{spec | vision_spec: vision_spec}

    assert {:ok, %{model: model, params: params}} =
             Bumblebee.load_model({:hf, @tiny_repo}, spec: spec)

    # 4*6 = 24 patches, each flattened to channels * temporal * patch * patch
    # = 3 * 2 * 14 * 14 = 1176.
    flat = vision_spec.num_channels * vision_spec.temporal_patch_size *
             vision_spec.patch_size * vision_spec.patch_size

    pixel_values = Nx.broadcast(0.1, {24, flat})

    # input_ids contain no image_token_id (default 151_655) so the visual
    # embedding scatter is a no-op; we just want the vision encoder to run.
    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
      "pixel_values" => pixel_values
    }

    outputs = Axon.predict(model, params, inputs)
    assert Nx.shape(outputs.logits) == {1, 8, 1024}
    # Sanity: no NaN / Inf in the logits.
    finite_count = outputs.logits |> Nx.is_infinity() |> Nx.sum() |> Nx.to_number()
    assert finite_count == 0
    nan_count = outputs.logits |> Nx.is_nan() |> Nx.sum() |> Nx.to_number()
    assert nan_count == 0
  end
end
