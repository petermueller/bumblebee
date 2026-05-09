defmodule Mix.Tasks.Bumblebee.CompareQwen3Vl do
  @shortdoc "Compares Bumblebee's Qwen3-VL forward pass against HuggingFace transformers"

  @moduledoc """
  Runs the same image+prompt through both Bumblebee's Qwen3-VL and the
  reference Hugging Face transformers implementation (via PythonX), then
  reports where the logits diverge.

  Useful when iterating on the Qwen3-VL or Qwen3.5 implementation —
  position-0 logits should match to float-precision; later positions
  show how much the implementation drifts.

  Example:

      mix bumblebee.compare_qwen3_vl
      mix bumblebee.compare_qwen3_vl --repo Qwen/Qwen3-VL-2B-Instruct \\
        --image test/fixtures/images/coco/39769.jpeg \\
        --prompt "Describe this image briefly."

  This task lives under `test/mix_tasks/` and is wired to the `:test`
  env via `preferred_envs` in `mix.exs`, so `:stb_image` and friends
  are loaded automatically. It is a development tool — not shipped
  with the published package.

  First run installs Python and the required PyPI packages (torch,
  transformers, pillow, torchvision) into PythonX's managed env via uv —
  expect a few-minutes one-time install.
  """

  use Mix.Task

  @default_repo "Qwen/Qwen3-VL-2B-Instruct"
  @default_image "test/fixtures/images/coco/39769.jpeg"
  @default_prompt "Describe this image briefly."

  @impl Mix.Task
  def run(argv) do
    {opts, _argv} =
      OptionParser.parse!(argv,
        strict: [
          repo: :string,
          image: :string,
          prompt: :string,
          show_logits: :integer,
          no_mrope: :boolean
        ]
      )

    repo = opts[:repo] || @default_repo
    image_path = opts[:image] || @default_image
    prompt_text = opts[:prompt] || @default_prompt
    show_logits = opts[:show_logits] || 16
    no_mrope? = opts[:no_mrope] || false

    Mix.Task.run("app.start")
    Nx.global_default_backend(EXLA.Backend)

    init_pythonx!()

    Mix.shell().info("=== Running PyTorch reference ===")
    py_data = run_python_reference(repo, image_path, prompt_text)

    Mix.shell().info("=== Running Bumblebee#{if no_mrope?, do: " (mRoPE DISABLED)", else: ""} ===")
    bb_data = run_bumblebee(repo, image_path, py_data, no_mrope?: no_mrope?)

    Mix.shell().info("=== Comparison ===")
    compare(py_data, bb_data, show_logits)
  end

  defp init_pythonx!() do
    Mix.shell().info("Initialising PythonX env (first run installs torch + transformers)...")

    Pythonx.uv_init("""
    [project]
    name = "bumblebee-qwen3vl-compare"
    version = "0.0.0"
    requires-python = ">=3.10,<3.14"
    dependencies = [
      "torch>=2.0",
      "torchvision",
      "transformers>=4.57",
      "pillow",
      "accelerate"
    ]
    """)
  end

  defp run_python_reference(repo, image_path, prompt_text) do
    code = """
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # PythonX hands Elixir binaries to Python as bytes; transformers wants str.
    repo = repo.decode("utf-8") if isinstance(repo, bytes) else repo
    image_path = image_path.decode("utf-8") if isinstance(image_path, bytes) else image_path
    prompt_text = prompt_text.decode("utf-8") if isinstance(prompt_text, bytes) else prompt_text

    processor = AutoProcessor.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.float32)
    model.eval()

    img = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        # Vision-only output for an apples-to-apples comparison.
        # Run the vision tower with output_hidden_states=True so we get
        # the per-block hidden states alongside the post-merger output.
        pv_typed = inputs["pixel_values"].type(model.model.visual.dtype)
        vision_output = model.model.visual(
            pv_typed,
            grid_thw=inputs["image_grid_thw"],
            output_hidden_states=True,
        )
        # hidden_states is a tuple of (num_patches, hidden_size) tensors,
        # one per block (and possibly an extra initial entry).
        per_layer_hidden = [hs.detach().cpu().tolist() for hs in vision_output.hidden_states]
        visual = vision_output.pooler_output

        out = model(**inputs, use_cache=False)

    logits = out.logits

    last = logits[:, -1, :]
    top10 = torch.topk(last, k=10, dim=-1)

    {
        "input_ids": inputs["input_ids"][0].tolist(),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "logits_shape": list(logits.shape),
        "logits_step0_first_n": logits[0, 0, :128].tolist(),
        "logits_last_first_n": logits[0, -1, :128].tolist(),
        "top10_ids": top10.indices[0].tolist(),
        "top10_vals": top10.values[0].tolist(),
        "visual_shape": list(visual.shape),
        "visual_first_token": visual[0, :].tolist(),
        "visual_last_token": visual[-1, :].tolist(),
        "visual_mid_token": visual[visual.shape[0] // 2, :].tolist(),
        # Sample pixel_values for byte-level comparison vs Bumblebee's featurizer.
        "pv_first_patch_first16": inputs["pixel_values"][0, :16].tolist(),
        "pv_mid_patch_first16": inputs["pixel_values"][inputs["pixel_values"].shape[0] // 2, :16].tolist(),
        "per_layer_hidden_count": len(per_layer_hidden),
        "per_layer_hidden_shape": list(vision_output.hidden_states[0].shape),
        # First three components of one mid token, per layer, for compact
        # dumping. Lets us track where the divergence first appears.
        "per_layer_mid_token_first3": [
            [hs[hs.shape[0] // 2, k].item() for k in range(3)]
            for hs in vision_output.hidden_states
        ],
    }
    """

    {result, _globals} =
      Pythonx.eval(code, %{
        "repo" => repo,
        "image_path" => image_path,
        "prompt_text" => prompt_text
      })

    Pythonx.decode(result)
  end

  defp run_bumblebee(repo, image_path, py_data, opts) do
    no_mrope? = Keyword.get(opts, :no_mrope?, false)

    {:ok, model_info} = Bumblebee.load_model({:hf, repo})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, repo})
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, repo})

    [t, h, w] = py_data["image_grid_thw"] |> List.first()
    model_info = Bumblebee.Multimodal.Qwen3VL.with_image_grid(model_info, t: t, h: h, w: w)

    # Optionally drop mrope_section to fall back to standard 1D rotary —
    # used as an A/B test when investigating where logit drift comes from.
    model_info =
      if no_mrope? do
        new_spec = %{model_info.spec | mrope_section: nil}
        new_model = new_spec.__struct__.model(new_spec)
        %{model_info | spec: new_spec, model: new_model}
      else
        model_info
      end

    image = StbImage.read_file!(image_path)
    image_inputs = Bumblebee.apply_featurizer(featurizer, image)

    [bt, bh, bw] = image_inputs["image_grid_thw"][[0, ..]] |> Nx.to_list()

    if {bt, bh, bw} != {t, h, w} do
      Mix.raise(
        "Featurizer grid mismatch: Bumblebee says (#{bt}, #{bh}, #{bw}) but " <>
          "PyTorch processor says (#{t}, #{h}, #{w})"
      )
    end

    input_ids = Nx.tensor([py_data["input_ids"]])
    seq_len = length(py_data["input_ids"])

    base_inputs = %{
      "input_ids" => input_ids,
      "pixel_values" => image_inputs["pixel_values"],
      "attention_mask" => Nx.broadcast(1, {1, seq_len})
    }

    # With mRoPE on, build 3-axis position ids; without it, leave the
    # slot empty and let the standard 1D default kick in.
    inputs =
      if no_mrope? do
        base_inputs
      else
        position_ids =
          Bumblebee.Multimodal.Qwen3VL.position_ids(input_ids, {t, h, w}, model_info.spec)

        Map.put(base_inputs, "position_ids", position_ids)
      end

    outputs = Axon.predict(model_info.model, model_info.params, inputs)

    last = outputs.logits[[.., -1, ..]]
    {top10_vals, top10_ids} = Nx.top_k(last, k: 10)

    # Also run the vision encoder on its own, to compare its output to
    # PyTorch's get_image_features(...). Lets us isolate whether logit
    # drift is rooted in the vision tower vs downstream processing.
    visual = run_vision_only(model_info, image_inputs["pixel_values"])

    pv = image_inputs["pixel_values"]
    {npp, _flat} = Nx.shape(pv)
    pv_first16 = pv[[0, 0..15]] |> Nx.to_flat_list()
    pv_mid16 = pv[[div(npp, 2), 0..15]] |> Nx.to_flat_list()

    %{
      "tokenizer" => tokenizer,
      "logits_step0_first_n" => outputs.logits[[.., 0, 0..127]] |> Nx.to_flat_list(),
      "logits_last_first_n" => outputs.logits[[.., -1, 0..127]] |> Nx.to_flat_list(),
      "top10_ids" => Nx.to_list(top10_ids[[0, ..]]),
      "top10_vals" => Nx.to_list(top10_vals[[0, ..]]),
      "visual" => visual,
      "pixel_values_first_patch_first16" => pv_first16,
      "pixel_values_mid_patch_first16" => pv_mid16
    }
  end

  # Build the vision encoder standalone and run it with the vision
  # subset of the loaded multimodal params. Returns three sample rows
  # of the post-merger visual embeddings (first / middle / last token).
  defp run_vision_only(model_info, pixel_values) do
    vision_spec = model_info.spec.vision_spec
    vision_model = vision_spec.__struct__.model(vision_spec)

    %Axon.ModelState{data: data} = model_info.params
    prefix = "vision_model."

    vision_data =
      for {k, v} <- data, String.starts_with?(k, prefix), into: %{} do
        {String.replace_prefix(k, prefix, ""), v}
      end

    vision_params = %Axon.ModelState{
      data: vision_data,
      parameters: vision_data |> Map.new(fn {k, v} -> {k, Map.keys(v)} end),
      state: %{},
      frozen_parameters: %{}
    }

    # Enable :output_hidden_states via Axon's global_layer_options so the
    # model emits per-block hidden state tensors.
    out =
      Axon.predict(vision_model, vision_params, %{"pixel_values" => pixel_values},
        global_layer_options: [output_hidden_states: true]
      )

    # out.hidden_state shape: {1, num_visual_tokens, out_hidden_size}
    hs = out.hidden_state
    {1, n, _hs} = Nx.shape(hs)

    # Pre-block embeddings (post-patch-embed + pos-embed), shape
    # {1, num_patches, hidden_size}. Useful for isolating whether the
    # divergence comes from the embedding stage or the transformer blocks.
    pre_block = out.pre_block_embeddings

    # Per-block hidden states (pre-merger, shape {1, num_patches, hidden_size}).
    # The encoder appends one entry per block, so this tuple has num_blocks
    # entries, indexed 0..num_blocks-1.
    per_layer_hidden_states = Tuple.to_list(out.hidden_states)

    # Prepend pre_block so our list matches PyTorch's hidden_states convention
    # (entry 0 = pre-block, entry i = post-block-(i-1)).
    full_layers = [pre_block | per_layer_hidden_states]

    per_layer_mid_token_first3 =
      Enum.map(full_layers, fn block_hs ->
        {1, num_patches, _hidden} = Nx.shape(block_hs)
        mid = div(num_patches, 2)
        for k <- 0..2, do: block_hs[[0, mid, k]] |> Nx.to_number()
      end)

    %{
      "shape" => Tuple.to_list(Nx.shape(hs)),
      "first_token" => hs[[0, 0, ..]] |> Nx.to_flat_list(),
      "mid_token" => hs[[0, div(n, 2), ..]] |> Nx.to_flat_list(),
      "last_token" => hs[[0, n - 1, ..]] |> Nx.to_flat_list(),
      "per_layer_count" => length(full_layers),
      "per_layer_mid_token_first3" => per_layer_mid_token_first3
    }
  end

  defp compare(py, bb, show_logits) do
    tokenizer = bb["tokenizer"]

    # Step 0 first-N logits
    n = show_logits

    step0_diff = max_abs_diff(py["logits_step0_first_n"], bb["logits_step0_first_n"], n)
    last_diff = max_abs_diff(py["logits_last_first_n"], bb["logits_last_first_n"], n)

    Mix.shell().info("Step 0, first #{n} logits: max abs diff = #{Float.round(step0_diff, 5)}")
    Mix.shell().info("Last,  first #{n} logits: max abs diff = #{Float.round(last_diff, 5)}")

    Mix.shell().info("\nTop-10 next tokens (last position):")
    Mix.shell().info("  rank   py id   py val   bb id   bb val   ok   token (py / bb)")

    py["top10_ids"]
    |> Enum.zip(bb["top10_ids"])
    |> Enum.with_index()
    |> Enum.each(fn {{py_id, bb_id}, i} ->
      py_v = Enum.at(py["top10_vals"], i)
      bb_v = Enum.at(bb["top10_vals"], i)
      ok = if py_id == bb_id, do: "y", else: "n"
      py_tok = Bumblebee.Tokenizer.decode(tokenizer, [py_id])
      bb_tok = Bumblebee.Tokenizer.decode(tokenizer, [bb_id])

      Mix.shell().info(
        "  #{i}     #{lpad(py_id, 6)}  #{lpad(Float.round(py_v, 2), 7)}  " <>
          "#{lpad(bb_id, 6)}  #{lpad(Float.round(bb_v, 2), 7)}   #{ok}    " <>
          "#{inspect(py_tok)} / #{inspect(bb_tok)}"
      )
    end)

    matches =
      Enum.zip(py["top10_ids"], bb["top10_ids"])
      |> Enum.count(fn {a, b} -> a == b end)

    Mix.shell().info("\nTop-10 ID agreement: #{matches} / 10")

    # Pixel-values byte equality (sanity check the featurizer)
    Mix.shell().info("\n=== Featurizer output sanity check ===")
    bb_pv = bb["pixel_values_first_patch_first16"]
    py_pv0 = py["pv_first_patch_first16"]
    py_pv_mid = py["pv_mid_patch_first16"]
    bb_pv_mid = bb["pixel_values_mid_patch_first16"]

    pv_diff_0 = max_abs_diff(py_pv0, bb_pv, 16)
    pv_diff_mid = max_abs_diff(py_pv_mid, bb_pv_mid, 16)
    Mix.shell().info("first patch, first 16: max_abs_diff = #{Float.round(pv_diff_0, 6)}")
    Mix.shell().info("mid patch,   first 16: max_abs_diff = #{Float.round(pv_diff_mid, 6)}")

    Mix.shell().info("\n=== Vision encoder output (post-merger visual embeddings) ===")
    Mix.shell().info("py shape: #{inspect(py["visual_shape"])}    bb shape: #{inspect(bb["visual"]["shape"])}")

    for {label, py_key, bb_key} <- [
          {"first token", "visual_first_token", "first_token"},
          {"mid token  ", "visual_mid_token", "mid_token"},
          {"last token ", "visual_last_token", "last_token"}
        ] do
      py_vec = py[py_key]
      bb_vec = bb["visual"][bb_key]

      diffs = Enum.zip(py_vec, bb_vec) |> Enum.map(fn {a, b} -> abs(a - b) end)
      max_diff = Enum.max(diffs)
      mean_diff = Enum.sum(diffs) / length(diffs)

      Mix.shell().info(
        "#{label}  max_abs=#{Float.round(max_diff, 6)}  mean_abs=#{Float.round(mean_diff, 6)}"
      )
    end

    # Per-block hidden states. Python's tuple may include an extra
    # initial entry (pre-block embeddings); align by trimming the
    # leading mismatch.
    py_per = py["per_layer_mid_token_first3"]
    bb_per = bb["visual"]["per_layer_mid_token_first3"]

    py_count = length(py_per)
    bb_count = length(bb_per)

    Mix.shell().info(
      "\n=== Per-layer hidden state (mid token, first 3 dims) [py: #{py_count}, bb: #{bb_count}] ==="
    )

    Mix.shell().info("  index  label              py[0]      py[1]      py[2]      bb[0]      bb[1]      bb[2]      max_diff")

    py_per
    |> Enum.zip(bb_per)
    |> Enum.with_index()
    |> Enum.each(fn {{py3, bb3}, i} ->
      diffs = Enum.zip(py3, bb3) |> Enum.map(fn {a, b} -> abs(a - b) end)
      max_diff = Enum.max(diffs)

      label =
        case i do
          0 -> "pre-block         "
          n -> "after block #{lpad(n - 1, 2)}    "
        end

      cols =
        Enum.map(py3, &Float.round(&1, 4)) ++ Enum.map(bb3, &Float.round(&1, 4))

      Mix.shell().info(
        "  #{lpad(i, 5)}  #{label}  " <>
          (cols |> Enum.map(&lpad(&1, 9)) |> Enum.join("  ")) <>
          "    #{Float.round(max_diff, 5)}"
      )
    end)
  end

  defp max_abs_diff(a, b, n) do
    a
    |> Enum.take(n)
    |> Enum.zip(Enum.take(b, n))
    |> Enum.map(fn {x, y} -> abs(x - y) end)
    |> Enum.max()
  end

  defp lpad(value, n) do
    value |> to_string() |> String.pad_leading(n)
  end
end
