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

    # Capture intermediate values inside block 0's attention via forward
    # hooks. We sample at idx 150 (mid-visual) to localise where drift vs
    # Bumblebee enters during the first decoder block.
    _captures = {}

    def _make_hook(name):
        def _hook(_mod, _args, output):
            t = output[0] if isinstance(output, tuple) else output
            _captures[name] = t.detach().cpu()
        return _hook

    _attn0 = model.model.language_model.layers[0].self_attn
    _ln0 = model.model.language_model.layers[0].input_layernorm
    _hooks = [
        _ln0.register_forward_hook(_make_hook("input_layernorm")),
        _attn0.q_proj.register_forward_hook(_make_hook("q_proj")),
        _attn0.k_proj.register_forward_hook(_make_hook("k_proj")),
        _attn0.v_proj.register_forward_hook(_make_hook("v_proj")),
        _attn0.q_norm.register_forward_hook(_make_hook("q_norm")),
        _attn0.k_norm.register_forward_hook(_make_hook("k_norm")),
    ]

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

        out = model(**inputs, use_cache=False, output_hidden_states=True)

    for _h in _hooks:
        _h.remove()

    # q_proj/k_proj/v_proj outputs are flat (1, seq, num_heads * head_dim).
    # q_norm / k_norm see the reshaped (1, seq, num_heads, head_dim).
    _block0 = {
        name: _captures[name][0, 150, :].flatten().tolist()
        for name in ("input_layernorm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm")
    }

    logits = out.logits
    text_hidden_states = out.hidden_states  # tuple, one per text block (+ embedding entry)

    # Compute the 3-axis position_ids HF uses internally so we can verify
    # Bumblebee's `position_ids/3` matches.
    _ids = inputs["input_ids"]
    _mm_type = ((_ids == 151655).int() + (_ids == 151656).int() * 2)
    _pos_ids, _ = model.model.get_rope_index(_ids, _mm_type, image_grid_thw=inputs["image_grid_thw"])
    if _pos_ids.ndim == 2:
        _pos_ids = _pos_ids[None, ...].expand(3, _pos_ids.shape[0], -1)

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
        "per_layer_mid_token_first3": [
            [hs[hs.shape[0] // 2, k].item() for k in range(3)]
            for hs in vision_output.hidden_states
        ],
        # Text-decoder per-block hidden states. shape (1, seq, hidden).
        # We sample the LAST token's first 3 dims so we can spot exactly
        # which block first diverges from the reference.
        "text_per_block_count": len(text_hidden_states),
        "text_per_block_last_token_first3": [
            [hs[0, -1, k].item() for k in range(3)]
            for hs in text_hidden_states
        ],
        # Also sample a visual-position (mid-visual, around index 150).
        "text_per_block_mid_visual_first3": [
            [hs[0, 150, k].item() for k in range(3)]
            for hs in text_hidden_states
        ],
        # Early text token (idx 2, before the image block). Causal
        # attention means this token only sees positions 0-2, none of
        # which are visual — so it isolates per-block math from any
        # cross-modal effect.
        "text_per_block_early_text_first3": [
            [hs[0, 2, k].item() for k in range(3)]
            for hs in text_hidden_states
        ],
        "text_total_count_with_embedding": len(text_hidden_states),
        "position_ids_head": _pos_ids[:, 0, :8].tolist(),
        "position_ids_tail": _pos_ids[:, 0, -8:].tolist(),
        # Pre-block hidden state, full vectors at a few sampled positions.
        # Used for full-dim max_abs comparison vs Bumblebee.
        "text_pre_block_full": {
            "early_text_2": text_hidden_states[0][0, 2, :].tolist(),
            "mid_visual_150": text_hidden_states[0][0, 150, :].tolist(),
            "last": text_hidden_states[0][0, -1, :].tolist(),
        },
        # Post-block-0 hidden state, full vectors at the same positions.
        "text_post_block_0_full": {
            "early_text_2": text_hidden_states[1][0, 2, :].tolist(),
            "mid_visual_150": text_hidden_states[1][0, 150, :].tolist(),
            "last": text_hidden_states[1][0, -1, :].tolist(),
        },
        # Per-stage outputs inside block 0 attention at idx 150.
        "block0_attn_idx150": _block0,
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

    outputs =
      Axon.predict(model_info.model, model_info.params, inputs,
        global_layer_options: [output_hidden_states: true]
      )

    last = outputs.logits[[.., -1, ..]]
    {top10_vals, top10_ids} = Nx.top_k(last, k: 10)

    # Per-text-block hidden states (last token, first 3 dims). One entry
    # per decoder block, plus a final post-norm entry that the multimodal
    # model appends — so length matches PyTorch's tuple of (per-block + final).
    text_per_block_last_token_first3 =
      outputs.hidden_states
      |> Tuple.to_list()
      |> Enum.map(fn hs ->
        for k <- 0..2, do: hs[[0, -1, k]] |> Nx.to_number()
      end)

    text_per_block_mid_visual_first3 =
      outputs.hidden_states
      |> Tuple.to_list()
      |> Enum.map(fn hs ->
        for k <- 0..2, do: hs[[0, 150, k]] |> Nx.to_number()
      end)

    text_per_block_early_text_first3 =
      outputs.hidden_states
      |> Tuple.to_list()
      |> Enum.map(fn hs ->
        for k <- 0..2, do: hs[[0, 2, k]] |> Nx.to_number()
      end)

    [pre_block | rest_blocks] = Tuple.to_list(outputs.hidden_states)
    [post_block_0 | _] = rest_blocks

    text_pre_block_full = %{
      "early_text_2" => pre_block[[0, 2, ..]] |> Nx.to_flat_list(),
      "mid_visual_150" => pre_block[[0, 150, ..]] |> Nx.to_flat_list(),
      "last" => pre_block[[0, -1, ..]] |> Nx.to_flat_list()
    }

    text_post_block_0_full = %{
      "early_text_2" => post_block_0[[0, 2, ..]] |> Nx.to_flat_list(),
      "mid_visual_150" => post_block_0[[0, 150, ..]] |> Nx.to_flat_list(),
      "last" => post_block_0[[0, -1, ..]] |> Nx.to_flat_list()
    }

    block0_attn_idx150 = block0_attn_intermediates(model_info, pre_block, 150)

    # Also run the vision encoder on its own, to compare its output to
    # PyTorch's get_image_features(...). Lets us isolate whether logit
    # drift is rooted in the vision tower vs downstream processing.
    visual = run_vision_only(model_info, image_inputs["pixel_values"])

    pv = image_inputs["pixel_values"]
    {npp, _flat} = Nx.shape(pv)
    pv_first16 = pv[[0, 0..15]] |> Nx.to_flat_list()
    pv_mid16 = pv[[div(npp, 2), 0..15]] |> Nx.to_flat_list()

    pos_ids_head =
      if no_mrope?,
        do: nil,
        else: position_ids_sample(inputs["position_ids"], 0..7)

    pos_ids_tail =
      if no_mrope?,
        do: nil,
        else: position_ids_sample(inputs["position_ids"], -8..-1)

    %{
      "tokenizer" => tokenizer,
      "logits_step0_first_n" => outputs.logits[[.., 0, 0..127]] |> Nx.to_flat_list(),
      "logits_last_first_n" => outputs.logits[[.., -1, 0..127]] |> Nx.to_flat_list(),
      "top10_ids" => Nx.to_list(top10_ids[[0, ..]]),
      "top10_vals" => Nx.to_list(top10_vals[[0, ..]]),
      "visual" => visual,
      "pixel_values_first_patch_first16" => pv_first16,
      "pixel_values_mid_patch_first16" => pv_mid16,
      "text_per_block_last_token_first3" => text_per_block_last_token_first3,
      "text_per_block_mid_visual_first3" => text_per_block_mid_visual_first3,
      "text_per_block_early_text_first3" => text_per_block_early_text_first3,
      "text_pre_block_full" => text_pre_block_full,
      "text_post_block_0_full" => text_post_block_0_full,
      "block0_attn_idx150" => block0_attn_idx150,
      "position_ids_head" => pos_ids_head,
      "position_ids_tail" => pos_ids_tail
    }
  end

  defp position_ids_sample(position_ids, range) do
    for axis <- 0..2 do
      Nx.to_list(position_ids[[axis, 0, range]])
    end
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

  # Replicate block 0's attention path (input_layernorm → q/k/v projection
  # → q/k norm) using extracted params, so we can compare each stage
  # against PyTorch hooks at a specific position.
  defp block0_attn_intermediates(model_info, pre_block, idx) do
    %Axon.ModelState{data: data} = model_info.params
    text_spec = model_info.spec.text_spec
    eps = text_spec.layer_norm_epsilon

    prefix = "text_model.decoder.blocks.0."
    ln_w = data[prefix <> "self_attention_norm"]["weight"]
    q_w = data[prefix <> "self_attention.query"]["kernel"]
    k_w = data[prefix <> "self_attention.key"]["kernel"]
    v_w = data[prefix <> "self_attention.value"]["kernel"]
    qn_w = data[prefix <> "self_attention.query_norm"]["weight"]
    kn_w = data[prefix <> "self_attention.key_norm"]["weight"]

    normed = rms_norm_normalization(pre_block, ln_w, eps)
    q_proj = Nx.dot(normed, q_w)
    k_proj = Nx.dot(normed, k_w)
    v_proj = Nx.dot(normed, v_w)

    head_dim = text_spec.attention_head_size
    {1, seq, _} = Nx.shape(q_proj)
    q_reshaped = Nx.reshape(q_proj, {1, seq, text_spec.num_attention_heads, head_dim})
    k_reshaped = Nx.reshape(k_proj, {1, seq, text_spec.num_key_value_heads, head_dim})

    q_normed = rms_norm_normalization(q_reshaped, qn_w, eps)
    k_normed = rms_norm_normalization(k_reshaped, kn_w, eps)

    %{
      "input_layernorm" => normed[[0, idx, ..]] |> Nx.to_flat_list(),
      "q_proj" => q_proj[[0, idx, ..]] |> Nx.to_flat_list(),
      "k_proj" => k_proj[[0, idx, ..]] |> Nx.to_flat_list(),
      "v_proj" => v_proj[[0, idx, ..]] |> Nx.to_flat_list(),
      "q_norm" => q_normed[[0, idx, .., ..]] |> Nx.to_flat_list(),
      "k_norm" => k_normed[[0, idx, .., ..]] |> Nx.to_flat_list()
    }
  end

  # rms_norm with :normalization upcast (matches Bumblebee.Layers default).
  defp rms_norm_normalization(x, weight, eps) do
    in_type = Nx.type(x)
    x_f32 = Nx.as_type(x, :f32)
    variance = x_f32 |> Nx.pow(2) |> Nx.mean(axes: [-1], keep_axes: true)
    normed = x_f32 |> Nx.multiply(Nx.rsqrt(Nx.add(variance, eps))) |> Nx.as_type(in_type)
    Nx.multiply(normed, weight)
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

    if bb["position_ids_head"] do
      Mix.shell().info("\n=== position_ids head (first 8 tokens) ===")
      print_position_ids_table(py["position_ids_head"], bb["position_ids_head"])
      Mix.shell().info("\n=== position_ids tail (last 8 tokens) ===")
      print_position_ids_table(py["position_ids_tail"], bb["position_ids_tail"])
    end

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

    # Text-decoder per-block hidden state. The two stacks use different
    # output_hidden_states conventions:
    #   HF: [input, post-0, post-1, ..., post-(N-2), post-norm]    length N+1
    #   BB: [input, post-0, post-1, ..., post-(N-1), post-norm]    length N+2
    # HF records the pre-block state for each layer (= post-state of the
    # previous layer) plus a final post-norm, so it never explicitly
    # records the last block's pre-norm output. Drop the leading input
    # from both, and drop bb's extra post-(N-1) entry, so each row
    # compares post-block-i for i = 0..N-2; the last row compares
    # post-final-norm.
    print_text_block_table = fn label, py_key, bb_key ->
      [py_pre | py_rest] = py[py_key]
      [bb_pre | bb_rest] = bb[bb_key]
      {bb_init, [_post_last_block, bb_post_norm]} = Enum.split(bb_rest, -2)

      py_rows = [py_pre] ++ py_rest
      bb_rows = [bb_pre] ++ bb_init ++ [bb_post_norm]

      n = length(py_rows)

      Mix.shell().info(
        "\n=== Text decoder per-block (#{label}, first 3 dims) [py: #{n}, bb: #{length(bb_rows)}] ==="
      )

      Mix.shell().info("  block   py[0]      py[1]      py[2]      bb[0]      bb[1]      bb[2]      max_diff")

      py_rows
      |> Enum.zip(bb_rows)
      |> Enum.with_index()
      |> Enum.each(fn {{py3, bb3}, i} ->
        diffs = Enum.zip(py3, bb3) |> Enum.map(fn {a, b} -> abs(a - b) end)
        max_diff = Enum.max(diffs)
        cols = Enum.map(py3, &Float.round(&1, 4)) ++ Enum.map(bb3, &Float.round(&1, 4))

        label =
          cond do
            i == 0 -> "pre  "
            i == n - 1 -> "norm "
            true -> lpad(i - 1, 5)
          end

        Mix.shell().info(
          "  #{label}  " <>
            (cols |> Enum.map(&lpad(&1, 9)) |> Enum.join("  ")) <>
            "    #{Float.round(max_diff, 5)}"
        )
      end)
    end

    print_text_block_table.("last token (text)", "text_per_block_last_token_first3",
      "text_per_block_last_token_first3")

    print_text_block_table.("mid visual (idx 150)", "text_per_block_mid_visual_first3",
      "text_per_block_mid_visual_first3")

    print_text_block_table.("early text (idx 2)", "text_per_block_early_text_first3",
      "text_per_block_early_text_first3")

    Mix.shell().info("\n=== Block 0 attention intermediates at idx 150 (full vector) ===")
    Mix.shell().info("  stage              max_abs_diff       py_max_abs       bb_max_abs")

    for stage <- ~w(input_layernorm q_proj k_proj v_proj q_norm k_norm) do
      py_v = py["block0_attn_idx150"][stage]
      bb_v = bb["block0_attn_idx150"][stage]

      diff_max =
        Enum.zip(py_v, bb_v) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max()

      py_mag = py_v |> Enum.map(&abs/1) |> Enum.max()
      bb_mag = bb_v |> Enum.map(&abs/1) |> Enum.max()

      Mix.shell().info(
        "  #{String.pad_trailing(stage, 16)} " <>
          "  #{Float.round(diff_max, 6) |> to_string() |> String.pad_leading(12)}     " <>
          "#{Float.round(py_mag, 6) |> to_string() |> String.pad_leading(12)}     " <>
          "#{Float.round(bb_mag, 6) |> to_string() |> String.pad_leading(12)}"
      )
    end

    Mix.shell().info("\n=== Full-vector hidden state diff (all 2048 dims) ===")
    Mix.shell().info("  position             pre-block          post-block-0")

    for pos <- ["early_text_2", "mid_visual_150", "last"] do
      py_pre = py["text_pre_block_full"][pos]
      bb_pre = bb["text_pre_block_full"][pos]
      py_post = py["text_post_block_0_full"][pos]
      bb_post = bb["text_post_block_0_full"][pos]

      pre_max =
        Enum.zip(py_pre, bb_pre) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max()

      post_max =
        Enum.zip(py_post, bb_post) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max()

      Mix.shell().info(
        "  #{String.pad_trailing(pos, 20)} " <>
          "max=#{Float.round(pre_max, 6) |> to_string() |> String.pad_leading(10)}      " <>
          "max=#{Float.round(post_max, 6) |> to_string() |> String.pad_leading(10)}"
      )
    end
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

  defp print_position_ids_table(py_axes, bb_axes) do
    Mix.shell().info("  axis  py                                              bb")

    for {label, axis} <- [{"t", 0}, {"h", 1}, {"w", 2}] do
      py_row = Enum.at(py_axes, axis)
      bb_row = Enum.at(bb_axes, axis)
      match = if py_row == bb_row, do: "=", else: "≠"

      Mix.shell().info(
        "  #{label}    #{inspect(py_row) |> String.pad_trailing(46)}  #{inspect(bb_row)}   #{match}"
      )
    end
  end
end
