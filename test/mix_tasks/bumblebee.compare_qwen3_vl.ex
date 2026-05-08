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

    %{
      "tokenizer" => tokenizer,
      "logits_step0_first_n" => outputs.logits[[.., 0, 0..127]] |> Nx.to_flat_list(),
      "logits_last_first_n" => outputs.logits[[.., -1, 0..127]] |> Nx.to_flat_list(),
      "top10_ids" => Nx.to_list(top10_ids[[0, ..]]),
      "top10_vals" => Nx.to_list(top10_vals[[0, ..]])
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
