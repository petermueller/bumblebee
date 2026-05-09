defmodule Bumblebee.Vision.Qwen3VLVision do
  import Nx.Defn

  alias Bumblebee.Shared

  options =
    [
      hidden_size: [
        default: 1024,
        doc: "the dimensionality of hidden layers"
      ],
      num_blocks: [
        default: 24,
        doc: "the number of Transformer blocks in the encoder"
      ],
      num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the encoder"
      ],
      intermediate_size: [
        default: 4096,
        doc:
          "the dimensionality of the intermediate layer in the transformer feed-forward network (FFN) in the encoder"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      patch_size: [
        default: 16,
        doc: "the size of the patch spatial dimensions"
      ],
      temporal_patch_size: [
        default: 2,
        doc: "the size of the patch temporal dimension (for video)"
      ],
      spatial_merge_size: [
        default: 2,
        doc: "the factor by which to merge spatial patches"
      ],
      out_hidden_size: [
        default: 2048,
        doc: "the output dimensionality after patch merger"
      ],
      num_position_embeddings: [
        default: 2304,
        doc: "the number of position embeddings"
      ],
      deepstack_visual_indexes: [
        default: [5, 11, 17],
        doc: "the encoder layer indices from which to extract DeepStack features (1-indexed)"
      ],
      activation: [
        default: :gelu_approx_tanh,
        doc: "the activation function"
      ],
      layer_norm_epsilon: [
        default: 1.0e-6,
        doc: "the epsilon used by the layer normalization layers"
      ],
      rotary_embedding_base: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      grid_t: [
        default: nil,
        doc:
          "temporal patch count of the input image/video. When set, used instead of falling back" <>
            " to a `sqrt(num_patches)` square-grid guess. For images, this is 1."
      ],
      grid_h: [
        default: nil,
        doc:
          "height patch count of the input image/video, equal to `image_height / patch_size`." <>
            " When `grid_t`, `grid_h`, `grid_w` are all set, the encoder uses them for position" <>
            " embedding interpolation, 2D rotary, and the spatial merger."
      ],
      grid_w: [
        default: nil,
        doc: "width patch count of the input image/video, equal to `image_width / patch_size`."
      ]
    ]

  @moduledoc """
  The Qwen3-VL vision encoder for processing images and video frames.

  ## Architectures

    * `:base` - the base vision encoder model

  ## Inputs

    * `"pixel_values"` - `{batch_size, num_channels, temporal, height, width}`

      Featurized image/video pixel values. For images, temporal=1.

    * `"grid_thw"` - `{batch_size, 3}`

      Grid dimensions [temporal, height, width] for each sample in the batch.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(spec) do
    # Template for pre-extracted patches.
    # If grid_t/grid_h/grid_w are configured, use them; otherwise fall back
    # to a 14x14 (224x224 / 16) square grid template.
    patch_size = spec.patch_size
    temporal_patch_size = spec.temporal_patch_size
    flattened_patch_size = spec.num_channels * temporal_patch_size * patch_size * patch_size

    num_patches =
      case {spec.grid_t, spec.grid_h, spec.grid_w} do
        {t, h, w} when is_integer(t) and is_integer(h) and is_integer(w) -> t * h * w
        _ -> 196
      end

    %{
      "pixel_values" => Nx.template({num_patches, flattened_patch_size}, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  defp inputs(spec) do
    # pixel_values from featurizer: {num_patches, channels * temporal * patch_h * patch_w}
    # This is the pre-extracted patch format like Python
    patch_size = spec.patch_size
    temporal_patch_size = spec.temporal_patch_size
    flattened_patch_size = spec.num_channels * temporal_patch_size * patch_size * patch_size
    pixel_shape = {nil, flattened_patch_size}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("pixel_values", shape: pixel_shape)
    ])
  end

  defp core(inputs, spec) do
    pixel_values = inputs["pixel_values"]

    embeddings = patch_embedding(pixel_values, spec, name: "patch_embed")
    embeddings = position_embedding(embeddings, spec, name: "pos_embed")
    encoder_outputs = encoder(embeddings, spec, name: "blocks")
    hidden_state = patch_merger(encoder_outputs.hidden_state, spec, name: "merger")

    %{
      hidden_state: hidden_state,
      pre_block_embeddings: embeddings,
      hidden_states: encoder_outputs.hidden_states,
      attentions: encoder_outputs.attentions,
      deepstack_hidden_states: encoder_outputs.deepstack_hidden_states
    }
  end

  # HF's patch embed is a Conv3d with kernel == stride == the full
  # patch volume, which collapses to a per-patch dense projection. We
  # keep the kernel param shape `{hidden, c, t, p_h, p_w}` to match
  # the saved PyTorch weight layout, then unfold it into a matmul.
  defp patch_embedding(pixel_values, spec, opts) do
    name = opts[:name]

    reshaped =
      Axon.nx(pixel_values, fn x ->
        {num_patches, _flat} = Nx.shape(x)
        Nx.reshape(
          x,
          {num_patches, spec.num_channels, spec.temporal_patch_size, spec.patch_size,
           spec.patch_size}
        )
      end)

    kernel_param =
      Axon.param(
        "kernel",
        fn _ ->
          {spec.hidden_size, spec.num_channels, spec.temporal_patch_size, spec.patch_size,
           spec.patch_size}
        end,
        initializer: kernel_initializer(spec)
      )

    bias_param =
      Axon.param(
        "bias",
        fn _ -> {spec.hidden_size} end,
        initializer: Axon.Initializers.zeros()
      )

    Axon.layer(
      fn x, kernel, bias, _opts ->
        {num_patches, c, t, h, w} = Nx.shape(x)
        {hidden_size, _, _, _, _} = Nx.shape(kernel)

        x_flat = Nx.reshape(x, {num_patches, c * t * h * w})
        k_flat = kernel |> Nx.reshape({hidden_size, c * t * h * w}) |> Nx.transpose()

        x_flat |> Nx.dot(k_flat) |> Nx.add(bias)
      end,
      [reshaped, kernel_param, bias_param],
      name: join(name, "proj"),
      op_name: :conv3d
    )
    |> Axon.nx(&Nx.new_axis(&1, 0))
  end

  defp position_embedding(embeddings, spec, opts) do
    name = opts[:name]

    # Learned position embeddings: {num_position_embeddings, hidden_size}
    # num_position_embeddings = 2304 = 48*48 (a 2D grid of positions)
    # We need to interpolate to the actual grid size using bilinear interpolation
    pos_embed_param =
      Axon.param(
        "weight",
        fn _ -> {spec.num_position_embeddings, spec.hidden_size} end,
        initializer: kernel_initializer(spec)
      )

    Axon.layer(
      fn embed, pos_embed, _opts ->
        # embed: {batch, num_patches, hidden_size}
        # pos_embed: {num_position_embeddings, hidden_size} = {2304, 1024} = {48*48, 1024}
        {_batch, num_patches, _hidden_size} = Nx.shape(embed)

        # Target grid: prefer grid_h/grid_w from spec; else fall back to a
        # square grid via sqrt(num_patches) (legacy path, only correct for
        # perfectly square images).
        {grid_h, grid_w} =
          case {spec.grid_h, spec.grid_w} do
            {h, w} when is_integer(h) and is_integer(w) -> {h, w}
            _ -> {trunc(:math.sqrt(num_patches)), trunc(:math.sqrt(num_patches))}
          end

        # Source grid size (48x48)
        src_grid_size = :math.sqrt(spec.num_position_embeddings) |> trunc()

        # Bilinear interpolation from src_grid to target grid

        # Create target grid indices (h, w may differ)
        h_idxs = Nx.linspace(0, src_grid_size - 1, n: grid_h, type: :f32)
        w_idxs = Nx.linspace(0, src_grid_size - 1, n: grid_w, type: :f32)

        # Floor and ceil indices
        h_floor = Nx.floor(h_idxs) |> Nx.as_type(:s32)
        w_floor = Nx.floor(w_idxs) |> Nx.as_type(:s32)
        h_ceil = Nx.add(h_floor, 1) |> Nx.min(src_grid_size - 1)
        w_ceil = Nx.add(w_floor, 1) |> Nx.min(src_grid_size - 1)

        # Interpolation weights
        dh = Nx.subtract(h_idxs, Nx.as_type(h_floor, :f32))
        dw = Nx.subtract(w_idxs, Nx.as_type(w_floor, :f32))

        # Reshape for broadcasting: h indices along first dim (grid_h, 1),
        # w indices along second (1, grid_w)
        h_floor_2d = Nx.reshape(h_floor, {grid_h, 1})
        h_ceil_2d = Nx.reshape(h_ceil, {grid_h, 1})
        w_floor_2d = Nx.reshape(w_floor, {1, grid_w})
        w_ceil_2d = Nx.reshape(w_ceil, {1, grid_w})

        # 4 corner indices (each broadcasts to {grid_h, grid_w})
        idx_ff = Nx.add(Nx.multiply(h_floor_2d, src_grid_size), w_floor_2d) |> Nx.flatten()
        idx_fc = Nx.add(Nx.multiply(h_floor_2d, src_grid_size), w_ceil_2d) |> Nx.flatten()
        idx_cf = Nx.add(Nx.multiply(h_ceil_2d, src_grid_size), w_floor_2d) |> Nx.flatten()
        idx_cc = Nx.add(Nx.multiply(h_ceil_2d, src_grid_size), w_ceil_2d) |> Nx.flatten()

        # Gather embeddings for all 4 corners
        emb_ff = Nx.take(pos_embed, idx_ff, axis: 0)
        emb_fc = Nx.take(pos_embed, idx_fc, axis: 0)
        emb_cf = Nx.take(pos_embed, idx_cf, axis: 0)
        emb_cc = Nx.take(pos_embed, idx_cc, axis: 0)

        # Compute bilinear weights ({grid_h, grid_w} -> flattened {num_patches, 1})
        dh_2d = Nx.reshape(dh, {grid_h, 1})
        dw_2d = Nx.reshape(dw, {1, grid_w})

        w_ff =
          Nx.multiply(Nx.subtract(1.0, dh_2d), Nx.subtract(1.0, dw_2d))
          |> Nx.flatten()
          |> Nx.reshape({num_patches, 1})

        w_fc =
          Nx.multiply(Nx.subtract(1.0, dh_2d), dw_2d)
          |> Nx.flatten()
          |> Nx.reshape({num_patches, 1})

        w_cf =
          Nx.multiply(dh_2d, Nx.subtract(1.0, dw_2d))
          |> Nx.flatten()
          |> Nx.reshape({num_patches, 1})

        w_cc = Nx.multiply(dh_2d, dw_2d) |> Nx.flatten() |> Nx.reshape({num_patches, 1})

        # Weighted sum for interpolated embeddings (in raster (row, col) order
        # over the grid_h * grid_w grid).
        interpolated_raster =
          Nx.add(
            Nx.add(
              Nx.multiply(emb_ff, w_ff),
              Nx.multiply(emb_fc, w_fc)
            ),
            Nx.add(
              Nx.multiply(emb_cf, w_cf),
              Nx.multiply(emb_cc, w_cc)
            )
          )

        # Reorder from raster (row, col) ordering to the merge-block-grouped
        # (h_block, w_block, m_h, m_w) ordering used by the featurizer / HF.
        # Match HF's `.view(h/m, m, w/m, m, -1).permute(0, 2, 1, 3, 4).flatten`.
        merge = spec.spatial_merge_size
        h_block_count = div(grid_h, merge)
        w_block_count = div(grid_w, merge)
        hidden_size = spec.hidden_size

        interpolated =
          interpolated_raster
          |> Nx.reshape({h_block_count, merge, w_block_count, merge, hidden_size})
          |> Nx.transpose(axes: [0, 2, 1, 3, 4])
          |> Nx.reshape({num_patches, hidden_size})

        # Add to embeddings (broadcast to batch dimension)
        Nx.add(embed, interpolated)
      end,
      [embeddings, pos_embed_param],
      name: name,
      op_name: :position_embedding
    )
  end

  defp encoder(embeddings, spec, opts) do
    name = opts[:name]

    # `deepstack_visual_indexes` is 0-indexed in HF (it's compared against
    # `enumerate(self.blocks)`), so use the values as-is.
    deepstack_indexes = MapSet.new(spec.deepstack_visual_indexes)

    # Qwen3-VL uses 2D spatial rotary embeddings where each patch has (row, col) position.
    # Python's rot_pos_emb computes row and col frequencies separately, then concatenates them.
    #
    # For each patch at position (row, col):
    # - First half of rotary_dim: row_position * inv_freq
    # - Second half of rotary_dim: col_position * inv_freq
    #
    # We compute 2D rotary embeddings (cos, sin) for all patches based on their grid position.
    rotary_2d =
      Axon.nx(embeddings, fn embed ->
        {_batch, seq_len, _hidden} = Nx.shape(embed)

        grid_w =
          case spec.grid_w do
            w when is_integer(w) -> w
            _ -> trunc(:math.sqrt(seq_len))
          end

        head_dim = div(spec.hidden_size, spec.num_attention_heads)
        rotary_dim = div(head_dim, 2)

        compute_2d_rotary_embedding(
          seq_len,
          grid_w,
          spec.spatial_merge_size,
          rotary_dim,
          spec.rotary_embedding_base
        )
      end)

    # Use custom transformer blocks with 2D rotary embedding
    # Since Layers.Transformer.blocks only supports 1D position-based rotary,
    # we implement vision transformer blocks directly
    vision_transformer_blocks(embeddings, rotary_2d, spec, deepstack_indexes, name)
  end

  # Compute 2D rotary embedding (cos, sin) for vision patches.
  #
  # Patches are stored in HF's merge-block-grouped order:
  # for an index `i`, decompose as (h_block, w_block, m_h, m_w) with
  #   h_block = i / (w_block_count * merge * merge)
  #   w_block = (i / (merge * merge)) rem w_block_count
  #   m_h     = (i / merge) rem merge
  #   m_w     = i rem merge
  # then row = h_block * merge + m_h, col = w_block * merge + m_w.
  #
  # Returns {cos, sin} each of shape {seq_len, rotary_dim}.
  defnp compute_2d_rotary_embedding(seq_len, grid_w, merge, rotary_dim, base) do
    # Walk patches in HF's order; recover (row, col) per index.
    positions = Nx.iota({seq_len})
    w_block_count = div(grid_w, merge)
    block_size = merge * merge
    row_block_size = w_block_count * block_size

    h_block = Nx.quotient(positions, row_block_size)
    rem1 = Nx.remainder(positions, row_block_size)
    w_block = Nx.quotient(rem1, block_size)
    rem2 = Nx.remainder(rem1, block_size)
    m_h = Nx.quotient(rem2, merge)
    m_w = Nx.remainder(rem2, merge)

    row_positions = h_block * merge + m_h
    col_positions = w_block * merge + m_w

    # Compute inverse frequencies (half rotary_dim because we split for row/col)
    half_rotary_dim = div(rotary_dim, 2)
    range = Nx.iota({half_rotary_dim}) |> Nx.multiply(2) |> Nx.divide(rotary_dim)
    inv_freq = 1.0 / Nx.pow(base, range)

    # Compute angles for rows and columns
    # row_angles: {seq_len, half_rotary_dim}
    row_angles = Nx.outer(row_positions, inv_freq)
    col_angles = Nx.outer(col_positions, inv_freq)

    # Concatenate row and col angles: {seq_len, rotary_dim}
    angles = Nx.concatenate([row_angles, col_angles], axis: -1)

    # Compute cos and sin
    cos = Nx.cos(angles)
    sin = Nx.sin(angles)

    {cos, sin}
  end

  # Custom vision transformer blocks with 2D rotary embedding
  defp vision_transformer_blocks(embeddings, rotary_2d, spec, deepstack_indexes, name) do
    head_dim = div(spec.hidden_size, spec.num_attention_heads)

    # Build blocks iteratively, collecting hidden states for deepstack
    {hidden_state, hidden_states, attentions} =
      Enum.reduce(0..(spec.num_blocks - 1), {embeddings, [], []}, fn idx,
                                                                     {hidden_state, hidden_states,
                                                                      attentions} ->
        block_name = join(name, idx)

        # Pre-norm
        normed =
          Axon.layer_norm(hidden_state,
            epsilon: spec.layer_norm_epsilon,
            name: join(block_name, "norm1")
          )

        # Self-attention with 2D rotary
        {attn_output, attn_weights} =
          vision_attention_with_2d_rotary(
            normed,
            rotary_2d,
            spec,
            head_dim,
            join(block_name, "attn")
          )

        hidden_state = Axon.add(hidden_state, attn_output)

        # FFN with pre-norm
        normed =
          Axon.layer_norm(hidden_state,
            epsilon: spec.layer_norm_epsilon,
            name: join(block_name, "norm2")
          )

        ffn_output =
          normed
          |> Axon.dense(spec.intermediate_size,
            kernel_initializer: kernel_initializer(spec),
            name: join(block_name, "mlp.fc1")
          )
          |> Layers.activation(spec.activation)
          |> Axon.dense(spec.hidden_size,
            kernel_initializer: kernel_initializer(spec),
            name: join(block_name, "mlp.fc2")
          )

        hidden_state = Axon.add(hidden_state, ffn_output)

        hidden_states = hidden_states ++ [hidden_state]
        attentions = attentions ++ [attn_weights]

        {hidden_state, hidden_states, attentions}
      end)

    # Extract and merge deepstack hidden states
    # Each deepstack feature is passed through a separate merger (same structure as main merger)
    deepstack_merged_features =
      deepstack_indexes
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.map(fn {layer_idx, merger_idx} ->
        hidden_state_at_layer =
          if layer_idx < length(hidden_states) do
            Enum.at(hidden_states, layer_idx)
          else
            List.last(hidden_states)
          end

        # Apply deepstack merger (same spatial merge + MLP as main merger)
        deepstack_merger(hidden_state_at_layer, spec, merger_idx, "deepstack_merger_list")
      end)

    %{
      hidden_state: hidden_state,
      hidden_states: Axon.container(List.to_tuple(hidden_states)),
      attentions: Axon.container(List.to_tuple(attentions)),
      deepstack_hidden_states: Axon.container(List.to_tuple(deepstack_merged_features))
    }
  end

  defp deepstack_merger(hidden_state, spec, index, name) do
    spatial_merger(hidden_state, spec,
      base_name: join(name, index),
      norm_name: "norm",
      mlp_fc1_name: "linear_fc1",
      mlp_fc2_name: "linear_fc2",
      norm_position: :post
    )
  end

  # Vision attention with 2D rotary embedding
  defp vision_attention_with_2d_rotary(hidden_state, rotary_2d, spec, head_dim, name) do
    # QKV projection (combined)
    qkv =
      Axon.dense(hidden_state, spec.hidden_size * 3,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "qkv")
      )

    # Split and reshape for multi-head attention
    {query, key, value} =
      Axon.layer(
        fn qkv, _opts ->
          {batch, seq_len, _} = Nx.shape(qkv)
          qkv_reshaped = Nx.reshape(qkv, {batch, seq_len, 3, spec.num_attention_heads, head_dim})
          qkv_transposed = Nx.transpose(qkv_reshaped, axes: [2, 0, 3, 1, 4])
          # {3, batch, heads, seq, head_dim}
          {qkv_transposed[0], qkv_transposed[1], qkv_transposed[2]}
        end,
        [qkv],
        name: join(name, "split_qkv")
      )
      |> then(fn layer ->
        q = Axon.nx(layer, fn {q, _k, _v} -> q end)
        k = Axon.nx(layer, fn {_q, k, _v} -> k end)
        v = Axon.nx(layer, fn {_q, _k, v} -> v end)
        {q, k, v}
      end)

    # Apply 2D rotary embedding to query and key
    {rotated_query, rotated_key} =
      Axon.layer(
        fn query, key, rotary_2d, _opts ->
          {cos, sin} = rotary_2d
          apply_2d_rotary_embedding(query, key, cos, sin)
        end,
        [query, key, rotary_2d],
        name: join(name, "rotary_2d")
      )
      |> then(fn layer ->
        q = Axon.nx(layer, fn {q, _k} -> q end)
        k = Axon.nx(layer, fn {_q, k} -> k end)
        {q, k}
      end)

    # Scaled dot-product attention
    scale = :math.sqrt(head_dim)

    attn_output =
      Axon.layer(
        fn query, key, value, _opts ->
          # query, key, value: {batch, heads, seq, head_dim}
          # Attention scores: {batch, heads, seq, seq}
          scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1])
          scores = Nx.divide(scores, scale)
          weights = Axon.Activations.softmax(scores, axis: -1)

          # Weighted sum: {batch, heads, seq, head_dim}
          output = Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])

          {output, weights}
        end,
        [rotated_query, rotated_key, value],
        name: join(name, "attention")
      )

    output = Axon.nx(attn_output, fn {out, _weights} -> out end)
    weights = Axon.nx(attn_output, fn {_out, weights} -> weights end)

    # Reshape and project output
    output =
      Axon.layer(
        fn x, _opts ->
          {batch, heads, seq_len, head_dim} = Nx.shape(x)
          hidden_size = heads * head_dim

          x
          |> Nx.transpose(axes: [0, 2, 1, 3])
          |> Nx.reshape({batch, seq_len, hidden_size})
        end,
        [output],
        name: join(name, "reshape_output")
      )

    output =
      Axon.dense(output, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "proj")
      )

    {output, weights}
  end

  # Apply 2D rotary embedding to query and key.
  #
  # cos, sin shape: {seq_len, head_dim/2} = `[row_freqs..., col_freqs...]`
  # query, key shape: {batch, heads, seq_len, head_dim}
  #
  # Matches HF's apply_rotary_pos_emb_vision: rotates the FULL head_dim
  # (after duplicating cos/sin to head_dim length) with rotate_half pairing
  # (k, k+head_dim/2). The duplication keeps rows paired with rows and cols
  # with cols across the rotation.
  defnp apply_2d_rotary_embedding(query, key, cos, sin) do
    # Duplicate along the freq axis so cos/sin span the full head_dim.
    cos_full = Nx.concatenate([cos, cos], axis: -1)
    sin_full = Nx.concatenate([sin, sin], axis: -1)

    # Broadcast over batch and heads: {1, 1, seq_len, head_dim}
    cos_full = cos_full |> Nx.new_axis(0) |> Nx.new_axis(0)
    sin_full = sin_full |> Nx.new_axis(0) |> Nx.new_axis(0)

    rotated_q = query * cos_full + rotate_half(query) * sin_full
    rotated_k = key * cos_full + rotate_half(key) * sin_full

    {rotated_q, rotated_k}
  end

  defnp rotate_half(x) do
    # Split in half along last dimension and swap with negation
    {batch, heads, seq, dim} = Nx.shape(x)
    half_dim = div(dim, 2)
    x1 = Nx.slice(x, [0, 0, 0, 0], [batch, heads, seq, half_dim])
    x2 = Nx.slice(x, [0, 0, 0, half_dim], [batch, heads, seq, half_dim])
    Nx.concatenate([Nx.negate(x2), x1], axis: -1)
  end

  defp patch_merger(hidden_state, spec, opts) do
    spatial_merger(hidden_state, spec,
      base_name: opts[:name],
      norm_name: "ln_q",
      mlp_fc1_name: "mlp.0",
      mlp_fc2_name: "mlp.2",
      norm_position: :pre
    )
  end

  # Shared spatial-merge head for both the main patch merger and the
  # DeepStack mergers. Each merge block of `spatial_merge_size^2`
  # consecutive patches gets flattened into one token, then projected
  # through `linear_fc1 / activation / linear_fc2`. Patches arrive in
  # HF's (h_block, w_block, m_h, m_w) order so the merge is a plain
  # reshape — no transpose. Norm placement is the only thing that
  # differs between the two callers (pre-merge for the main merger,
  # post-merge for DeepStack).
  defp spatial_merger(hidden_state, spec, opts) do
    base_name = Keyword.fetch!(opts, :base_name)
    norm_name = Keyword.fetch!(opts, :norm_name)
    mlp_fc1_name = Keyword.fetch!(opts, :mlp_fc1_name)
    mlp_fc2_name = Keyword.fetch!(opts, :mlp_fc2_name)
    norm_position = Keyword.fetch!(opts, :norm_position)

    merge_size = spec.spatial_merge_size * spec.spatial_merge_size
    mlp_input_size = spec.hidden_size * merge_size

    norm = fn x ->
      Axon.layer_norm(x, epsilon: spec.layer_norm_epsilon, name: join(base_name, norm_name))
    end

    reshape =
      &Axon.nx(&1, fn x ->
        {batch, num_patches, hidden} = Nx.shape(x)
        Nx.reshape(x, {batch, div(num_patches, merge_size), merge_size * hidden})
      end)

    merged =
      case norm_position do
        :pre -> hidden_state |> norm.() |> reshape.()
        :post -> hidden_state |> reshape.() |> norm.()
      end

    merged
    |> Axon.dense(mlp_input_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(base_name, mlp_fc1_name)
    )
    |> Layers.activation(spec.activation)
    |> Axon.dense(spec.out_hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(base_name, mlp_fc2_name)
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    # Support loading from the entire Qwen3VL configuration
    def load(spec, %{"model_type" => "qwen3_vl", "vision_config" => data}) do
      load(spec, data)
    end

    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          num_blocks: {"depth", number()},
          num_attention_heads: {"num_heads", number()},
          num_channels: {"in_channels", number()},
          patch_size: {"patch_size", number()},
          temporal_patch_size: {"temporal_patch_size", number()},
          spatial_merge_size: {"spatial_merge_size", number()},
          activation: {"hidden_act", activation()},
          initializer_scale: {"initializer_range", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      # Handle both embed_dim (Qwen2-VL) and hidden_size (Qwen3-VL)
      hidden_size = data["hidden_size"] || data["embed_dim"] || spec.hidden_size
      opts = Keyword.put(opts, :hidden_size, hidden_size)

      # Compute derived values
      # intermediate_size from config or computed as hidden_size * mlp_ratio (default mlp_ratio = 4)
      mlp_ratio = Map.get(data, "mlp_ratio", 4)
      intermediate_size = data["intermediate_size"] || hidden_size * mlp_ratio

      # out_hidden_size is typically the text model's hidden_size
      # If not specified, it comes from the parent config or defaults
      out_hidden_size = Map.get(data, "out_hidden_size", spec.out_hidden_size)

      opts =
        opts
        |> Keyword.put(:intermediate_size, intermediate_size)
        |> Keyword.put(:out_hidden_size, out_hidden_size)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        # Patch embedding - keep 3D conv kernel as-is
        # PyTorch Conv3d weight shape: {out_channels, in_channels, temporal, h, w} = {1024, 3, 2, 16, 16}
        # Our custom layer expects the same shape
        "patch_embed.proj" => %{
          "kernel" => {
            [{"visual.patch_embed.proj", "weight"}],
            fn [kernel] ->
              # Keep in PyTorch format: {out_channels, in_channels, t, h, w}
              kernel
            end
          },
          "bias" => {
            [{"visual.patch_embed.proj", "bias"}],
            fn [bias] -> bias end
          }
        },
        # Learned position embeddings
        "pos_embed" => "visual.pos_embed",
        # Transformer blocks - using custom 2D rotary attention
        "blocks.{n}.norm1" => "visual.blocks.{n}.norm1",
        "blocks.{n}.attn.qkv" => "visual.blocks.{n}.attn.qkv",
        "blocks.{n}.attn.proj" => "visual.blocks.{n}.attn.proj",
        "blocks.{n}.norm2" => "visual.blocks.{n}.norm2",
        "blocks.{n}.mlp.fc1" => "visual.blocks.{n}.mlp.linear_fc1",
        "blocks.{n}.mlp.fc2" => "visual.blocks.{n}.mlp.linear_fc2",
        # Patch merger - Qwen3VL uses linear_fc1/fc2/norm naming
        "merger.ln_q" => "visual.merger.norm",
        "merger.mlp.0" => "visual.merger.linear_fc1",
        "merger.mlp.2" => "visual.merger.linear_fc2",
        # DeepStack mergers - same structure as main merger
        "deepstack_merger_list.{n}.norm" => "visual.deepstack_merger_list.{n}.norm",
        "deepstack_merger_list.{n}.linear_fc1" => "visual.deepstack_merger_list.{n}.linear_fc1",
        "deepstack_merger_list.{n}.linear_fc2" => "visual.deepstack_merger_list.{n}.linear_fc2"
      }
    end
  end
end
