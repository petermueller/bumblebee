defmodule Bumblebee.Vision.Qwen3VLFeaturizer do
  alias Bumblebee.Shared

  options = [
    resize: [
      default: true,
      doc: "whether to resize the input to the given `:size`"
    ],
    size: [
      default: %{height: 448, width: 448},
      doc: """
      the size to resize the input to, given as `%{height: ..., width: ...}`. Only has
      an effect if `:resize` is `true`
      """
    ],
    resize_method: [
      default: :bicubic,
      doc:
        "the resizing method, either of `:nearest`, `:bilinear`, `:bicubic`, `:lanczos3`, `:lanczos5`"
    ],
    normalize: [
      default: true,
      doc: "whether or not to normalize the input with mean and standard deviation"
    ],
    image_mean: [
      default: [0.5, 0.5, 0.5],
      doc: "the sequence of mean values for each channel, to be used when normalizing images"
    ],
    image_std: [
      default: [0.5, 0.5, 0.5],
      doc:
        "the sequence of standard deviations for each channel, to be used when normalizing images"
    ],
    patch_size: [
      default: 16,
      doc: "the spatial patch size"
    ],
    temporal_patch_size: [
      default: 2,
      doc: "the temporal patch size for video frames"
    ],
    merge_size: [
      default: 2,
      doc: "the merge factor for spatial patches"
    ],
    min_pixels: [
      default: nil,
      doc:
        "minimum number of pixels for `smart_resize`. If set, the input image is" <>
          " resized so that the (height, width) area is at least this value. Loaded" <>
          " from `size.shortest_edge` in `preprocessor_config.json`."
    ],
    max_pixels: [
      default: nil,
      doc:
        "maximum number of pixels for `smart_resize`. If set, the input image is" <>
          " resized so that the (height, width) area is at most this value. Loaded" <>
          " from `size.longest_edge`."
    ]
  ]

  @moduledoc """
  Qwen3-VL featurizer for image and video data.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct Shared.option_defaults(options)

  @behaviour Bumblebee.Featurizer
  @behaviour Bumblebee.Configurable

  alias Bumblebee.Utils.Image

  @impl true
  def config(featurizer, opts) do
    Shared.put_config_attrs(featurizer, opts)
  end

  @impl true
  def process_input(featurizer, input) do
    images = normalize_input(input)

    for image_or_video <- images do
      process_single_input(featurizer, image_or_video)
    end
    |> Nx.concatenate()
  end

  defp normalize_input(input) when is_list(input), do: input
  defp normalize_input(%{image: _} = input), do: [input]
  defp normalize_input(%{video: _} = input), do: [input]
  defp normalize_input(input), do: [%{image: input}]

  defp process_single_input(featurizer, %{video: frames}) when is_list(frames) do
    # Video input: process multiple frames
    frames
    |> Enum.map(&process_frame(featurizer, &1))
    |> Nx.stack()
    # Stack frames along temporal dimension: {batch, temporal, height, width, channels}
    |> Nx.transpose(axes: [1, 0, 2, 3, 4])
  end

  defp process_single_input(featurizer, %{image: image}) do
    # Single image: temporal dimension = 1
    image
    |> process_frame(featurizer)
    |> Nx.new_axis(1)

    # Shape: {batch, 1, height, width, channels}
  end

  defp process_single_input(featurizer, image) do
    # Assume it's just an image
    process_single_input(featurizer, %{image: image})
  end

  defp process_frame(frame, featurizer) do
    frame =
      frame
      |> Image.to_batched_tensor()
      |> Nx.as_type(:f32)
      |> Image.normalize_channels(length(featurizer.image_mean))

    {_, h, w, _} = Nx.shape(frame)

    # Image dimensions must be multiples of patch_size * merge_size so the
    # vision encoder can divide them into patches and then group those into
    # spatial-merge blocks. Beyond that, when min_pixels / max_pixels are
    # configured (Python's `smart_resize`), respect the [min_pixels,
    # max_pixels] area window while preserving aspect ratio.
    factor = featurizer.patch_size * featurizer.merge_size
    {target_h, target_w} = smart_resize(h, w, factor, featurizer.min_pixels, featurizer.max_pixels)

    NxImage.resize(frame, {target_h, target_w}, method: featurizer.resize_method)
  end

  # Picks a (height, width) such that:
  #   - both are multiples of `factor`,
  #   - both are at least `factor` (so we always have at least one patch),
  #   - their product (the pixel area) lies in [min_pixels, max_pixels],
  #     when those bounds are set,
  #   - the aspect ratio of the input is preserved as closely as possible.
  defp smart_resize(h, w, factor, min_pixels, max_pixels) do
    h_bar = max(round_to_multiple(h, factor), factor)
    w_bar = max(round_to_multiple(w, factor), factor)

    cond do
      is_integer(max_pixels) and h_bar * w_bar > max_pixels ->
        beta = :math.sqrt(h * w / max_pixels)
        h_b = max(floor_to_multiple(h / beta, factor), factor)
        w_b = max(floor_to_multiple(w / beta, factor), factor)
        {h_b, w_b}

      is_integer(min_pixels) and h_bar * w_bar < min_pixels ->
        beta = :math.sqrt(min_pixels / (h * w))
        h_b = max(ceil_to_multiple(h * beta, factor), factor)
        w_b = max(ceil_to_multiple(w * beta, factor), factor)
        {h_b, w_b}

      true ->
        {h_bar, w_bar}
    end
  end

  defp round_to_multiple(value, factor) do
    div(round(value) + div(factor, 2), factor) * factor
  end

  defp floor_to_multiple(value, factor) do
    div(trunc(value), factor) * factor
  end

  defp ceil_to_multiple(value, factor) do
    Kernel.ceil(value / factor) * factor
  end

  @impl true
  def batch_template(featurizer, batch_size) do
    # Get height/width from size config, defaulting to 224 if not specified
    {height, width} =
      case featurizer.size do
        %{height: h, width: w} -> {h, w}
        %{shortest_edge: edge} when edge < 10000 -> {edge, edge}
        _ -> {224, 224}
      end

    num_channels = length(featurizer.image_mean)
    # Output shape includes temporal dimension: {batch, channels, temporal, height, width}
    # For template, we use temporal=1 (single image case)
    %{
      "pixel_values" => Nx.template({batch_size, num_channels, 1, height, width}, :f32)
    }
  end

  @impl true
  def process_batch(featurizer, images) do
    # images shape: {batch, temporal, height, width, channels}
    images = NxImage.to_continuous(images, 0, 1)

    images =
      if featurizer.normalize do
        NxImage.normalize(
          images,
          Nx.tensor(featurizer.image_mean),
          Nx.tensor(featurizer.image_std)
        )
      else
        images
      end

    # Extract patches like Python processor
    # Python format: {num_patches, channels * temporal * patch_h * patch_w}
    {batch, temporal, height, width, channels} = Nx.shape(images)

    patch_size = featurizer.patch_size
    temporal_patch_size = featurizer.temporal_patch_size

    # For single images (temporal=1), Python duplicates the frame to match temporal_patch_size
    {images, temporal} =
      if temporal < temporal_patch_size do
        # Repeat the frame to match temporal_patch_size
        repeated = Nx.tile(images, [1, temporal_patch_size, 1, 1, 1])
        {repeated, temporal_patch_size}
      else
        {images, temporal}
      end

    patches_h = div(height, patch_size)
    patches_w = div(width, patch_size)
    patches_t = div(temporal, temporal_patch_size)

    merge = featurizer.merge_size
    h_block = div(patches_h, merge)
    w_block = div(patches_w, merge)

    # Reshape and reorder to match HuggingFace's Qwen2VLImageProcessor
    # patch ordering. Patches are grouped by `merge_size`x`merge_size`
    # spatial blocks so the downstream patch merger becomes a trivial
    # reshape (and so the loaded vision-encoder weights, which were
    # trained against this ordering, behave correctly).
    #
    # Input: {batch, temporal, height, width, channels}
    # Factor (height, width) into (h_block, m_h, p) and (w_block, m_w, p):
    images =
      images
      |> Nx.reshape(
        {batch, temporal, h_block, merge, patch_size, w_block, merge, patch_size, channels}
      )
      # Permute to Python's order:
      # (batch, h_block, w_block, m_h, m_w, channels, temporal, p_h, p_w)
      |> Nx.transpose(axes: [0, 2, 5, 3, 6, 8, 1, 4, 7])
      # Flatten patches: {batch, num_patches, channels * temporal * patch_h * patch_w}
      |> Nx.reshape(
        {batch, patches_t * patches_h * patches_w,
         channels * temporal_patch_size * patch_size * patch_size}
      )

    # For a single batch item, flatten to {num_patches, flattened_patch_size}
    # This matches Python's format
    {_batch, num_patches, patch_values} = Nx.shape(images)
    pixel_values = Nx.reshape(images, {num_patches, patch_values})

    # Generate grid_thw (temporal, height_patches, width_patches) per image
    image_grid_thw = Nx.tensor([[patches_t, patches_h, patches_w]])

    %{
      "pixel_values" => pixel_values,
      "image_grid_thw" => image_grid_thw
    }
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(featurizer, data) do
      import Shared.Converters

      opts =
        convert!(data,
          resize: {"do_resize", boolean()},
          resize_method: {"resample", resize_method()},
          normalize: {"do_normalize", boolean()},
          image_mean: {"image_mean", list(number())},
          image_std: {"image_std", list(number())},
          patch_size: {"patch_size", number()},
          temporal_patch_size: {"temporal_patch_size", number()},
          merge_size: {"merge_size", number()}
        )

      # Qwen3-VL preprocessor_config.json carries `size` as a pixel-area
      # window: `{"shortest_edge": min_pixels, "longest_edge": max_pixels}`.
      # Older configs use `{"height": h, "width": w}` for a fixed target. We
      # branch on the keys present and populate the corresponding featurizer
      # fields. (The `image_size/0` converter doesn't recognise the
      # area-window form, so we read it manually.)
      opts =
        case Map.get(data, "size") do
          %{"shortest_edge" => min_p, "longest_edge" => max_p}
          when is_integer(min_p) and is_integer(max_p) ->
            Keyword.merge(opts, min_pixels: min_p, max_pixels: max_p)

          %{"height" => h, "width" => w} when is_integer(h) and is_integer(w) ->
            Keyword.put(opts, :size, %{height: h, width: w})

          _ ->
            opts
        end

      @for.config(featurizer, opts)
    end
  end
end
