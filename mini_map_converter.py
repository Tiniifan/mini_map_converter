import math
import argparse
import sys
import os
from PIL import Image, ImageDraw

# --- Constants ---
# Expected input size for CS and GX modes (original minimap)
EXPECTED_SIZE_CS_GX = (256, 256)
# Expected input size for GO mode (transformed image)
EXPECTED_SIZE_GO = (320, 208)
# Default parameters for transformations (can be overridden if needed)
DEFAULT_NEW_MAP_SIZE = (208, 208)
DEFAULT_NEW_IMAGE_SIZE = (320, 208)
DEFAULT_OFFSET = (56, 0)
# The size the GO mode should output (original minimap size)
DEFAULT_GO_OUTPUT_SIZE = (256, 256)


# --- Transformation Functions ---

def adapt_bounding_box_and_image(
    old_bbox,
    old_minimap_image,
    new_map_size=DEFAULT_NEW_MAP_SIZE,
    new_image_size=DEFAULT_NEW_IMAGE_SIZE,
    offset=DEFAULT_OFFSET,
    resampling_filter=Image.Resampling.LANCZOS,
):
    """
    Transforms the bounding box coordinates and the image from an original minimap
    to a larger canvas where the minimap is resized and offset.

    Args:
        old_bbox (tuple): (minx, miny, maxx, maxy) in world coordinates for the input image.
        old_minimap_image (PIL.Image.Image): The original minimap image object.
        new_map_size (tuple): Target size (width, height) in pixels for the resized minimap.
        new_image_size (tuple): Size (width, height) in pixels of the final output canvas.
        offset (tuple): (x, y) position of the top-left corner of the resized minimap
                        on the new canvas.
        resampling_filter (Image.Resampling): PIL filter for resizing.

    Returns:
        tuple: (new_bbox, new_image)
            new_bbox (tuple): (minx, miny, maxx, maxy) world coordinates adapted to the new canvas.
            new_image (PIL.Image.Image): The resulting image with the resized minimap placed on a transparent background.
    """
    # --- 1. Calculate the new Bounding Box ---
    minx, miny, maxx, maxy = old_bbox
    old_world_width = maxx - minx
    old_world_height = maxy - miny

    if new_map_size[0] <= 0 or new_map_size[1] <= 0:
        raise ValueError("new_map_size cannot have zero or negative dimensions")

    # Handle zero-sized world dimensions for scale calculation
    scale_x = 0 if new_map_size[0] == 0 else (old_world_width / new_map_size[0] if old_world_width != 0 else 0)
    scale_y = 0 if new_map_size[1] == 0 else (old_world_height / new_map_size[1] if old_world_height != 0 else 0)

    pad_left = offset[0]
    pad_top = offset[1]
    pad_right = new_image_size[0] - (pad_left + new_map_size[0])
    pad_bottom = new_image_size[1] - (pad_top + new_map_size[1])

    if pad_right < 0 or pad_bottom < 0:
        raise ValueError("The sum of offset and new_map_size exceeds new_image_size")

    world_pad_left = pad_left * scale_x
    world_pad_right = pad_right * scale_x
    world_pad_top = pad_top * scale_y
    world_pad_bottom = pad_bottom * scale_y

    new_minx = minx - world_pad_left
    new_maxx = maxx + world_pad_right
    new_miny = miny - world_pad_top
    new_maxy = maxy + world_pad_bottom
    new_bbox = (new_minx, new_miny, new_maxx, new_maxy)

    # --- 2. Image Transformation ---
    # a) Resize the old minimap
    resized_minimap = old_minimap_image.resize(new_map_size, resample=resampling_filter)

    # b) Create a new transparent canvas of the target size
    new_image = Image.new('RGBA', new_image_size, (0, 0, 0, 0)) # Transparent background

    # c) Paste the resized minimap onto the transparent canvas at the offset
    if resized_minimap.mode == 'RGBA':
        new_image.paste(resized_minimap, offset, mask=resized_minimap)
    else:
        new_image.paste(resized_minimap, offset)

    return new_bbox, new_image

def inverse_adapt_bounding_box_and_image(
    new_bbox,
    new_image, # The image containing the potentially padded minimap
    output_original_size=DEFAULT_GO_OUTPUT_SIZE, # Target size for the output (original) minimap
    new_map_size=DEFAULT_NEW_MAP_SIZE, # Size of the minimap *within* new_image
    new_image_size=DEFAULT_NEW_IMAGE_SIZE, # Expected total size of new_image
    offset=DEFAULT_OFFSET, # Offset where the minimap was placed in new_image
    resampling_filter=Image.Resampling.LANCZOS
    ):
    """
    Reverses the transformation: calculates the original bounding box and
    extracts/resizes the minimap from the larger canvas image.

    Args:
        new_bbox (tuple): (minx, miny, maxx, maxy) world coordinates adapted to the input 'new_image'.
        new_image (PIL.Image.Image): The image object containing the transformed minimap.
        output_original_size (tuple): Target size (width, height) for the extracted/resized output image.
        new_map_size (tuple): Size (width, height) of the minimap area within 'new_image'.
        new_image_size (tuple): Expected size (width, height) of the input 'new_image' for bbox calculations.
        offset (tuple): (x, y) position where the minimap is expected to be within 'new_image'.
        resampling_filter (Image.Resampling): PIL filter for resizing.

    Returns:
        tuple: (old_bbox, old_minimap_image)
            old_bbox (tuple): (minx, miny, maxx, maxy) original world coordinates.
            old_minimap_image (PIL.Image.Image): The extracted and resized minimap image.
    """
    # Optional: Strict check on input image size if required, but bbox calc uses parameter
    if new_image.size != new_image_size:
        print(f"Warning: Input image size {new_image.size} differs from expected {new_image_size}. "
              f"Calculations will use {new_image_size}, cropping will use actual size.", file=sys.stderr)
        # Or: raise ValueError(f"Input image size {new_image.size} does not match expected {new_image_size}")

    # --- 1. Calculate the original Bounding Box ---
    new_minx, new_miny, new_maxx, new_maxy = new_bbox
    new_world_width = new_maxx - new_minx
    new_world_height = new_maxy - new_miny

    # Use parameters for calculation consistency
    calc_image_width = new_image_size[0]
    calc_image_height = new_image_size[1]

    if calc_image_width <= 0 or calc_image_height <= 0:
        raise ValueError("new_image_size parameter cannot have zero or negative dimensions")
    if new_map_size[0] <= 0 or new_map_size[1] <= 0:
        raise ValueError("new_map_size cannot have zero or negative dimensions")

    # Calculate original world dimensions based on the ratio of map size to total image size
    old_world_width = 0 if calc_image_width == 0 else new_world_width * (new_map_size[0] / calc_image_width)
    old_world_height = 0 if calc_image_height == 0 else new_world_height * (new_map_size[1] / calc_image_height)

    # Recalculate scale based on original world dimensions and the *resized* map size
    scale_x = 0 if new_map_size[0] == 0 else (old_world_width / new_map_size[0] if old_world_width != 0 else 0)
    scale_y = 0 if new_map_size[1] == 0 else (old_world_height / new_map_size[1] if old_world_height != 0 else 0)

    # Recalculate padding based on parameters
    pad_left = offset[0]
    pad_top = offset[1]
    pad_right = calc_image_width - (pad_left + new_map_size[0])
    pad_bottom = calc_image_height - (pad_top + new_map_size[1])

    if pad_right < 0 or pad_bottom < 0:
       raise ValueError("Calculated inconsistency: offset + new_map_size exceeds new_image_size.")

    # Convert pixel padding back to world padding using the scale
    world_pad_left = pad_left * scale_x
    world_pad_right = pad_right * scale_x
    world_pad_top = pad_top * scale_y
    world_pad_bottom = pad_bottom * scale_y

    # Reverse the final adaptation step
    old_minx = new_minx + world_pad_left
    old_miny = new_miny + world_pad_top
    old_maxx = new_maxx - world_pad_right
    old_maxy = new_maxy - world_pad_bottom
    old_bbox = (old_minx, old_miny, old_maxx, old_maxy)

    # --- 2. Image Extraction and Resizing ---
    # Define the crop box based on offset and map size
    crop_box = (
        offset[0],
        offset[1],
        offset[0] + new_map_size[0],
        offset[1] + new_map_size[1]
    )

    # Ensure cropping does not exceed actual image dimensions
    real_crop_box = (
        max(0, crop_box[0]),
        max(0, crop_box[1]),
        min(new_image.size[0], crop_box[2]),
        min(new_image.size[1], crop_box[3])
    )

    if real_crop_box[0] >= real_crop_box[2] or real_crop_box[1] >= real_crop_box[3]:
        # Crop area is invalid or outside the image
        print(f"Error: Calculated crop box {crop_box} is outside the actual image bounds {new_image.size}.", file=sys.stderr)
        raise ValueError("Cannot extract minimap, crop box is outside image bounds.")


    extracted_minimap = new_image.crop(real_crop_box)

    # Resize the extracted part to the desired original output size
    old_minimap_image = extracted_minimap.resize(output_original_size, resample=resampling_filter)

    return old_bbox, old_minimap_image

# --- Helper Function ---
def parse_float(value_str):
    """Converts a string to float, accepting '.' or ',' as decimal separator."""
    try:
        # Replace comma with dot for standard float conversion
        return float(value_str.replace(',', '.'))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{value_str}'")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimap Converter: Adapts minimap images and bounding boxes between formats.",
        formatter_class=argparse.RawTextHelpFormatter, # Keep newline formatting in description
        # Add epilog for examples
        epilog="Examples:\n"
               "  python mini_map_converter.py -cs original_map.png -400 -350 400 450\n"
               "  python mini_map_converter.py -go transformed_map.png -616 -350 616 450"
    )

    parser.add_argument(
        "mode",
        choices=['go', 'cs', 'gx'],
        help="Conversion mode:\n"
             "  -go: Inverse transformation. Expects 320x208 input image.\n"
             "       Input bbox is the adapted world coords.\n"
             "       Output is 256x256 image and original world coords.\n"
             "  -cs: Forward transformation. Expects 256x256 input image.\n"
             "       Input bbox is the original world coords.\n"
             "       Output is 320x208 image and adapted world coords.\n"
             "  -gx: Same as -cs."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file."
    )
    parser.add_argument(
        "bbox_coords",
        nargs=4,
        type=str, # Read as string first to handle comma decimals
        metavar='COORD', # Use a simple string instead of tuple for metavar
        help="Bounding box coordinates (min_x min_y max_x max_y) in world units. Accepts '.' or ',' as decimal separator."
    )

    # Check if any arguments were provided. If not, print help and exit.
    # Note: argparse often handles this, but we can be more explicit or customize.
    # However, fixing the metavar should allow argparse's default handling to work.
    # Let's rely on argparse first.

    try:
        args = parser.parse_args()
    except SystemExit as e:
         # Argparse calls sys.exit(2) on error, catch it to potentially add more info or just exit.
         # For this case, letting argparse print its own error is fine.
         sys.exit(e.code) # Exit with the same code argparse used

    # --- Input Validation ---
    try:
        input_bbox_list = [parse_float(coord) for coord in args.bbox_coords]
        input_bbox = tuple(input_bbox_list)
    except argparse.ArgumentTypeError as e:
        print(f"Error parsing bounding box: {e}", file=sys.stderr)
        # Print usage information as well
        parser.print_usage(sys.stderr)
        sys.exit(1)

    try:
        img = Image.open(args.image_path)
    except FileNotFoundError:
        print(f"Error: Input image file not found: '{args.image_path}'", file=sys.stderr)
        parser.print_usage(sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error opening or reading image file '{args.image_path}': {e}", file=sys.stderr)
        parser.print_usage(sys.stderr)
        sys.exit(1)

    # --- Mode Specific Logic ---
    result_bbox = None
    result_image = None
    output_suffix = args.mode.lstrip('-') # e.g., "go", "cs", "gx"

    try:
        if args.mode == 'go':
            # Inverse transformation
            if img.size != EXPECTED_SIZE_GO:
                print(f"Error: Mode '-go' requires input image size {EXPECTED_SIZE_GO}, but got {img.size}.", file=sys.stderr)
                parser.print_usage(sys.stderr)
                sys.exit(1)
            print(f"Mode: -go (Inverse). Input image: {args.image_path}, Input bbox: {input_bbox}")
            result_bbox, result_image = inverse_adapt_bounding_box_and_image(
                new_bbox=input_bbox, # Input bbox is the 'new' one for inverse
                new_image=img,
                output_original_size=DEFAULT_GO_OUTPUT_SIZE,
                new_map_size=DEFAULT_NEW_MAP_SIZE,
                new_image_size=DEFAULT_NEW_IMAGE_SIZE, # Use parameter size for consistency
                offset=DEFAULT_OFFSET
            )

        elif args.mode in ['cs', 'gx']:
            # Forward transformation
            if img.size != EXPECTED_SIZE_CS_GX:
                print(f"Error: Mode '{args.mode}' requires input image size {EXPECTED_SIZE_CS_GX}, but got {img.size}.", file=sys.stderr)
                parser.print_usage(sys.stderr)
                sys.exit(1)
            print(f"Mode: {args.mode} (Forward). Input image: {args.image_path}, Input bbox: {input_bbox}")
            result_bbox, result_image = adapt_bounding_box_and_image(
                old_bbox=input_bbox, # Input bbox is the 'old' one for forward
                old_minimap_image=img,
                new_map_size=DEFAULT_NEW_MAP_SIZE,
                new_image_size=DEFAULT_NEW_IMAGE_SIZE,
                offset=DEFAULT_OFFSET
            )

    except ValueError as e:
        print(f"Error during transformation: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)


    # --- Output ---
    if result_image and result_bbox:
        base, ext = os.path.splitext(args.image_path)
        # Ensure extension is reasonable, default to .png if input had none or weird one
        if not ext.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:
            ext = '.png'
        output_path = f"{base}_{output_suffix}{ext}"

        try:
            result_image.save(output_path)
            print(f"Output image saved to: '{output_path}'")
            # Print the calculated bounding box coordinates
            print(f"Calculated Bounding Box: {result_bbox[0]} {result_bbox[1]} {result_bbox[2]} {result_bbox[3]}")
        except Exception as e:
            print(f"Error saving output image to '{output_path}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Should not happen if error handling above works, but as a safeguard
        print("Error: Processing failed to produce a result.", file=sys.stderr)
        sys.exit(1)

    sys.exit(0) # Success