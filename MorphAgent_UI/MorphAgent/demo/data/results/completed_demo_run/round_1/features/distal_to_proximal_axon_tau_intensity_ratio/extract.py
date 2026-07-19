def extract(img, seg):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops, label

    # Normalize the image to the range [0.0, 1.0] for consistency
    img = np.asarray(img, dtype=np.float32)
    img = img / 9042.0  # Normalize uint16 data using its max value

    # If no segmentation masks are provided, return 0.0
    if seg is None or not seg:
        return 0.0

    # Access relevant segmentation masks
    proximal_mask = seg.get("mask_bundle")  # Proximal Tau regions (bundled filaments)
    distal_mask = seg.get("mask_filament")  # Distal Tau regions (thin filaments)

    # Ensure masks are provided
    if proximal_mask is None or distal_mask is None:
        return 0.0

    # Ensure the dimensions of the masks match the input image
    if proximal_mask.shape != img.shape or distal_mask.shape != img.shape:
        return 0.0

    # Compute the mean intensity of Tau signal in the proximal axonal regions
    proximal_intensity = np.mean(img[proximal_mask > 0])

    # Compute the mean intensity of Tau signal in the distal axonal regions
    distal_intensity = np.mean(img[distal_mask > 0])

    # To avoid division by zero, return 0.0 if the proximal intensity is effectively zero
    if proximal_intensity == 0:
        return 0.0

    # Compute the distal-to-proximal Tau intensity ratio
    ratio = distal_intensity / proximal_intensity

    return float(ratio)
