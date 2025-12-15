from io import BytesIO
from PIL import Image, ImageOps


def preprocess_image_to_bytes(file_bytes: bytes, max_size: int) -> bytes:
    """
    Normalize and resize the image to reduce latency and cost.
    Returns JPEG-encoded bytes.
    """
    with Image.open(BytesIO(file_bytes)) as img:
        img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90, optimize=True)
        return buf.getvalue()

