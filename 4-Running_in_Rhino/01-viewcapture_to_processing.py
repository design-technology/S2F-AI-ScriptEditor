import io

from PIL import Image
import scriptcontext as sc

from System.IO import MemoryStream
import System.Drawing.Imaging as sdi
import System.Drawing as sd
from System import Array, Byte

import Eto.Drawing as ed

def bitmap_to_pil(bitmap: sd.Bitmap) -> Image.Image:
    net_stream = MemoryStream()
    
    bitmap.Save(net_stream, sdi.ImageFormat.Png)
    
    byte_array = net_stream.ToArray()
    pil_image = Image.open(io.BytesIO(byte_array))

    return pil_image.convert("RGB")
    
def pil_to_bitmap(pil_image: Image.Image) -> ed.Bitmap:
    if pil_image is None:
        return None
    
    stream = io.BytesIO()
    pil_image.save(stream, format="PNG")
    byte_array = stream.getvalue()
    byte_array_net = Array[Byte](byte_array)  # Convert to .NET Byte[]

    return ed.Bitmap(byte_array_net)

if __name__ == "__main__":
    sd_bitmap = sc.doc.Views.ActiveView.CaptureToBitmap(sd.Size(512,512), False, False, False)
    pil_bitmap = bitmap_to_pil(sd_bitmap)
    eto_bitmap = pil_to_bitmap(pil_bitmap)

