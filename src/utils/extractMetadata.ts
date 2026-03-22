import exifr from 'exifr';

export async function extractMetadata(file: File): Promise<{ flags: string[] }> {
  const flags: string[] = [];
  
  try {
    const exif = await exifr.parse(file, { tiff: true, exif: true, gps: true });
    
    if (!exif) {
      flags.push("No camera metadata found — possible AI generation or screenshot");
      return { flags };
    }

    if (exif.Software && (exif.Software.includes('Photoshop') || exif.Software.includes('Lightroom'))) {
      flags.push("Edited with image software detected");
    }

    if (!exif.Make || !exif.Model) {
      flags.push("No camera make/model found");
    }

    if (!exif.latitude || !exif.longitude) {
      flags.push("GPS data absent — common in manipulated media");
    }

    if (!exif.DateTimeOriginal) {
      flags.push("No original capture timestamp found");
    } else {
      // Check if created in last 60 seconds
      const captureTime = new Date(exif.DateTimeOriginal).getTime();
      const now = Date.now();
      if (now - captureTime < 60000) {
        flags.push("File created very recently");
      }
    }

    return { flags };
  } catch (error) {
    console.error("EXIF parsing error:", error);
    return { flags: ["Could not read metadata"] };
  }
}
