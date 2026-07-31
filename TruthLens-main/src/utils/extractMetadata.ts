import exifr from 'exifr';

export async function extractMetadata(file: File): Promise<{ flags: string[] }> {
  const flags: string[] = [];
  
  if (file.type.startsWith('video/')) {
    flags.push("Video file — standard image EXIF metadata not applicable");
    return { flags };
  }

  try {
    const exif = await exifr.parse(file, { tiff: true, exif: true, gps: true });
    
    if (!exif) {
      flags.push("No camera metadata found — possible AI generation or screenshot");
      return { flags };
    }

    if (exif.Software) {
      const software = exif.Software.toLowerCase();
      const aiTools = ['midjourney', 'dall-e', 'dalle', 'stable diffusion', 'firefly', 'generative fill', 'ai', 'topaz', 'runway', 'canva'];
      
      if (aiTools.some(tool => software.includes(tool))) {
        flags.push(`AI generation/editing software detected: ${exif.Software}`);
      } else if (software.includes('photoshop') || software.includes('lightroom')) {
        flags.push("Edited with image software detected (potential for AI generative fill)");
      } else {
        flags.push(`Software tag present: ${exif.Software}`);
      }
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
  } catch (error: any) {
    console.error("EXIF parsing error:", error.message || error);
    if (error.message === 'Unknown file format') {
      return { flags: ["No camera metadata found — possible AI generation or screenshot"] };
    }
    return { flags: ["Could not read metadata"] };
  }
}
