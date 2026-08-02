export async function extractVideoFrame(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    video.playsInline = true;

    const url = URL.createObjectURL(file);
    video.src = url;

    video.onloadeddata = () => {
      video.currentTime = 0.1; // Seek to 0.1s to avoid black frames
    };

    video.onseeked = () => {
      const canvas = document.createElement('canvas');
      // Keep the frame at native resolution so forensic high-frequency detail
      // (GAN fingerprints, blending seams, compression blocking) survives to the
      // model. Draw preserves aspect ratio — no letterboxing, no edge crop.
      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 360;
      // Cap the longest side to 1280 to bound memory on 4K sources.
      const maxSide = 1280;
      const scale = Math.min(1, maxSide / Math.max(vw, vh));
      canvas.width = Math.round(vw * scale);
      canvas.height = Math.round(vh * scale);
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        URL.revokeObjectURL(url);
        reject(new Error("Could not get canvas context"));
        return;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      URL.revokeObjectURL(url);
      
      // Return base64 without data URI prefix
      const base64 = dataUrl.split(',')[1];
      resolve(base64);
    };

    video.onerror = (e) => {
      URL.revokeObjectURL(url);
      reject(new Error("Error loading video"));
    };
  });
}
