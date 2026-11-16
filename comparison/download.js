const studyButton = document.getElementById('download-study');
const studyStatus = document.getElementById('download-status');
async function studySha256(buffer) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)), b => b.toString(16).padStart(2, '0')).join('');
}
studyButton.addEventListener('click', async () => {
  studyButton.disabled = true;
  studyStatus.textContent = 'Preparing download…';
  try {
    const response = await fetch('download-parts.json');
    if (!response.ok) throw new Error('The download manifest could not be loaded.');
    const manifest = await response.json();
    const buffers = [];
    let downloaded = 0;
    for (const part of manifest.parts) {
      const result = await fetch(part.path);
      if (!result.ok) throw new Error('A download request failed. Please try again.');
      const buffer = await result.arrayBuffer();
      if (buffer.byteLength !== part.bytes || await studySha256(buffer) !== part.sha256) throw new Error('A file integrity check failed. Please try again.');
      buffers.push(buffer);
      downloaded += buffer.byteLength;
      studyStatus.textContent = `Downloaded ${(downloaded / 1e6).toFixed(1)} of ${(manifest.bytes / 1e6).toFixed(1)} MB…`;
    }
    const blob = new Blob(buffers, {type: 'application/zip'});
    if (blob.size !== manifest.bytes || await studySha256(await blob.arrayBuffer()) !== manifest.sha256) throw new Error('The package integrity check failed. Please try again.');
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = manifest.filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    studyStatus.textContent = 'Package verified. Your ZIP download has started.';
    studyButton.textContent = 'Download again';
  } catch (error) {
    studyStatus.textContent = error.message || 'The download could not finish. Please try again.';
  } finally {
    studyButton.disabled = false;
  }
});
