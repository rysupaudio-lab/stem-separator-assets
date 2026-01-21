/**
 * HTDemucs Web Worker
 * Handles Hybrid Transformer Demucs inference via ONNX Runtime Web
 * Domain: Waveform (Time Domain)
 */

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.js');

const CONFIG = {
    // Model specific settings
    SAMPLE_RATE: 44100,
    SEGMENT_SIZE: 44100 * 6, // ~6 seconds per chunk to be safe on memory
    OVERLAP: 0.25, // 25% overlap
};

let session = null;

// Standard Demucs Output Mapping (typically)
// 0: Drums
// 1: Bass
// 2: Other
// 3: Vocals
// We will verify this by checking if the output shape allows us to infer, or just assuming standard.

self.onmessage = async function (e) {
    const { type, data } = e.data;
    try {
        if (type === 'loadModel') {
            await loadModel(data.modelUrl);
            self.postMessage({ type: 'modelLoaded' });
        } else if (type === 'separate') {
            const result = await separate(data.left, data.right);
            self.postMessage({ type: 'complete', result });
        }
    } catch (err) {
        console.error("Worker Error:", err);
        self.postMessage({ type: 'error', error: err.message });
    }
};

async function loadModel(modelUrl) {
    self.postMessage({ type: 'status', message: 'Downloading HTDemucs model (174MB)...' });

    // Check for cached model to avoid re-downloading if possible, though browser cache handles most
    const response = await fetch(modelUrl);
    const contentLength = parseInt(response.headers.get('Content-Length') || '0');

    // Simple progress stream
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        if (contentLength > 0) {
            self.postMessage({ type: 'downloadProgress', loaded, total: contentLength });
        }
    }

    const modelBlob = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        modelBlob.set(chunk, offset);
        offset += chunk.length;
    }

    self.postMessage({ type: 'status', message: 'Loading model into ONNX Runtime...' });

    try {
        session = await ort.InferenceSession.create(modelBlob.buffer, {
            executionProviders: ['wasm'], // WebGL often fails on huge audio tensors, WASM is safer for Demucs
            graphOptimizationLevel: 'all'
        });

        console.log("Model Loaded. Inputs:", session.inputNames, "Outputs:", session.outputNames);

    } catch (e) {
        throw new Error(`Failed to create ONNX session: ${e.message}`);
    }
}

async function separate(left, right) {
    if (!session) throw new Error("Model not loaded");

    const length = left.length;
    // Pad to align with segment size if needed? 
    // Demucs can handle variable length, but for OLA chunking we need consistency.

    const segSize = CONFIG.SEGMENT_SIZE;
    const overlap = Math.floor(segSize * CONFIG.OVERLAP);
    const step = segSize - overlap;

    // Output buffers for Vocals (Left/Right)
    // We assume Vocals is index 3.
    const vocalsL = new Float32Array(length);
    const vocalsR = new Float32Array(length);
    const countBuffer = new Float32Array(length); // To normalize OLA

    const totalChunks = Math.ceil((length - overlap) / step);

    for (let i = 0; i < totalChunks; i++) {
        const start = i * step;
        const end = Math.min(start + segSize, length);
        const actualLen = end - start;

        // Prepare Input Tensor [1, 2, actualLen]
        const chunkL = left.slice(start, end);
        const chunkR = right.slice(start, end);

        // Demucs expects stereo input. 
        // If the chunk is smaller than segSize (last chunk), usually we verify if model necessitates fixed size.
        // HTDemucs ONNX typically accepts dynamic third dimension.

        const inputData = new Float32Array(2 * actualLen);
        // Interleaved or Planar? ONNX usually expects NCHW (Planar) -> [1, 2, L]
        // So R follows L.
        inputData.set(chunkL, 0);
        inputData.set(chunkR, actualLen);

        const inputTensor = new ort.Tensor('float32', inputData, [1, 2, actualLen]);

        // Run Inference
        const feeds = { [session.inputNames[0]]: inputTensor };
        const results = await session.run(feeds);
        const outputTensor = results[session.outputNames[0]];
        // Output Shape: [1, 4, 2, actualLen]
        // 1 = Batch
        // 4 = Sources (Drums, Bass, Other, Vocals)
        // 2 = Channels (Stereo)
        // actualLen = Samples

        const outData = outputTensor.data;

        // Indices mapping (NCHW flattened):
        // Source 0 (Drums): start 0
        // Source 1 (Bass): start 1 * 2 * L
        // Source 2 (Other): start 2 * 2 * L
        // Source 3 (Vocals): start 3 * 2 * L

        const L = actualLen;
        const numSources = 4; // Expected
        const vocabIdx = 3;

        const vocabOffset = vocabIdx * 2 * L;
        // Vocals Left starts at vocabOffset
        // Vocals Right starts at vocabOffset + L

        const vL = outData.subarray(vocabOffset, vocabOffset + L);
        const vR = outData.subarray(vocabOffset + L, vocabOffset + 2 * L);

        // Overlap-Add with simple linear crossfade (or simple cosine window)
        // For simplicity and performance, we'll use a Hanning window on the chunk.
        // Or simpler: triangular.

        for (let t = 0; t < L; t++) {
            // Hanning window
            // w[n] = 0.5 * (1 - cos(2*PI*n / (N-1)))
            const w = 0.5 * (1 - Math.cos(2 * Math.PI * t / (L - 1)));

            // Or simpler: standard unity weight if we manage overlap perfectly, 
            // but for independent chunks, windowing is safer.

            const pos = start + t;
            if (pos < length) {
                vocalsL[pos] += vL[t] * w;
                vocalsR[pos] += vR[t] * w;
                countBuffer[pos] += w;
            }
        }

        // Report Progress
        if (i % 2 === 0 || i === totalChunks - 1) {
            self.postMessage({ type: 'progress', progress: (i + 1) / totalChunks });
        }
    }

    // Normalize by Window Sum
    for (let i = 0; i < length; i++) {
        if (countBuffer[i] > 1e-8) {
            vocalsL[i] /= countBuffer[i];
            vocalsR[i] /= countBuffer[i];
        }
    }

    // Create "Instrumental" by subtracting Vocals from Original
    // (Inverse Phase cancellation method is cheap and perfect for alignment)
    const instL = new Float32Array(length);
    const instR = new Float32Array(length);

    for (let i = 0; i < length; i++) {
        // Instrumental = Original - Vocals
        instL[i] = left[i] - vocalsL[i];
        instR[i] = right[i] - vocalsR[i];
    }

    return {
        vocals: { left: vocalsL, right: vocalsR },
        instrumental: { left: instL, right: instR }
    };
}
