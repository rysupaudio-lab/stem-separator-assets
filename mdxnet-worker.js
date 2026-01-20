/**
 * MDX-Net Web Worker
 * Runs heavy audio processing off the main thread
 */

// Import ONNX Runtime
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.js');

// Configure WASM paths (local)
// ort.env.wasm.wasmPaths = './'; // Default behavior looks for .wasm in same directory

const MDX_CONSTANTS = {
    SAMPLE_RATE: 44100,
    N_FFT: 6144,
    HOP_LENGTH: 1024,
    DIM_F: 3072,
    DIM_C: 4,
    CHUNK_SIZE: 256,
    OVERLAP: 0.5
};

let session = null;

self.onmessage = async function (e) {
    const { type, data } = e.data;

    if (type === 'loadModel') {
        try {
            await loadModel(data.modelUrl);
            self.postMessage({ type: 'modelLoaded' });
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message });
        }
    } else if (type === 'separate') {
        try {
            const result = await separate(data.left, data.right);
            self.postMessage({ type: 'complete', result });
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message });
        }
    }
};

async function loadModel(modelUrl) {
    self.postMessage({ type: 'status', message: 'Downloading model...' });

    const response = await fetch(modelUrl);
    const contentLength = parseInt(response.headers.get('Content-Length') || '0');
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        self.postMessage({ type: 'downloadProgress', loaded, total: contentLength });
    }

    const modelBuffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        modelBuffer.set(chunk, offset);
        offset += chunk.length;
    }

    self.postMessage({ type: 'status', message: 'Loading model into memory...' });

    session = await ort.InferenceSession.create(modelBuffer.buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });

    console.log('MDX-Net model loaded in worker');
}

async function separate(leftChannel, rightChannel) {
    if (!session) throw new Error('Model not loaded');

    const { N_FFT, HOP_LENGTH, DIM_F, DIM_C, CHUNK_SIZE, OVERLAP } = MDX_CONSTANTS;
    const numSamples = leftChannel.length;

    // Pre-compute window
    const window = new Float32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) {
        window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / N_FFT));
    }

    self.postMessage({ type: 'status', message: 'Computing STFT...' });

    // Optimized STFT using typed arrays and pre-computed window
    const stftLeft = optimizedSTFT(leftChannel, window);
    const stftRight = optimizedSTFT(rightChannel, window);

    const numFrames = stftLeft.numFrames;
    const chunkStep = Math.floor(CHUNK_SIZE * (1 - OVERLAP));
    const numChunks = Math.max(1, Math.ceil((numFrames - CHUNK_SIZE) / chunkStep) + 1);

    // Pre-allocate output buffers
    const vocalReal = new Float32Array(numFrames * DIM_F * 2);
    const vocalImag = new Float32Array(numFrames * DIM_F * 2);
    const weights = new Float32Array(numFrames);

    self.postMessage({ type: 'status', message: `Processing ${numChunks} chunks...` });

    // Pre-allocate input buffer (reuse for each chunk)
    const inputData = new Float32Array(1 * DIM_C * DIM_F * CHUNK_SIZE);

    for (let c = 0; c < numChunks; c++) {
        const startFrame = Math.min(c * chunkStep, Math.max(0, numFrames - CHUNK_SIZE));
        const endFrame = Math.min(startFrame + CHUNK_SIZE, numFrames);
        const actualChunkSize = endFrame - startFrame;

        // Fill input buffer
        inputData.fill(0);
        const numBins = N_FFT / 2 + 1;

        for (let t = 0; t < actualChunkSize; t++) {
            const frameIdx = startFrame + t;
            for (let f = 0; f < DIM_F; f++) {
                const stftIdx = frameIdx * numBins + f;
                const ch0Idx = f * CHUNK_SIZE + t;
                const ch1Idx = DIM_F * CHUNK_SIZE + ch0Idx;
                const ch2Idx = 2 * DIM_F * CHUNK_SIZE + ch0Idx;
                const ch3Idx = 3 * DIM_F * CHUNK_SIZE + ch0Idx;

                inputData[ch0Idx] = stftLeft.real[stftIdx] || 0;
                inputData[ch1Idx] = stftLeft.imag[stftIdx] || 0;
                inputData[ch2Idx] = stftRight.real[stftIdx] || 0;
                inputData[ch3Idx] = stftRight.imag[stftIdx] || 0;
            }
        }

        // Run inference
        const inputTensor = new ort.Tensor('float32', inputData, [1, DIM_C, DIM_F, CHUNK_SIZE]);
        const feeds = { [session.inputNames[0]]: inputTensor };
        const results = await session.run(feeds);
        const outputData = results[session.outputNames[0]].data;

        // Accumulate output
        for (let t = 0; t < actualChunkSize; t++) {
            const frameIdx = startFrame + t;
            const w = 0.5 * (1 - Math.cos(2 * Math.PI * t / actualChunkSize));

            for (let f = 0; f < DIM_F; f++) {
                const ch0Idx = f * CHUNK_SIZE + t;
                const outIdx = frameIdx * DIM_F + f;

                vocalReal[outIdx * 2] += outputData[ch0Idx] * w;
                vocalImag[outIdx * 2] += outputData[DIM_F * CHUNK_SIZE + ch0Idx] * w;
                vocalReal[outIdx * 2 + 1] += outputData[2 * DIM_F * CHUNK_SIZE + ch0Idx] * w;
                vocalImag[outIdx * 2 + 1] += outputData[3 * DIM_F * CHUNK_SIZE + ch0Idx] * w;
            }
            weights[frameIdx] += w;
        }

        self.postMessage({
            type: 'progress',
            progress: (c + 1) / numChunks,
            current: c + 1,
            total: numChunks
        });
    }

    // Normalize
    const numBins = N_FFT / 2 + 1;
    for (let t = 0; t < numFrames; t++) {
        if (weights[t] > 0) {
            for (let f = 0; f < DIM_F; f++) {
                const idx = t * DIM_F + f;
                vocalReal[idx * 2] /= weights[t];
                vocalImag[idx * 2] /= weights[t];
                vocalReal[idx * 2 + 1] /= weights[t];
                vocalImag[idx * 2 + 1] /= weights[t];
            }
        }
    }

    // Build vocal STFT
    self.postMessage({ type: 'status', message: 'Computing inverse STFT...' });

    const vocalStftLeft = { real: new Float32Array(numFrames * numBins), imag: new Float32Array(numFrames * numBins), numFrames, numBins };
    const vocalStftRight = { real: new Float32Array(numFrames * numBins), imag: new Float32Array(numFrames * numBins), numFrames, numBins };
    const instStftLeft = { real: new Float32Array(numFrames * numBins), imag: new Float32Array(numFrames * numBins), numFrames, numBins };
    const instStftRight = { real: new Float32Array(numFrames * numBins), imag: new Float32Array(numFrames * numBins), numFrames, numBins };

    for (let t = 0; t < numFrames; t++) {
        for (let f = 0; f < numBins; f++) {
            const stftIdx = t * numBins + f;
            if (f < DIM_F) {
                const maskIdx = t * DIM_F + f;
                vocalStftLeft.real[stftIdx] = vocalReal[maskIdx * 2];
                vocalStftLeft.imag[stftIdx] = vocalImag[maskIdx * 2];
                vocalStftRight.real[stftIdx] = vocalReal[maskIdx * 2 + 1];
                vocalStftRight.imag[stftIdx] = vocalImag[maskIdx * 2 + 1];
            }
            instStftLeft.real[stftIdx] = stftLeft.real[stftIdx] - vocalStftLeft.real[stftIdx];
            instStftLeft.imag[stftIdx] = stftLeft.imag[stftIdx] - vocalStftLeft.imag[stftIdx];
            instStftRight.real[stftIdx] = stftRight.real[stftIdx] - vocalStftRight.real[stftIdx];
            instStftRight.imag[stftIdx] = stftRight.imag[stftIdx] - vocalStftRight.imag[stftIdx];
        }
    }

    // Inverse STFT
    const vocalsLeft = optimizedISTFT(vocalStftLeft, numSamples, window);
    const vocalsRight = optimizedISTFT(vocalStftRight, numSamples, window);
    const instLeft = optimizedISTFT(instStftLeft, numSamples, window);
    const instRight = optimizedISTFT(instStftRight, numSamples, window);

    return {
        vocals: { left: vocalsLeft, right: vocalsRight },
        instrumental: { left: instLeft, right: instRight }
    };
}

function optimizedSTFT(signal, window) {
    const N_FFT = MDX_CONSTANTS.N_FFT;
    const HOP = MDX_CONSTANTS.HOP_LENGTH;
    const numFrames = Math.ceil(signal.length / HOP);
    const numBins = N_FFT / 2 + 1;

    const real = new Float32Array(numFrames * numBins);
    const imag = new Float32Array(numFrames * numBins);

    // Pre-compute bit-reversal indices
    const bitRev = new Uint32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) {
        let j = 0, bit = N_FFT >> 1;
        for (let k = i; k > 0; k >>= 1, bit >>= 1) {
            if (k & 1) j |= bit;
        }
        bitRev[i] = j;
    }

    // Pre-compute twiddle factors
    const twiddleReal = new Float32Array(N_FFT / 2);
    const twiddleImag = new Float32Array(N_FFT / 2);
    for (let i = 0; i < N_FFT / 2; i++) {
        const theta = -2 * Math.PI * i / N_FFT;
        twiddleReal[i] = Math.cos(theta);
        twiddleImag[i] = Math.sin(theta);
    }

    const frameReal = new Float32Array(N_FFT);
    const frameImag = new Float32Array(N_FFT);
    const tempReal = new Float32Array(N_FFT);

    for (let frame = 0; frame < numFrames; frame++) {
        const start = frame * HOP;

        // Extract and window frame
        for (let i = 0; i < N_FFT; i++) {
            const idx = start + i - N_FFT / 2;
            tempReal[i] = (idx >= 0 && idx < signal.length) ? signal[idx] * window[i] : 0;
        }

        // Bit-reversal permutation
        for (let i = 0; i < N_FFT; i++) {
            frameReal[bitRev[i]] = tempReal[i];
            frameImag[i] = 0;
        }

        // Cooley-Tukey FFT
        for (let size = 2; size <= N_FFT; size *= 2) {
            const halfSize = size / 2;
            const step = N_FFT / size;

            for (let i = 0; i < N_FFT; i += size) {
                for (let j = 0; j < halfSize; j++) {
                    const twIdx = j * step;
                    const cos = twiddleReal[twIdx];
                    const sin = twiddleImag[twIdx];

                    const re = frameReal[i + j + halfSize] * cos - frameImag[i + j + halfSize] * sin;
                    const im = frameReal[i + j + halfSize] * sin + frameImag[i + j + halfSize] * cos;

                    frameReal[i + j + halfSize] = frameReal[i + j] - re;
                    frameImag[i + j + halfSize] = frameImag[i + j] - im;
                    frameReal[i + j] += re;
                    frameImag[i + j] += im;
                }
            }
        }

        // Store results
        const baseIdx = frame * numBins;
        for (let bin = 0; bin < numBins; bin++) {
            real[baseIdx + bin] = frameReal[bin];
            imag[baseIdx + bin] = frameImag[bin];
        }
    }

    return { real, imag, numFrames, numBins };
}

function optimizedISTFT(stft, length, window) {
    const N_FFT = MDX_CONSTANTS.N_FFT;
    const HOP = MDX_CONSTANTS.HOP_LENGTH;
    const { numFrames, numBins } = stft;

    const output = new Float32Array(length);
    const windowSum = new Float32Array(length);

    // Pre-compute bit-reversal and twiddle factors (same as STFT but conjugate)
    const bitRev = new Uint32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) {
        let j = 0, bit = N_FFT >> 1;
        for (let k = i; k > 0; k >>= 1, bit >>= 1) {
            if (k & 1) j |= bit;
        }
        bitRev[i] = j;
    }

    const twiddleReal = new Float32Array(N_FFT / 2);
    const twiddleImag = new Float32Array(N_FFT / 2);
    for (let i = 0; i < N_FFT / 2; i++) {
        const theta = -2 * Math.PI * i / N_FFT;
        twiddleReal[i] = Math.cos(theta);
        twiddleImag[i] = Math.sin(theta);
    }

    const fullReal = new Float32Array(N_FFT);
    const fullImag = new Float32Array(N_FFT);
    const frameReal = new Float32Array(N_FFT);
    const frameImag = new Float32Array(N_FFT);

    for (let frame = 0; frame < numFrames; frame++) {
        const baseIdx = frame * numBins;

        // Build full spectrum
        for (let bin = 0; bin < numBins; bin++) {
            fullReal[bin] = stft.real[baseIdx + bin];
            fullImag[bin] = -stft.imag[baseIdx + bin]; // Conjugate for IFFT
            if (bin > 0 && bin < numBins - 1) {
                fullReal[N_FFT - bin] = stft.real[baseIdx + bin];
                fullImag[N_FFT - bin] = stft.imag[baseIdx + bin];
            }
        }

        // Bit-reversal
        for (let i = 0; i < N_FFT; i++) {
            frameReal[bitRev[i]] = fullReal[i];
            frameImag[bitRev[i]] = fullImag[i];
        }

        // FFT (on conjugate = IFFT)
        for (let size = 2; size <= N_FFT; size *= 2) {
            const halfSize = size / 2;
            const step = N_FFT / size;

            for (let i = 0; i < N_FFT; i += size) {
                for (let j = 0; j < halfSize; j++) {
                    const twIdx = j * step;
                    const cos = twiddleReal[twIdx];
                    const sin = twiddleImag[twIdx];

                    const re = frameReal[i + j + halfSize] * cos - frameImag[i + j + halfSize] * sin;
                    const im = frameReal[i + j + halfSize] * sin + frameImag[i + j + halfSize] * cos;

                    frameReal[i + j + halfSize] = frameReal[i + j] - re;
                    frameImag[i + j + halfSize] = frameImag[i + j] - im;
                    frameReal[i + j] += re;
                    frameImag[i + j] += im;
                }
            }
        }

        // Overlap-add
        const start = frame * HOP;
        for (let i = 0; i < N_FFT; i++) {
            const idx = start + i - N_FFT / 2;
            if (idx >= 0 && idx < length) {
                output[idx] += (frameReal[i] / N_FFT) * window[i];
                windowSum[idx] += window[i] * window[i];
            }
        }
    }

    // Normalize
    for (let i = 0; i < length; i++) {
        if (windowSum[i] > 1e-8) {
            output[i] /= windowSum[i];
        }
    }

    return output;
}
