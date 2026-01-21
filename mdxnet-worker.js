/**
 * MDX-Net Web Worker
 * Optimized for N_FFT=6144 (Mixed-Radix FFT: 3 * 2048)
 */

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.js');

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
let twiddle6144 = null;
let window6144 = null;
let bitRev2048 = null;
let twiddle2048 = null;

self.onmessage = async function (e) {
    const { type, data } = e.data;
    try {
        if (type === 'loadModel') {
            initTables();
            await loadModel(data.modelUrl);
            self.postMessage({ type: 'modelLoaded' });
        } else if (type === 'separate') {
            const result = await separate(data.left, data.right);
            self.postMessage({ type: 'complete', result });
        }
    } catch (err) {
        console.error(err);
        self.postMessage({ type: 'error', error: err.message });
    }
};

function initTables() {
    if (twiddle6144) return;

    // 6144 Tables
    const N = 6144;
    twiddle6144 = { real: new Float32Array(N), imag: new Float32Array(N) };
    for (let k = 0; k < N; k++) {
        const theta = -2 * Math.PI * k / N;
        twiddle6144.real[k] = Math.cos(theta);
        twiddle6144.imag[k] = Math.sin(theta);
    }

    window6144 = new Float32Array(N);
    for (let i = 0; i < N; i++) {
        window6144[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / N));
    }

    // 2048 Tables
    const M = 2048;
    bitRev2048 = new Uint32Array(M);
    for (let i = 0; i < M; i++) {
        let j = 0, bit = M >> 1;
        for (let k = i; k > 0; k >>= 1, bit >>= 1) if (k & 1) j |= bit;
        bitRev2048[i] = j;
    }

    twiddle2048 = { real: new Float32Array(M / 2), imag: new Float32Array(M / 2) };
    for (let i = 0; i < M / 2; i++) {
        const theta = -2 * Math.PI * i / M;
        twiddle2048.real[i] = Math.cos(theta);
        twiddle2048.imag[i] = Math.sin(theta);
    }
}

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
    session = await ort.InferenceSession.create(modelBuffer.buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });
}

async function separate(leftChannel, rightChannel) {
    if (!session) throw new Error('Model not loaded');
    const { N_FFT, CHUNK_SIZE, OVERLAP, DIM_F, DIM_C } = MDX_CONSTANTS;
    const numSamples = leftChannel.length;

    self.postMessage({ type: 'status', message: 'Computing STFT...' });
    const stftLeft = stft6144(leftChannel);
    const stftRight = stft6144(rightChannel);

    const numFrames = stftLeft.numFrames;
    const numBins = stftLeft.numBins;

    const chunkStep = Math.floor(CHUNK_SIZE * (1 - OVERLAP));
    const numChunks = Math.max(1, Math.ceil((numFrames - CHUNK_SIZE) / chunkStep) + 1);

    const vocalReal = new Float32Array(numFrames * DIM_F * 2);
    const vocalImag = new Float32Array(numFrames * DIM_F * 2);
    const weights = new Float32Array(numFrames);
    const inputData = new Float32Array(1 * DIM_C * DIM_F * CHUNK_SIZE);

    for (let c = 0; c < numChunks; c++) {
        const startFrame = Math.min(c * chunkStep, Math.max(0, numFrames - CHUNK_SIZE));
        const endFrame = Math.min(startFrame + CHUNK_SIZE, numFrames);
        const actualChunkSize = endFrame - startFrame;

        inputData.fill(0);
        for (let t = 0; t < actualChunkSize; t++) {
            const frameIdx = startFrame + t;
            for (let f = 0; f < DIM_F; f++) {
                const stftIdx = frameIdx * numBins + f;

                const valL_re = stftLeft.real[stftIdx];
                const valL_im = stftLeft.imag[stftIdx];
                const valR_re = stftRight.real[stftIdx];
                const valR_im = stftRight.imag[stftIdx];

                const ch0 = f * CHUNK_SIZE + t;
                inputData[ch0] = valL_re || 0;
                inputData[DIM_F * CHUNK_SIZE + ch0] = valL_im || 0;
                inputData[2 * DIM_F * CHUNK_SIZE + ch0] = valR_re || 0;
                inputData[3 * DIM_F * CHUNK_SIZE + ch0] = valR_im || 0;
            }
        }

        const inputTensor = new ort.Tensor('float32', inputData, [1, DIM_C, DIM_F, CHUNK_SIZE]);
        const feeds = { [session.inputNames[0]]: inputTensor };
        const results = await session.run(feeds);
        const outputData = results[session.outputNames[0]].data;

        // Accumulate
        for (let t = 0; t < actualChunkSize; t++) {
            const frameIdx = startFrame + t;
            const w = 0.5 * (1 - Math.cos(2 * Math.PI * t / actualChunkSize));
            for (let f = 0; f < DIM_F; f++) {
                const ch0 = f * CHUNK_SIZE + t;
                const outIdx = frameIdx * DIM_F + f;
                vocalReal[outIdx * 2] += outputData[ch0] * w;
                vocalImag[outIdx * 2] += outputData[DIM_F * CHUNK_SIZE + ch0] * w;
                vocalReal[outIdx * 2 + 1] += outputData[2 * DIM_F * CHUNK_SIZE + ch0] * w;
                vocalImag[outIdx * 2 + 1] += outputData[3 * DIM_F * CHUNK_SIZE + ch0] * w;
            }
            weights[frameIdx] += w;
        }

        if (c % 5 === 0 || c === numChunks - 1) {
            self.postMessage({ type: 'progress', progress: (c + 1) / numChunks });
        }
    }

    // Normalize
    for (let t = 0; t < numFrames; t++) {
        if (weights[t] > 1e-8) {
            for (let f = 0; f < DIM_F; f++) {
                const idx = t * DIM_F + f;
                vocalReal[idx * 2] /= weights[t];
                vocalImag[idx * 2] /= weights[t];
                vocalReal[idx * 2 + 1] /= weights[t];
                vocalImag[idx * 2 + 1] /= weights[t];
            }
        }
    }

    // Reconstruct
    const vocalStftLeft = { real: new Float32Array(numFrames * numBins), imag: new Float32Array(numFrames * numBins), numFrames, numBins };
    const vocalStftRight = { real: new Float32Array(numFrames * numBins), imag: new Float32Array(numFrames * numBins), numFrames, numBins };

    // Fill vocal STFT (copy 0..DIM_F, zero rest)
    for (let t = 0; t < numFrames; t++) {
        for (let f = 0; f < DIM_F; f++) {
            const idx = t * DIM_F + f;
            const stftIdx = t * numBins + f;
            vocalStftLeft.real[stftIdx] = vocalReal[idx * 2];
            vocalStftLeft.imag[stftIdx] = vocalImag[idx * 2];
            vocalStftRight.real[stftIdx] = vocalReal[idx * 2 + 1];
            vocalStftRight.imag[stftIdx] = vocalImag[idx * 2 + 1];
        }
    }

    // Compute Instrumental
    const instStftLeft = { real: new Float32Array(numFrames * numBins), imag: new Float32Array(numFrames * numBins), numFrames, numBins };
    const instStftRight = { real: new Float32Array(numFrames * numBins), imag: new Float32Array(numFrames * numBins), numFrames, numBins };

    for (let i = 0; i < stftLeft.real.length; i++) {
        instStftLeft.real[i] = stftLeft.real[i] - vocalStftLeft.real[i];
        instStftLeft.imag[i] = stftLeft.imag[i] - vocalStftLeft.imag[i];
        instStftRight.real[i] = stftRight.real[i] - vocalStftRight.real[i];
        instStftRight.imag[i] = stftRight.imag[i] - vocalStftRight.imag[i];
    }

    self.postMessage({ type: 'status', message: 'Computing inverse STFT...' });
    const vocalsLeft = istft6144(vocalStftLeft, numSamples);
    const vocalsRight = istft6144(vocalStftRight, numSamples);
    const instLeft = istft6144(instStftLeft, numSamples);
    const instRight = istft6144(instStftRight, numSamples);

    return {
        vocals: { left: vocalsLeft, right: vocalsRight },
        instrumental: { left: instLeft, right: instRight }
    };
}

function stft6144(signal) {
    const { N_FFT, HOP_LENGTH } = MDX_CONSTANTS;
    const numFrames = Math.ceil(signal.length / HOP_LENGTH);
    const numBins = N_FFT / 2 + 1;
    const real = new Float32Array(numFrames * numBins);
    const imag = new Float32Array(numFrames * numBins);
    const frameData = new Float32Array(N_FFT);

    for (let frame = 0; frame < numFrames; frame++) {
        const start = frame * HOP_LENGTH;
        for (let i = 0; i < N_FFT; i++) {
            const idx = start + i - N_FFT / 2;
            frameData[i] = (idx >= 0 && idx < signal.length) ? signal[idx] * window6144[i] : 0;
        }

        const { real: fReal, imag: fImag } = fft6144(frameData);

        for (let bin = 0; bin < numBins; bin++) {
            const idx = frame * numBins + bin;
            real[idx] = fReal[bin];
            imag[idx] = fImag[bin];
        }
    }
    return { real, imag, numFrames, numBins };
}

function istft6144(stft, length) {
    const { N_FFT, HOP_LENGTH } = MDX_CONSTANTS;
    const { numFrames, numBins } = stft;
    const output = new Float32Array(length);
    const overlapSum = new Float32Array(length);
    const fullReal = new Float32Array(N_FFT);
    const fullImag = new Float32Array(N_FFT);

    for (let frame = 0; frame < numFrames; frame++) {
        const baseIdx = frame * numBins;
        for (let bin = 0; bin < numBins; bin++) {
            fullReal[bin] = stft.real[baseIdx + bin];
            fullImag[bin] = stft.imag[baseIdx + bin];
            if (bin > 0 && bin < numBins - 1) {
                fullReal[N_FFT - bin] = fullReal[bin];
                fullImag[N_FFT - bin] = -fullImag[bin];
            }
        }

        const frameData = ifft6144(fullReal, fullImag);

        const start = frame * HOP_LENGTH;
        for (let i = 0; i < N_FFT; i++) {
            const idx = start + i - N_FFT / 2;
            if (idx >= 0 && idx < length) {
                output[idx] += frameData[i] * window6144[i];
                overlapSum[idx] += window6144[i] * window6144[i];
            }
        }
    }
    for (let i = 0; i < length; i++) {
        if (overlapSum[i] > 1e-8) output[i] /= overlapSum[i];
    }
    return output;
}

function fft6144(input) {
    const N = 6144;
    const M = 2048;
    const x0 = new Float32Array(M);
    const x1 = new Float32Array(M);
    const x2 = new Float32Array(M);

    for (let r = 0; r < M; r++) {
        x0[r] = input[3 * r];
        x1[r] = input[3 * r + 1];
        x2[r] = input[3 * r + 2];
    }

    const X0 = fft2048(x0);
    const X1 = fft2048(x1);
    const X2 = fft2048(x2);

    const finalReal = new Float32Array(N);
    const finalImag = new Float32Array(N);

    for (let k = 0; k < N; k++) {
        const k_mod = k % M;
        const re0 = X0.real[k_mod];
        const im0 = X0.imag[k_mod];

        const re1 = X1.real[k_mod];
        const im1 = X1.imag[k_mod];

        const re2 = X2.real[k_mod];
        const im2 = X2.imag[k_mod];

        const w1_re = twiddle6144.real[k];
        const w1_im = twiddle6144.imag[k];

        const idx2 = (2 * k) % N;
        const w2_re = twiddle6144.real[idx2];
        const w2_im = twiddle6144.imag[idx2];

        const t1_re = re1 * w1_re - im1 * w1_im;
        const t1_im = re1 * w1_im + im1 * w1_re;

        const t2_re = re2 * w2_re - im2 * w2_im;
        const t2_im = re2 * w2_im + im2 * w2_re;

        finalReal[k] = re0 + t1_re + t2_re;
        finalImag[k] = im0 + t1_im + t2_im;
    }
    return { real: finalReal, imag: finalImag };
}

function ifft6144(real, imag) {
    const N = 6144;
    const conjImag = new Float32Array(N);
    for (let i = 0; i < N; i++) conjImag[i] = -imag[i];
    const result = fft_complex_6144(real, conjImag);
    const out = new Float32Array(N);
    for (let i = 0; i < N; i++) out[i] = result.real[i] / N;
    return out;
}

function fft_complex_6144(realIn, imagIn) {
    const N = 6144;
    const M = 2048;
    const r0 = new Float32Array(M), i0 = new Float32Array(M);
    const r1 = new Float32Array(M), i1 = new Float32Array(M);
    const r2 = new Float32Array(M), i2 = new Float32Array(M);

    for (let r = 0; r < M; r++) {
        r0[r] = realIn[3 * r]; i0[r] = imagIn[3 * r];
        r1[r] = realIn[3 * r + 1]; i1[r] = imagIn[3 * r + 1];
        r2[r] = realIn[3 * r + 2]; i2[r] = imagIn[3 * r + 2];
    }

    const X0 = fft2048_complex(r0, i0);
    const X1 = fft2048_complex(r1, i1);
    const X2 = fft2048_complex(r2, i2);

    const finalReal = new Float32Array(N);
    const finalImag = new Float32Array(N);

    for (let k = 0; k < N; k++) {
        const k_mod = k % M;
        const re0 = X0.real[k_mod], im0 = X0.imag[k_mod];
        const re1 = X1.real[k_mod], im1 = X1.imag[k_mod];
        const re2 = X2.real[k_mod], im2 = X2.imag[k_mod];

        const w1_re = twiddle6144.real[k];
        const w1_im = twiddle6144.imag[k];

        const idx2 = (2 * k) % N;
        const w2_re = twiddle6144.real[idx2];
        const w2_im = twiddle6144.imag[idx2];

        const t1_re = re1 * w1_re - im1 * w1_im;
        const t1_im = re1 * w1_im + im1 * w1_re;

        const t2_re = re2 * w2_re - im2 * w2_im;
        const t2_im = re2 * w2_im + im2 * w2_re;

        finalReal[k] = re0 + t1_re + t2_re;
        finalImag[k] = im0 + t1_im + t2_im;
    }
    return { real: finalReal, imag: finalImag };
}

function fft2048(input) {
    const M = 2048;
    const real = new Float32Array(M);
    const imag = new Float32Array(M);
    for (let i = 0; i < M; i++) {
        real[bitRev2048[i]] = input[i];
    }
    return fft2048_core(real, imag);
}

function fft2048_complex(realIn, imagIn) {
    const M = 2048;
    const real = new Float32Array(M);
    const imag = new Float32Array(M);
    for (let i = 0; i < M; i++) {
        const rev = bitRev2048[i];
        real[rev] = realIn[i];
        imag[rev] = imagIn[i];
    }
    return fft2048_core(real, imag);
}

function fft2048_core(real, imag) {
    const M = 2048;
    for (let size = 2; size <= M; size *= 2) {
        const halfSize = size / 2;
        const step = M / size;
        for (let i = 0; i < M; i += size) {
            for (let j = 0; j < halfSize; j++) {
                const twIdx = j * step;
                const cos = twiddle2048.real[twIdx];
                const sin = twiddle2048.imag[twIdx];

                const re = real[i + j + halfSize] * cos - imag[i + j + halfSize] * sin;
                const im = real[i + j + halfSize] * sin + imag[i + j + halfSize] * cos;

                real[i + j + halfSize] = real[i + j] - re;
                imag[i + j + halfSize] = imag[i + j] - im;
                real[i + j] += re;
                imag[i + j] += im;
            }
        }
    }
    return { real, imag };
}
