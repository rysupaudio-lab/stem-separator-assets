// --- PThread Proxy Logic ---
// Emscripten spawns sub-workers by reloading the current script.
// We must detect this and load the actual engine glue script instead of running our wrapper logic.
if (typeof self !== 'undefined' && self.name && (self.name.indexOf('pthread') !== -1 || self.name.indexOf('wasm-worker') !== -1)) {
    console.log('Worker: Sub-worker (pthread) detected. Loading engine glue script...');
    importScripts('demucs_onnx_simd.js');
    // Emscripten will take over from here.
}

let demucsEngineInstance = null;
let modelLoaded = false;
let lastWasmLogs = [];

// Global error handler to catch exit(1) and other hard crashes
self.onerror = function (msg, url, line, col, error) {
    console.error('Worker Global Error:', msg, error);
    self.postMessage({
        type: 'error',
        error: 'Engine Crash (Global): ' + msg,
        debugLog: lastWasmLogs.join('\n')
    });
};

self.onmessage = async function (e) {
    const { type, data } = e.data;

    if (type === 'init') {
        try {
            console.log('Worker Diagnostic Check:');
            console.log('- crossOriginIsolated:', self.crossOriginIsolated);
            console.log('- SharedArrayBuffer:', typeof self.SharedArrayBuffer !== 'undefined');

            if (!self.crossOriginIsolated || typeof self.SharedArrayBuffer === 'undefined') {
                throw new Error('Worker Security Error: Secure Context (Isolated) is missing. Refresh the page once.');
            }

            self.postMessage({ type: 'status', message: 'Loading WASM engine...' });

            // Load the engine glue script using importScripts for better Emscripten compatibility
            console.log('Worker: Loading glue script from:', data.wasmJsUrl);
            importScripts(data.wasmJsUrl);

            // Determine the factory function (libdemucs or createLibDemucs are common defaults)
            const moduleInit = self.libdemucs || self.createLibDemucs;
            if (typeof moduleInit !== 'function') {
                console.error('Worker: Glue script loaded but factory not found. Available globals:', Object.keys(self).filter(k => k.toLowerCase().includes('demucs')));
                throw new Error('WASM factory not found in engine script.');
            }

            demucsEngineInstance = await moduleInit({
                locateFile: (path) => path.endsWith('.wasm') ? data.wasmUrl : path,
                print: (msg) => {
                    console.log('WASM PRINT:', msg);
                    lastWasmLogs.push('[OUT] ' + msg);
                    if (lastWasmLogs.length > 50) lastWasmLogs.shift();
                },
                printErr: (msg) => {
                    console.error('WASM ERR:', msg);
                    lastWasmLogs.push('[ERR] ' + msg);
                    if (lastWasmLogs.length > 50) lastWasmLogs.shift();
                    if (msg.includes('Cannot enlarge memory') || msg.includes('memory allocation')) {
                        self.postMessage({ type: 'error', error: 'Out of Memory: Your device out of RAM for this song.' });
                    }
                }
            });

            self.postMessage({ type: 'status', message: 'Readying AI model...' });
            const modelResponse = await fetch(data.modelUrl);
            if (!modelResponse.ok) throw new Error('Model download failed: ' + modelResponse.status);

            const modelBytes = await modelResponse.arrayBuffer();

            // Alignment logic for M1 SIMD (16-byte)
            const alignment = 16;
            let modelPtrRaw = 0;
            let modelPtr = 0;

            if (demucsEngineInstance._emscripten_builtin_memalign) {
                console.log('Worker: Using engine memalign...');
                modelPtr = demucsEngineInstance._emscripten_builtin_memalign(alignment, modelBytes.byteLength);
                modelPtrRaw = modelPtr;
            } else {
                console.log('Worker: Manual 16-byte alignment...');
                modelPtrRaw = demucsEngineInstance._malloc(modelBytes.byteLength + alignment);
                modelPtr = (modelPtrRaw + (alignment - 1)) & ~(alignment - 1);
                console.log('Worker: Raw Ptr:', modelPtrRaw, 'Aligned Ptr:', modelPtr);
            }

            if (!modelPtr) throw new Error('Failed to allocate ' + modelBytes.byteLength + ' bytes for the AI model.');

            demucsEngineInstance.HEAPU8.set(new Uint8Array(modelBytes), modelPtr);
            const res = demucsEngineInstance._modelInit(modelPtr, modelBytes.byteLength);

            // Free the original pointer if we used manual alignment
            demucsEngineInstance._free(modelPtrRaw);

            if (res !== 0) throw new Error('Engine initialization failed code ' + res);

            modelLoaded = true;
            self.postMessage({ type: 'initialized' });
        } catch (err) {
            self.postMessage({
                type: 'error',
                error: 'Init Failure: ' + err.message,
                debugLog: lastWasmLogs.join('\n')
            });
        }
    }

    if (type === 'separate') {
        if (!modelLoaded) return;
        try {
            const { left, right, sampleRate } = data;
            const numSamples = left.length;
            const outputSize = 4 * 2 * numSamples;
            console.log('Worker: Allocating buffers with alignment...');

            let outPtrRaw = 0, outPtr = 0;
            let leftPtrRaw = 0, leftPtr = 0;
            let rightPtrRaw = 0, rightPtr = 0;

            if (demucsEngineInstance._emscripten_builtin_memalign) {
                outPtr = demucsEngineInstance._emscripten_builtin_memalign(alignment, outputSize * 4);
                leftPtr = demucsEngineInstance._emscripten_builtin_memalign(alignment, numSamples * 4);
                rightPtr = demucsEngineInstance._emscripten_builtin_memalign(alignment, numSamples * 4);
                outPtrRaw = outPtr; leftPtrRaw = leftPtr; rightPtrRaw = rightPtr;
            } else {
                outPtrRaw = demucsEngineInstance._malloc((outputSize * 4) + alignment);
                leftPtrRaw = demucsEngineInstance._malloc((numSamples * 4) + alignment);
                rightPtrRaw = demucsEngineInstance._malloc((numSamples * 4) + alignment);

                outPtr = (outPtrRaw + (alignment - 1)) & ~(alignment - 1);
                leftPtr = (leftPtrRaw + (alignment - 1)) & ~(alignment - 1);
                rightPtr = (rightPtrRaw + (alignment - 1)) & ~(alignment - 1);
            }

            if (!outPtr || !leftPtr || !rightPtr) throw new Error('Out of memory (Buffer allocation failed)');

            let heap = new Float32Array(demucsEngineInstance.HEAPU8.buffer);
            heap.set(left, leftPtr >> 2);
            heap.set(right, rightPtr >> 2);

            demucsEngineInstance._modelDemixSegment(leftPtr, rightPtr, numSamples, outPtr, (progress) => {
                self.postMessage({ type: 'progress', progress });
            });

            let outHeap = new Float32Array(demucsEngineInstance.HEAPU8.buffer);
            const stems = {
                drums: {
                    left: new Float32Array(outHeap.subarray(outPtr >> 2, (outPtr >> 2) + numSamples)),
                    right: new Float32Array(outHeap.subarray((outPtr >> 2) + numSamples, (outPtr >> 2) + numSamples * 2))
                },
                bass: {
                    left: new Float32Array(outHeap.subarray((outPtr >> 2) + numSamples * 2, (outPtr >> 2) + numSamples * 3)),
                    right: new Float32Array(outHeap.subarray((outPtr >> 2) + numSamples * 3, (outPtr >> 2) + numSamples * 4))
                },
                other: {
                    left: new Float32Array(outHeap.subarray((outPtr >> 2) + numSamples * 4, (outPtr >> 2) + numSamples * 5)),
                    right: new Float32Array(outHeap.subarray((outPtr >> 2) + numSamples * 5, (outPtr >> 2) + numSamples * 6))
                },
                vocals: {
                    left: new Float32Array(outHeap.subarray((outPtr >> 2) + numSamples * 6, (outPtr >> 2) + numSamples * 7)),
                    right: new Float32Array(outHeap.subarray((outPtr >> 2) + numSamples * 7, (outPtr >> 2) + numSamples * 8))
                }
            };

            demucsEngineInstance._free(leftPtrRaw);
            demucsEngineInstance._free(rightPtrRaw);
            demucsEngineInstance._free(outPtrRaw);

            const transferableBuffers = [
                stems.drums.left.buffer, stems.drums.right.buffer,
                stems.bass.left.buffer, stems.bass.right.buffer,
                stems.other.left.buffer, stems.other.right.buffer,
                stems.vocals.left.buffer, stems.vocals.right.buffer
            ];

            self.postMessage({ type: 'complete', stems, sampleRate }, transferableBuffers);
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message, debugLog: lastWasmLogs.join('\n') });
        }
    }
};
