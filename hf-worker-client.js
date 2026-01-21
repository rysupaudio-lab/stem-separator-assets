/**
 * Hugging Face Gradio Client for Vocal Separation
 * Specifically for the Politrees/audio-separator_UVR space
 */

class HFWorkerClient {
    constructor(options = {}) {
        this.spaceUrl = options.spaceUrl || 'https://politrees-audio-separator-uvr.hf.space';
        this.onStatus = options.onStatus || (() => { });
        this.onProgress = options.onProgress || (() => { });
    }

    async separate(file, modelName = "MelBand Roformer Kim | Big Beta v5e FT by Unwa") {
        try {
            // 1. Upload File
            this.onStatus('Uploading audio to Hugging Face...');
            const uploadUrl = `${this.spaceUrl}/gradio_api/upload`;
            const formData = new FormData();
            formData.append('files', file);

            const uploadRes = await fetch(uploadUrl, {
                method: 'POST',
                body: formData
            });

            if (!uploadRes.ok) throw new Error(`Upload failed: ${uploadRes.statusText}`);
            const uploadData = await uploadRes.json();
            const tempFilePath = uploadData[0]; // HF returns an array of paths
            console.log("Uploaded to HF:", tempFilePath);

            // 2. Trigger Prediction (Using fn_index 5 for separation)
            const payload = {
                data: [
                    { path: tempFilePath, meta: { _type: "gradio.FileData" } },
                    modelName,
                    256,   // Segment Size
                    false, // Override Segment Size
                    8,     // Overlap
                    0,     // Pitch Shift
                    "",    // Model Directory (Empty for default)
                    "",    // Output Directory (Empty for default)
                    "wav", // Output Format
                    0.9,   // Normalization
                    0,     // Amplification
                    1,     // Batch Size
                    ""     // Rename Stems
                ]
            };

            this.onStatus('Queuing separation job (GPU)...');

            // Using the modern Gradio /call endpoint for SSE status
            const callUrl = `${this.spaceUrl}/gradio_api/call/5`;
            const callRes = await fetch(callUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!callRes.ok) throw new Error(`Trigger failed: ${callRes.statusText}`);
            const { event_id } = await callRes.json();
            console.log("Job Event ID:", event_id);

            // 3. Poll for Completion (SSE style)
            return await this.pollStatus(event_id);

        } catch (err) {
            console.error("HF Worker Error:", err);
            throw err;
        }
    }

    async pollStatus(eventId) {
        return new Promise((resolve, reject) => {
            const statusUrl = `${this.spaceUrl}/gradio_api/call/5/${eventId}`;
            const eventSource = new EventSource(statusUrl);

            eventSource.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                console.log("HF Event:", data.msg, data);

                switch (data.msg) {
                    case 'progress':
                        if (data.progress_data && data.progress_data[0]) {
                            const prog = data.progress_data[0];
                            this.onProgress(prog.index / prog.total, `Processing: ${prog.unit || ''}`);
                        }
                        break;
                    case 'process_started':
                        this.onStatus('Separation started on T4 GPU...');
                        break;
                    case 'process_completed':
                        eventSource.close();
                        if (data.success) {
                            const results = data.output.data;
                            // Index 0: Vocals File Info
                            // Index 1: Instrumental File Info
                            // Index 2: Message/Log
                            resolve(await this.processResults(results));
                        } else {
                            reject(new Error("HF Processing failed: " + (data.output.error || "Unknown error")));
                        }
                        break;
                    case 'heartbeat':
                        break;
                    case 'error':
                        eventSource.close();
                        reject(new Error("HF Stream Error: " + data.error));
                        break;
                }
            };

            eventSource.onerror = (err) => {
                eventSource.close();
                reject(new Error("HF Connection interrupted"));
            };
        });
    }

    async processResults(data) {
        this.onStatus('Downloading results...');

        // Data[0] and Data[1] are typically objects with 'url' and 'orig_name'
        const vocalsInfo = data[0];
        const instInfo = data[1];

        // Result URLs are relative to the space if not absolute
        const getFullUrl = (info) => {
            if (!info || !info.url) return null;
            return info.url.startsWith('http') ? info.url : `${this.spaceUrl}/file=${info.path}`;
        };

        const vUrl = getFullUrl(vocalsInfo);
        const iUrl = getFullUrl(instInfo);

        if (!vUrl || !iUrl) throw new Error("Could not find result URLs in HF output");

        // Fetch and convert to Audio Data (Waveform) to keep system consistent
        const [vBlob, iBlob] = await Promise.all([
            fetch(vUrl).then(r => r.blob()),
            fetch(iUrl).then(r => r.blob())
        ]);

        return {
            vocalsBlob: vBlob,
            instrumentalBlob: iBlob,
            vocalsUrl: URL.createObjectURL(vBlob),
            instrumentalUrl: URL.createObjectURL(iBlob)
        };
    }
}
