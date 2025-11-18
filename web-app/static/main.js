const recordBtn = document.getElementById("record");
const recordText = document.getElementById("record-text");
const uploadBtn = document.getElementById("upload");
const fileInput = document.getElementById("audioFile");
const titleText = document.getElementById("main-title");

let mediaRecorder = null;
let chunks = [];
let isRecording = false;

// send audio to Flask
async function sendToServer(blob, filename = "recording.wav"){
    const formData = new FormData();
    formData.append("audio", blob, filename);

    titleText.textContent = "Detecting...";

    try {
        const res = await fetch("/api/classify-upload", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        if(!res.ok || data.error){
            titleText.textContent = "Error";
            console.error("Server error:", data.error || "Unknown error");
            return;
        }

        // update ui
        const inst = data.instrument || "unknown";
        const conf = (data.confidence * 100).toFixed(1);
        titleText.textContent = `${inst} (${conf}%)`;
    } catch (err){
        console.error("Upload failed:", err);
        titleText.textContent = "Error";
    }
}

// recording
recordBtn.addEventListener("click", async () => {
    if(!isRecording){
        // start recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({audio: true});
            // Use the browser's preferred audio format (usually webm)
            const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' :
                           MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' :
                           MediaRecorder.isTypeSupported('audio/ogg') ? 'audio/ogg' : '';
            
            mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
            chunks = [];

            mediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) {
                    chunks.push(e.data);
                }
            };
            
            mediaRecorder.onstop = async () => {
                if (chunks.length === 0) {
                    titleText.textContent = "Error: No audio recorded";
                    console.error("No audio chunks recorded");
                    return;
                }
                
                const blob = new Blob(chunks, {type: mediaRecorder.mimeType || 'audio/webm'});
                
                if (blob.size === 0) {
                    titleText.textContent = "Error: Empty recording";
                    console.error("Recorded blob is empty");
                    return;
                }
                
                // Use appropriate file extension based on MIME type
                const ext = blob.type.includes('webm') ? '.webm' : 
                           blob.type.includes('ogg') ? '.ogg' : '.wav';
                console.log(`Sending audio: ${blob.size} bytes, type: ${blob.type}, ext: ${ext}`);
                await sendToServer(blob, `recording${ext}`);
            };

            // Request data periodically to avoid empty chunks
            mediaRecorder.start(100); // Request data every 100ms
            isRecording = true;
            recordText.textContent = "Stop";
            titleText.textContent = "Listening...";
        } catch (err) {
            console.error("Mic error:", err);
            titleText.textContent = "Error: Mic access denied";
        }
    } else {
        // stop recording
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            // Stop all tracks to release microphone
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        isRecording = false;
        recordText.textContent = "Record";
        titleText.textContent = "Processing..."
    }
    
}); 

// upload 
uploadBtn.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", async () => {
    if (fileInput.files.length === 0){
        return;
    }
    const file = fileInput.files[0];
    titleText.textContent = "Processing...";
    await sendToServer(file);
})
