const recordBtn = document.getElementById("record");
const recordText = document.getElementById("record-text");
const uploadBtn = document.getElementById("upload");
const fileInput = document.getElementById("audioFile");
const titleText = document.getElementById("main-title");

let mediaRecorder = null;
let chunks = [];
let isRecording = false;

// send audio to Flask
async function sendToServer(blob){
    const formData = new FormData();
    formData.append("audio", blob, "recording.wav");

    titleText.textContent = "Detecting...";

    try {
        const res = await fetch("/api/classify-upload", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        if(!res.ok || data.error){
            titleText.textContent = "Error";
            console.error("Server error:", data.error);
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
            mediaRecorder = new MediaRecorder(stream);
            chunks = [];

            mediaRecorder.ondataavailable = e => chunks.push(e.data);
            mediaRecorder.onstop = async () => {
                const blob = new Blob(chunks, {type: "audio/wav"});
                await sendToServer(blob)  
            };

            mediaRecorder.start();
            isRecording = true;
            recordText.textContent = "Stop";
            titleText.textContent = "Listening...";
        } catch (err) {
            console.error("Mic error:", err);
        }
    } else {
        // stop recording
        mediaRecorder.stop()
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
