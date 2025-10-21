let isMonitoring = true;

function startMonitoring() {
    document.getElementById("video-stream").src = "/video_feed";
    document.getElementById("driver-status").innerText = "Monitoring...";
    isMonitoring = true;
}

function stopMonitoring() {
    document.getElementById("video-stream").src = "";
    document.getElementById("driver-status").innerText = "Stopped";
    isMonitoring = false;
}
